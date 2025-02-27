from __future__ import annotations
import asyncio
import logging
import os
import io
from uuid import uuid4
from telegram import BotCommandScopeAllGroupChats, Update, constants
from telegram import (
    InlineKeyboardMarkup,
    InlineKeyboardButton,
    InlineQueryResultArticle,
)
from telegram import InputTextMessageContent, BotCommand
from telegram.error import RetryAfter, TimedOut, BadRequest
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    filters,
    InlineQueryHandler,
    CallbackQueryHandler,
    Application,
    ContextTypes,
    CallbackContext,
)
from pydub import AudioSegment
from datetime import datetime
from PIL import Image
from utils import (
    is_group_chat,
    get_thread_id,
    message_text,
    wrap_with_indicator,
    split_into_chunks,
    edit_message_with_retry,
    get_stream_cutoff_values,
    is_allowed,
    get_remaining_budget,
    is_admin,
    is_within_budget,
    get_reply_to_message_id,
    add_chat_request_to_usage_tracker,
    error_handler,
    is_direct_result,
    handle_direct_result,
    cleanup_intermediate_files,
    encode_image,
)
from openai_helper import OpenAIHelper, localized_text
from usage_tracker import UsageTracker
import database


class ChatGPTTelegramBot:
    def __init__(self, config: dict, openai: OpenAIHelper):
        self.config = config
        self.openai = openai
        bot_language = self.config["bot_language"]
        self.commands = [
            BotCommand(
                command="help",
                description=localized_text("help_description", bot_language),
            ),
            BotCommand(
                command="myid",
                description="Affiche votre ID et nom d'utilisateur"
            ),
            BotCommand(
                command="reset",
                description=localized_text("reset_description", bot_language),
            ),
            BotCommand(
                command="stats",
                description=localized_text("stats_description", bot_language),
            ),
            BotCommand(
                command="resend",
                description=localized_text("resend_description", bot_language),
            ),
            BotCommand(
                command="subscribe", description="Check or manage your subscription"
            ),
            BotCommand(
                command="status", description="View your current subscription status"
            ),
            BotCommand(command="upgrade", description="Upgrade to a Premium plan"),
        ]
        if self.config.get("enable_image_generation", False):
            self.commands.append(
                BotCommand(
                    command="image",
                    description=localized_text("image_description", bot_language),
                )
            )
        if self.config.get("enable_tts_generation", False):
            self.commands.append(
                BotCommand(
                    command="tts",
                    description=localized_text("tts_description", bot_language),
                )
            )

        self.group_commands = [
            BotCommand(
                command="chat",
                description=localized_text("chat_description", bot_language),
            )
        ] + self.commands
        self.disallowed_message = localized_text("disallowed", bot_language)
        self.budget_limit_message = localized_text("budget_limit", bot_language)
        self.usage = {}
        # Pré-initialiser usage avec les utilisateurs existants
        logs_dir = "usage_logs"
        if os.path.exists(logs_dir):
            for filename in os.listdir(logs_dir):
                if filename.endswith(".json"):
                    user_id = filename[:-5]  # Enlever ".json"
                    try:
                        self.usage[user_id] = UsageTracker(user_id, "unknown")
                    except Exception as e:
                        logging.warning(f"Failed to preload UsageTracker for user {user_id}: {str(e)}")
        self.last_message = {}
        self.inline_queries_cache = {}
        database.init_db()

    async def check_subscription(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE, is_inline=False
    ) -> tuple[str, int] | None:
        user_id = str(
            update.inline_query.from_user.id
            if is_inline
            else update.message.from_user.id
        )
        status = database.get_subscriber_status(user_id)
        if not status:
            database.add_or_update_subscriber(user_id, "free", full_name=user.full_name)
            status = ("free", None, 0, datetime.now().date().isoformat(), user.full_name)
        plan, end_date, message_count, last_reset_date, full_name = status
        if plan == "premium" and end_date and datetime.now().isoformat() > end_date:
            database.add_or_update_subscriber(user_id, "free")
            if not is_inline:
                await update.effective_message.reply_text(
                    "Your subscription has expired. You are now on the free plan."
                )
            return "free", 0
        if plan == "free" and message_count >= self.config["free_message_limit"]:
            if not is_inline:
                await update.effective_message.reply_text(
                    f"You've reached the daily limit ({self.config['free_message_limit']} messages). Upgrade with /upgrade."
                )
            return None
        return plan, message_count

    async def my_id(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Commande /myID pour afficher l'ID et le nom de l'utilisateur"""
        user = update.effective_user
        if is_group_chat(update):
            # Dans les groupes, affiche l'ID de l'utilisateur qui a envoyé la commande
            replied_user = update.message.reply_to_message.from_user if update.message.reply_to_message else user
            response = f"👤 ID : {replied_user.id}\n🪪 Nom : {replied_user.full_name}"
        else:
            # En privé, affiche les infos de l'utilisateur actuel
            response = (
                f"🔑 Votre ID Telegram : {user.id}\n"
                f"👤 Votre nom : {user.full_name}\n\n"
                f"⚠️ Ne partagez jamais votre ID avec des inconnus !"
            )
    
        await update.message.reply_text(response)
    async def help(self, update: Update, _: ContextTypes.DEFAULT_TYPE):
        commands = self.group_commands if is_group_chat(update) else self.commands
        commands_description = [
            f"/{command.command} - {command.description}" for command in commands
        ]
        bot_language = self.config["bot_language"]
        help_text = (
            localized_text("help_text", bot_language)[0]
            + "\n\n"
            + "\n".join(commands_description)
            + "\n\n"
            + localized_text("help_text", bot_language)[1]
            + "\n\n"
            + localized_text("help_text", bot_language)[2]
        )
        await update.message.reply_text(help_text, disable_web_page_preview=True)

    async def stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not await self.check_allowed_and_within_budget(update, context):
            return
        user_id = update.message.from_user.id
        if user_id not in self.usage:
            self.usage[user_id] = UsageTracker(user_id, update.message.from_user.name)
        tokens_today, tokens_month = self.usage[user_id].get_current_token_usage()
        images_today, images_month = self.usage[user_id].get_current_image_count()
        (
            transcribe_minutes_today,
            transcribe_seconds_today,
            transcribe_minutes_month,
            transcribe_seconds_month,
        ) = self.usage[user_id].get_current_transcription_duration()
        vision_today, vision_month = self.usage[user_id].get_current_vision_tokens()
        characters_today, characters_month = self.usage[user_id].get_current_tts_usage()
        current_cost = self.usage[user_id].get_current_cost()
        messages_today = self.usage[user_id].get_current_message_count()
        chat_id = update.effective_chat.id
        chat_messages, chat_token_length = self.openai.get_conversation_stats(chat_id)
        remaining_budget = get_remaining_budget(self.config, self.usage, update)
        bot_language = self.config["bot_language"]

        text_current_conversation = (
            f"*{localized_text('stats_conversation', bot_language)[0]}*:\n"
            f"{chat_messages} {localized_text('stats_conversation', bot_language)[1]}\n"
            f"{chat_token_length} {localized_text('stats_conversation', bot_language)[2]}\n"
            "----------------------------\n"
        )

        text_today_images = (
            f"{images_today} {localized_text('stats_images', bot_language)}\n"
            if self.config.get("enable_image_generation", False)
            else ""
        )
        text_today_vision = (
            f"{vision_today} {localized_text('stats_vision', bot_language)}\n"
            if self.config.get("enable_vision", False)
            else ""
        )
        text_today_tts = (
            f"{characters_today} {localized_text('stats_tts', bot_language)}\n"
            if self.config.get("enable_tts_generation", False)
            else ""
        )
        text_today = (
            f"*{localized_text('usage_today', bot_language)}*:\n"
            f"{tokens_today} {localized_text('stats_tokens', bot_language)}\n"
            f"{text_today_images}{text_today_vision}{text_today_tts}"
            f"{transcribe_minutes_today} {localized_text('stats_transcribe', bot_language)[0]} "
            f"{transcribe_seconds_today} {localized_text('stats_transcribe', bot_language)[1]}\n"
            f"Messages: {messages_today}\n"
            f"{localized_text('stats_total', bot_language)}{current_cost['cost_today']:.2f}\n"
            "----------------------------\n"
        )

        text_month_images = (
            f"{images_month} {localized_text('stats_images', bot_language)}\n"
            if self.config.get("enable_image_generation", False)
            else ""
        )
        text_month_vision = (
            f"{vision_month} {localized_text('stats_vision', bot_language)}\n"
            if self.config.get("enable_vision", False)
            else ""
        )
        text_month_tts = (
            f"{characters_month} {localized_text('stats_tts', bot_language)}\n"
            if self.config.get("enable_tts_generation", False)
            else ""
        )
        text_month = (
            f"*{localized_text('usage_month', bot_language)}*:\n"
            f"{tokens_month} {localized_text('stats_tokens', bot_language)}\n"
            f"{text_month_images}{text_month_vision}{text_month_tts}"
            f"{transcribe_minutes_month} {localized_text('stats_transcribe', bot_language)[0]} "
            f"{transcribe_seconds_month} {localized_text('stats_transcribe', bot_language)[1]}\n"
            f"{localized_text('stats_total', bot_language)}{current_cost['cost_month']:.2f}"
        )

        text_budget = "\n\n" + (
            f"{localized_text('stats_budget', bot_language)}{localized_text(self.config['budget_period'], bot_language)}: ${remaining_budget:.2f}.\n"
            if remaining_budget < float("inf")
            else ""
        )

        usage_text = text_current_conversation + text_today + text_month + text_budget
        await update.message.reply_text(
            usage_text, parse_mode=constants.ParseMode.MARKDOWN
        )

    async def resend(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not await self.check_allowed_and_within_budget(update, context):
            return
        chat_id = update.effective_chat.id
        if chat_id not in self.last_message:
            await update.effective_message.reply_text(
                message_thread_id=get_thread_id(update),
                text=localized_text("resend_failed", self.config["bot_language"]),
            )
            return
        with update.message._unfrozen() as message:
            message.text = self.last_message.pop(chat_id)
        await self.prompt(update=update, context=context)

    async def reset(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not await self.check_allowed_and_within_budget(update, context):
            return
        chat_id = update.effective_chat.id
        reset_content = message_text(update.message)
        self.openai.reset_chat_history(chat_id=chat_id, content=reset_content)
        await update.effective_message.reply_text(
            message_thread_id=get_thread_id(update),
            text=localized_text("reset_done", self.config["bot_language"]),
        )

    async def image(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self.config[
            "enable_image_generation"
        ] or not await self.check_allowed_and_within_budget(update, context):
            return
        subscription_result = await self.check_subscription(update, context)
        if subscription_result is None:
            return
        plan, _ = subscription_result
        image_query = message_text(update.message)
        if not image_query:
            await update.effective_message.reply_text(
                message_thread_id=get_thread_id(update),
                text=localized_text("image_no_prompt", self.config["bot_language"]),
            )
            return
        logging.info(
            f"Image request from {update.message.from_user.name} (id: {update.message.from_user.id})"
        )

        async def _generate():
            try:
                image_url, image_size = await self.openai.generate_image(
                    prompt=image_query, plan=plan
                )
                if self.config["image_receive_mode"] == "photo":
                    await update.effective_message.reply_photo(
                        reply_to_message_id=get_reply_to_message_id(
                            self.config, update
                        ),
                        photo=image_url,
                    )
                elif self.config["image_receive_mode"] == "document":
                    await update.effective_message.reply_document(
                        reply_to_message_id=get_reply_to_message_id(
                            self.config, update
                        ),
                        document=image_url,
                    )
                else:
                    raise Exception(
                        f"Invalid IMAGE_RECEIVE_MODE: {self.config['image_receive_mode']}"
                    )
                user_id = update.message.from_user.id
                self.usage[user_id].add_image_request(
                    image_size, self.config["image_prices"]
                )
                if (
                    str(user_id) not in self.config["allowed_user_ids"].split(",")
                    and "guests" in self.usage
                ):
                    self.usage["guests"].add_image_request(
                        image_size, self.config["image_prices"]
                    )
            except Exception as e:
                logging.exception(e)
                await update.effective_message.reply_text(
                    message_thread_id=get_thread_id(update),
                    text=f"{localized_text('image_fail', self.config['bot_language'])}: {str(e)}",
                    parse_mode=constants.ParseMode.MARKDOWN,
                )

        await wrap_with_indicator(
            update, context, _generate, constants.ChatAction.UPLOAD_PHOTO
        )

    async def tts(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self.config[
            "enable_tts_generation"
        ] or not await self.check_allowed_and_within_budget(update, context):
            return
        subscription_result = await self.check_subscription(update, context)
        if subscription_result is None:
            return
        plan, _ = subscription_result
        tts_query = message_text(update.message)
        if not tts_query:
            await update.effective_message.reply_text(
                message_thread_id=get_thread_id(update),
                text=localized_text("tts_no_prompt", self.config["bot_language"]),
            )
            return
        logging.info(
            f"TTS request from {update.message.from_user.name} (id: {update.message.from_user.id})"
        )

        async def _generate():
            try:
                speech_file, text_length = await self.openai.generate_speech(
                    text=tts_query
                )
                await update.effective_message.reply_voice(
                    reply_to_message_id=get_reply_to_message_id(self.config, update),
                    voice=speech_file,
                )
                speech_file.close()
                user_id = update.message.from_user.id
                self.usage[user_id].add_tts_request(
                    text_length, self.config["tts_model"], self.config["tts_prices"]
                )
                if (
                    str(user_id) not in self.config["allowed_user_ids"].split(",")
                    and "guests" in self.usage
                ):
                    self.usage["guests"].add_tts_request(
                        text_length, self.config["tts_model"], self.config["tts_prices"]
                    )
            except Exception as e:
                logging.exception(e)
                await update.effective_message.reply_text(
                    message_thread_id=get_thread_id(update),
                    text=f"{localized_text('tts_fail', self.config['bot_language'])}: {str(e)}",
                    parse_mode=constants.ParseMode.MARKDOWN,
                )

        await wrap_with_indicator(
            update, context, _generate, constants.ChatAction.UPLOAD_VOICE
        )

    async def transcribe(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self.config[
            "enable_transcription"
        ] or not await self.check_allowed_and_within_budget(update, context):
            return
        subscription_result = await self.check_subscription(update, context)
        if subscription_result is None:
            return
        plan, _ = subscription_result
        if is_group_chat(update) and self.config["ignore_group_transcriptions"]:
            logging.info("Transcription from group chat, ignoring...")
            return
        chat_id = update.effective_chat.id
        filename = update.message.effective_attachment.file_unique_id

        async def _execute():
            filename_mp3 = f"{filename}.mp3"
            bot_language = self.config["bot_language"]
            try:
                media_file = await context.bot.get_file(
                    update.message.effective_attachment.file_id
                )
                await media_file.download_to_drive(filename)
            except Exception as e:
                logging.exception(e)
                await update.effective_message.reply_text(
                    message_thread_id=get_thread_id(update),
                    text=f"{localized_text('media_download_fail', bot_language)[0]}: {str(e)}",
                    parse_mode=constants.ParseMode.MARKDOWN,
                )
                return

            try:
                audio_track = AudioSegment.from_file(filename)
                audio_track.export(filename_mp3, format="mp3")
                logging.info(
                    f"Transcribe request from {update.message.from_user.name} (id: {update.message.from_user.id})"
                )
            except Exception as e:
                logging.exception(e)
                await update.effective_message.reply_text(
                    message_thread_id=get_thread_id(update),
                    text=localized_text("media_type_fail", bot_language),
                )
                if os.path.exists(filename):
                    os.remove(filename)
                return

            user_id = update.message.from_user.id
            if user_id not in self.usage:
                self.usage[user_id] = UsageTracker(
                    user_id, update.message.from_user.name
                )

            try:
                transcript = await self.openai.transcribe(filename_mp3)
                transcription_price = self.config["transcription_price"]
                self.usage[user_id].add_transcription_seconds(
                    audio_track.duration_seconds, transcription_price
                )
                if (
                    str(user_id) not in self.config["allowed_user_ids"].split(",")
                    and "guests" in self.usage
                ):
                    self.usage["guests"].add_transcription_seconds(
                        audio_track.duration_seconds, transcription_price
                    )

                response_to_transcription = any(
                    transcript.lower().startswith(prefix.lower()) if prefix else False
                    for prefix in self.config["voice_reply_prompts"]
                )

                if (
                    self.config["voice_reply_transcript"]
                    and not response_to_transcription
                ):
                    transcript_output = f"_{localized_text('transcript', bot_language)}:_\n\"{transcript}\""
                    chunks = split_into_chunks(transcript_output)
                    for index, chunk in enumerate(chunks):
                        await update.effective_message.reply_text(
                            message_thread_id=get_thread_id(update),
                            reply_to_message_id=(
                                get_reply_to_message_id(self.config, update)
                                if index == 0
                                else None
                            ),
                            text=chunk,
                            parse_mode=constants.ParseMode.MARKDOWN,
                        )
                else:
                    response, total_tokens = await self.openai.get_chat_response(
                        chat_id=chat_id, query=transcript, plan=plan
                    )
                    self.usage[user_id].add_chat_tokens(
                        total_tokens, self.config["token_price"]
                    )
                    if (
                        str(user_id) not in self.config["allowed_user_ids"].split(",")
                        and "guests" in self.usage
                    ):
                        self.usage["guests"].add_chat_tokens(
                            total_tokens, self.config["token_price"]
                        )
                    transcript_output = (
                        f"_{localized_text('transcript', bot_language)}:_\n\"{transcript}\"\n\n"
                        f"_{localized_text('answer', bot_language)}:_\n{response}"
                    )
                    chunks = split_into_chunks(transcript_output)
                    for index, chunk in enumerate(chunks):
                        await update.effective_message.reply_text(
                            message_thread_id=get_thread_id(update),
                            reply_to_message_id=(
                                get_reply_to_message_id(self.config, update)
                                if index == 0
                                else None
                            ),
                            text=chunk,
                            parse_mode=constants.ParseMode.MARKDOWN,
                        )
            except Exception as e:
                logging.exception(e)
                await update.effective_message.reply_text(
                    message_thread_id=get_thread_id(update),
                    text=f"{localized_text('transcribe_fail', bot_language)}: {str(e)}",
                    parse_mode=constants.ParseMode.MARKDOWN,
                )
            finally:
                if os.path.exists(filename_mp3):
                    os.remove(filename_mp3)
                if os.path.exists(filename):
                    os.remove(filename)

        await wrap_with_indicator(
            update, context, _execute, constants.ChatAction.TYPING
        )

    async def vision(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self.config[
            "enable_vision"
        ] or not await self.check_allowed_and_within_budget(update, context):
            return
        subscription_result = await self.check_subscription(update, context)
        if subscription_result is None:
            return
        plan, _ = subscription_result
        chat_id = update.effective_chat.id
        prompt = update.message.caption
        if is_group_chat(update):
            if self.config["ignore_group_vision"]:
                logging.info("Vision from group chat, ignoring...")
                return
            trigger_keyword = self.config["group_trigger_keyword"]
            if (prompt is None and trigger_keyword) or (
                prompt and not prompt.lower().startswith(trigger_keyword.lower())
            ):
                logging.info("Vision from group chat with wrong keyword, ignoring...")
                return
        image = update.message.effective_attachment[-1]

        async def _execute():
            bot_language = self.config["bot_language"]
            try:
                media_file = await context.bot.get_file(image.file_id)
                temp_file = io.BytesIO(await media_file.download_as_bytearray())
            except Exception as e:
                logging.exception(e)
                await update.effective_message.reply_text(
                    message_thread_id=get_thread_id(update),
                    text=f"{localized_text('media_download_fail', bot_language)[0]}: {str(e)}",
                    parse_mode=constants.ParseMode.MARKDOWN,
                )
                return

            temp_file_png = io.BytesIO()
            try:
                original_image = Image.open(temp_file)
                original_image.save(temp_file_png, format="PNG")
                logging.info(
                    f"Vision request from {update.message.from_user.name} (id: {update.message.from_user.id})"
                )
            except Exception as e:
                logging.exception(e)
                await update.effective_message.reply_text(
                    message_thread_id=get_thread_id(update),
                    text=localized_text("media_type_fail", bot_language),
                )
                return

            user_id = update.message.from_user.id
            if user_id not in self.usage:
                self.usage[user_id] = UsageTracker(
                    user_id, update.message.from_user.name
                )

            if self.config["stream"]:
                stream_response = self.openai.interpret_image_stream(
                    chat_id=chat_id, fileobj=temp_file_png, prompt=prompt, plan=plan
                )
                i = 0
                prev = ""
                sent_message = None
                backoff = 0
                stream_chunk = 0
                async for content, tokens in stream_response:
                    if is_direct_result(content):
                        return await handle_direct_result(self.config, update, content)
                    if not content.strip():
                        continue
                    stream_chunks = split_into_chunks(content)
                    if len(stream_chunks) > 1:
                        content = stream_chunks[-1]
                        if stream_chunk != len(stream_chunks) - 1:
                            stream_chunk += 1
                            try:
                                await edit_message_with_retry(
                                    context,
                                    chat_id,
                                    str(sent_message.message_id),
                                    stream_chunks[-2],
                                )
                            except:
                                pass
                            try:
                                sent_message = (
                                    await update.effective_message.reply_text(
                                        message_thread_id=get_thread_id(update),
                                        text=content or "...",
                                    )
                                )
                            except:
                                pass
                            continue
                    cutoff = get_stream_cutoff_values(update, content) + backoff
                    if i == 0:
                        try:
                            if sent_message:
                                await context.bot.delete_message(
                                    chat_id=sent_message.chat_id,
                                    message_id=sent_message.message_id,
                                )
                            sent_message = await update.effective_message.reply_text(
                                message_thread_id=get_thread_id(update),
                                reply_to_message_id=get_reply_to_message_id(
                                    self.config, update
                                ),
                                text=content,
                            )
                        except:
                            continue
                    elif (
                        abs(len(content) - len(prev)) > cutoff
                        or tokens != "not_finished"
                    ):
                        prev = content
                        try:
                            use_markdown = tokens != "not_finished"
                            await edit_message_with_retry(
                                context,
                                chat_id,
                                str(sent_message.message_id),
                                content,
                                markdown=use_markdown,
                            )
                        except RetryAfter as e:
                            backoff += 5
                            await asyncio.sleep(e.retry_after)
                            continue
                        except TimedOut:
                            backoff += 5
                            await asyncio.sleep(0.5)
                            continue
                        except Exception:
                            backoff += 5
                            continue
                        await asyncio.sleep(0.01)
                    i += 1
                    if tokens != "not_finished":
                        total_tokens = int(tokens)
                        vision_token_price = self.config["vision_token_price"]
                        self.usage[user_id].add_vision_tokens(
                            total_tokens, vision_token_price
                        )
                        if (
                            str(user_id)
                            not in self.config["allowed_user_ids"].split(",")
                            and "guests" in self.usage
                        ):
                            self.usage["guests"].add_vision_tokens(
                                total_tokens, vision_token_price
                            )
            else:
                try:
                    interpretation, total_tokens = await self.openai.interpret_image(
                        chat_id, temp_file_png, prompt=prompt, plan=plan
                    )
                    try:
                        await update.effective_message.reply_text(
                            message_thread_id=get_thread_id(update),
                            reply_to_message_id=get_reply_to_message_id(
                                self.config, update
                            ),
                            text=interpretation,
                            parse_mode=constants.ParseMode.MARKDOWN,
                        )
                    except BadRequest:
                        await update.effective_message.reply_text(
                            message_thread_id=get_thread_id(update),
                            reply_to_message_id=get_reply_to_message_id(
                                self.config, update
                            ),
                            text=interpretation,
                        )
                    vision_token_price = self.config["vision_token_price"]
                    self.usage[user_id].add_vision_tokens(
                        total_tokens, vision_token_price
                    )
                    if (
                        str(user_id) not in self.config["allowed_user_ids"].split(",")
                        and "guests" in self.usage
                    ):
                        self.usage["guests"].add_vision_tokens(
                            total_tokens, vision_token_price
                        )
                except Exception as e:
                    logging.exception(e)
                    await update.effective_message.reply_text(
                        message_thread_id=get_thread_id(update),
                        text=f"{localized_text('vision_fail', bot_language)}: {str(e)}",
                        parse_mode=constants.ParseMode.MARKDOWN,
                    )

        await wrap_with_indicator(
            update, context, _execute, constants.ChatAction.TYPING
        )

    async def prompt(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not await self.check_allowed_and_within_budget(update, context):
            return
        if update.edited_message or not update.message or update.message.via_bot:
            return
        subscription_result = await self.check_subscription(update, context)
        if subscription_result is None:
            return
        plan, _ = subscription_result
        logging.info(
            f"Message from {update.message.from_user.name} (id: {update.message.from_user.id})"
        )
        chat_id = update.effective_chat.id
        user_id = update.message.from_user.id
        prompt = message_text(update.message)
        self.last_message[chat_id] = prompt

        if is_group_chat(update):
            trigger_keyword = self.config["group_trigger_keyword"]
            if prompt.lower().startswith(
                trigger_keyword.lower()
            ) or update.message.text.lower().startswith("/chat"):
                if prompt.lower().startswith(trigger_keyword.lower()):
                    prompt = prompt[len(trigger_keyword) :].strip()
                if (
                    update.message.reply_to_message
                    and update.message.reply_to_message.text
                    and update.message.reply_to_message.from_user.id != context.bot.id
                ):
                    prompt = f'"{update.message.reply_to_message.text}" {prompt}'
            else:
                if (
                    update.message.reply_to_message
                    and update.message.reply_to_message.from_user.id == context.bot.id
                ):
                    logging.info("Message is a reply to the bot, allowing...")
                else:
                    logging.warning(
                        "Message does not start with trigger keyword, ignoring..."
                    )
                    return

        try:
            total_tokens = 0
            if self.config["stream"]:
                await update.effective_message.reply_chat_action(
                    action=constants.ChatAction.TYPING,
                    message_thread_id=get_thread_id(update),
                )
                stream_response = self.openai.get_chat_response_stream(
                    chat_id=chat_id, query=prompt, plan=plan
                )
                i = 0
                prev = ""
                sent_message = None
                backoff = 0
                stream_chunk = 0
                async for content, tokens in stream_response:
                    if is_direct_result(content):
                        return await handle_direct_result(self.config, update, content)
                    if not content.strip():
                        continue
                    stream_chunks = split_into_chunks(content)
                    if len(stream_chunks) > 1:
                        content = stream_chunks[-1]
                        if stream_chunk != len(stream_chunks) - 1:
                            stream_chunk += 1
                            try:
                                await edit_message_with_retry(
                                    context,
                                    chat_id,
                                    str(sent_message.message_id),
                                    stream_chunks[-2],
                                )
                            except:
                                pass
                            try:
                                sent_message = (
                                    await update.effective_message.reply_text(
                                        message_thread_id=get_thread_id(update),
                                        text=content or "...",
                                    )
                                )
                            except:
                                pass
                            continue
                    cutoff = get_stream_cutoff_values(update, content) + backoff
                    if i == 0:
                        try:
                            if sent_message:
                                await context.bot.delete_message(
                                    chat_id=sent_message.chat_id,
                                    message_id=sent_message.message_id,
                                )
                            sent_message = await update.effective_message.reply_text(
                                message_thread_id=get_thread_id(update),
                                reply_to_message_id=get_reply_to_message_id(
                                    self.config, update
                                ),
                                text=content,
                            )
                        except:
                            continue
                    elif (
                        abs(len(content) - len(prev)) > cutoff
                        or tokens != "not_finished"
                    ):
                        prev = content
                        try:
                            use_markdown = tokens != "not_finished"
                            await edit_message_with_retry(
                                context,
                                chat_id,
                                str(sent_message.message_id),
                                content,
                                markdown=use_markdown,
                            )
                        except RetryAfter as e:
                            backoff += 5
                            await asyncio.sleep(e.retry_after)
                            continue
                        except TimedOut:
                            backoff += 5
                            await asyncio.sleep(0.5)
                            continue
                        except Exception:
                            backoff += 5
                            continue
                        await asyncio.sleep(0.01)
                    i += 1
                    if tokens != "not_finished":
                        total_tokens = int(tokens)
            else:

                async def _reply():
                    nonlocal total_tokens
                    response, total_tokens = await self.openai.get_chat_response(
                        chat_id=chat_id, query=prompt, plan=plan
                    )
                    if is_direct_result(response):
                        return await handle_direct_result(self.config, update, response)
                    chunks = split_into_chunks(response)
                    for index, chunk in enumerate(chunks):
                        try:
                            await update.effective_message.reply_text(
                                message_thread_id=get_thread_id(update),
                                reply_to_message_id=(
                                    get_reply_to_message_id(self.config, update)
                                    if index == 0
                                    else None
                                ),
                                text=chunk,
                                parse_mode=constants.ParseMode.MARKDOWN,
                            )
                        except Exception:
                            await update.effective_message.reply_text(
                                message_thread_id=get_thread_id(update),
                                reply_to_message_id=(
                                    get_reply_to_message_id(self.config, update)
                                    if index == 0
                                    else None
                                ),
                                text=chunk,
                            )

                await wrap_with_indicator(
                    update, context, _reply, constants.ChatAction.TYPING
                )
            add_chat_request_to_usage_tracker(
                self.usage, self.config, user_id, total_tokens
            )
        except Exception as e:
            logging.exception(e)
            await update.effective_message.reply_text(
                message_thread_id=get_thread_id(update),
                text=f"{localized_text('chat_fail', self.config['bot_language'])} {str(e)}",
                parse_mode=constants.ParseMode.MARKDOWN,
            )

    async def inline_query(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.inline_query.query
        if len(query) < 3:
            return
        if not await self.check_allowed_and_within_budget(
            update, context, is_inline=True
        ):
            return
        subscription_result = await self.check_subscription(
            update, context, is_inline=True
        )
        if subscription_result is None:
            return
        plan, _ = subscription_result
        callback_data_suffix = "gpt:"
        result_id = str(uuid4())
        self.inline_queries_cache[result_id] = query
        callback_data = f"{callback_data_suffix}{result_id}"
        await self.send_inline_query_result(
            update, result_id, message_content=query, callback_data=callback_data
        )

    async def send_inline_query_result(
        self, update: Update, result_id, message_content, callback_data=""
    ):
        try:
            reply_markup = None
            bot_language = self.config["bot_language"]
            if callback_data:
                reply_markup = InlineKeyboardMarkup(
                    [
                        [
                            InlineKeyboardButton(
                                text=f'🤖 {localized_text("answer_with_chatgpt", bot_language)}',
                                callback_data=callback_data,
                            )
                        ]
                    ]
                )
            inline_query_result = InlineQueryResultArticle(
                id=result_id,
                title=localized_text("ask_chatgpt", bot_language),
                input_message_content=InputTextMessageContent(message_content),
                description=message_content,
                thumbnail_url="https://user-images.githubusercontent.com/11541888/223106202-7576ff11-2c8e-408d-94ea-b02a7a32149a.png",
                reply_markup=reply_markup,
            )
            await update.inline_query.answer([inline_query_result], cache_time=0)
        except Exception as e:
            logging.error(f"Error generating inline query result: {e}")

    async def handle_callback_inline_query(
        self, update: Update, context: CallbackContext
    ):
        callback_data = update.callback_query.data
        user_id = update.callback_query.from_user.id
        inline_message_id = update.callback_query.inline_message_id
        name = update.callback_query.from_user.name
        callback_data_suffix = "gpt:"
        bot_language = self.config["bot_language"]
        answer_tr = localized_text("answer", bot_language)
        loading_tr = localized_text("loading", bot_language)

        try:
            if callback_data.startswith(callback_data_suffix):
                unique_id = callback_data.split(":")[1]
                total_tokens = 0
                query = self.inline_queries_cache.get(unique_id)
                if not query:
                    error_message = f'{localized_text("error", bot_language)}. {localized_text("try_again", bot_language)}'
                    await edit_message_with_retry(
                        context,
                        chat_id=None,
                        message_id=inline_message_id,
                        text=f"{query}\n\n_{answer_tr}:_\n{error_message}",
                        is_inline=True,
                    )
                    return
                self.inline_queries_cache.pop(unique_id)

                subscription_result = await self.check_subscription(
                    update, context, is_inline=True
                )
                if subscription_result is None:
                    return
                plan, _ = subscription_result

                unavailable_message = localized_text(
                    "function_unavailable_in_inline_mode", bot_language
                )
                if self.config["stream"]:
                    stream_response = self.openai.get_chat_response_stream(
                        chat_id=user_id, query=query, plan=plan
                    )
                    i = 0
                    prev = ""
                    backoff = 0
                    async for content, tokens in stream_response:
                        if is_direct_result(content):
                            await edit_message_with_retry(
                                context,
                                chat_id=None,
                                message_id=inline_message_id,
                                text=f"{query}\n\n_{answer_tr}:_\n{unavailable_message}",
                                is_inline=True,
                            )
                            return
                        if not content.strip():
                            continue
                        cutoff = get_stream_cutoff_values(update, content) + backoff
                        if i == 0:
                            await edit_message_with_retry(
                                context,
                                chat_id=None,
                                message_id=inline_message_id,
                                text=f"{query}\n\n{answer_tr}:\n{content}",
                                is_inline=True,
                            )
                        elif (
                            abs(len(content) - len(prev)) > cutoff
                            or tokens != "not_finished"
                        ):
                            prev = content
                            use_markdown = tokens != "not_finished"
                            divider = "_" if use_markdown else ""
                            text = (
                                f"{query}\n\n{divider}{answer_tr}:{divider}\n{content}"
                            )
                            text = text[:4096]  # Telegram inline limit
                            try:
                                await edit_message_with_retry(
                                    context,
                                    chat_id=None,
                                    message_id=inline_message_id,
                                    text=text,
                                    markdown=use_markdown,
                                    is_inline=True,
                                )
                            except RetryAfter as e:
                                backoff += 5
                                await asyncio.sleep(e.retry_after)
                                continue
                            except TimedOut:
                                backoff += 5
                                await asyncio.sleep(0.5)
                                continue
                        i += 1
                        if tokens != "not_finished":
                            total_tokens = int(tokens)
                else:

                    async def _send_inline():
                        nonlocal total_tokens
                        await context.bot.edit_message_text(
                            inline_message_id=inline_message_id,
                            text=f"{query}\n\n_{answer_tr}:_\n{loading_tr}",
                            parse_mode=constants.ParseMode.MARKDOWN,
                        )
                        response, total_tokens = await self.openai.get_chat_response(
                            chat_id=user_id, query=query, plan=plan
                        )
                        if is_direct_result(response):
                            await edit_message_with_retry(
                                context,
                                chat_id=None,
                                message_id=inline_message_id,
                                text=f"{query}\n\n_{answer_tr}:_\n{unavailable_message}",
                                is_inline=True,
                            )
                            return
                        text_content = f"{query}\n\n_{answer_tr}:_\n{response}"
                        text_content = text_content[:4096]  # Telegram inline limit
                        await edit_message_with_retry(
                            context,
                            chat_id=None,
                            message_id=inline_message_id,
                            text=text_content,
                            is_inline=True,
                        )

                    await wrap_with_indicator(
                        update,
                        context,
                        _send_inline,
                        constants.ChatAction.TYPING,
                        is_inline=True,
                    )
                add_chat_request_to_usage_tracker(
                    self.usage, self.config, user_id, total_tokens
                )
        except Exception as e:
            logging.error(f"Failed to handle inline query callback: {e}")
            await edit_message_with_retry(
                context,
                chat_id=None,
                message_id=inline_message_id,
                text=f"{query}\n\n_{answer_tr}:_\n{localized_text('chat_fail', bot_language)} {str(e)}",
                is_inline=True,
            )

    async def subscribe(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = str(update.message.from_user.id)
        full_name = update.message.from_user.full_name
        status = database.get_subscriber_status(user_id)
        if not status:
            # Ajouter l'utilisateur avec son nom complet
            database.add_or_update_subscriber(user_id, "free", full_name=full_name)
            status = ("free", None, 0, datetime.now().date().isoformat(), full_name)
        plan, end_date, message_count, last_reset_date, full_name = status

        if plan == "premium" and end_date and datetime.now().isoformat() < end_date:
            await update.message.reply_text(
                f"Vous êtes abonné au plan Premium jusqu'au {end_date}."
            )
        else:
            await update.message.reply_text(
                f"Plan actuel : {plan.capitalize()}. Passez à Premium avec /upgrade."
            )

    # --- Handler pour les Notifications ---
    async def send_notification(self, user_id: str, message: str):
        """Envoie une notification à un utilisateur"""
        await self.application.bot.send_message(chat_id=user_id, text=message)
    
    async def status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = str(update.message.from_user.id)
        status = database.get_subscriber_status(user_id)
        free_limit = self.config.get(
            "free_message_limit", 5
        )  # Valeur par défaut si manquant
        if status:
            plan, end_date, message_count, last_reset_date, full_name = status
            if plan == "premium":
                await update.message.reply_text(
                    f"Plan: Premium\nExpires: {end_date}\nMessages today: {message_count}"
                )
            else:
                await update.message.reply_text(
                    f"Plan: Free\nMessages today: {message_count}/{free_limit}"
                )
        else:
            await update.message.reply_text(
                f"Plan: Free\nMessages today: 0/{free_limit}"
            )

    async def upgrade(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            f"To upgrade to Premium (${self.config['subscription_price']} for {self.config['subscription_duration']} days), "
            "contact the admin or follow this link: [Payment Link]."
        )

    async def admin_list(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not is_admin(self.config, update.message.from_user.id):
            await update.message.reply_text("Commande réservée aux admins.")
            return
        subscribers = database.list_subscribers()
        if subscribers:
            response = "Liste des abonnés :\n"
            for user_id, plan, end_date, full_name in subscribers:
                response += (
                    f"👤 Nom : {full_name}\n"
                    f"🆔 ID : {user_id}\n"
                    f"📜 Plan : {plan}\n"
                    f"📅 Expire le : {end_date or 'N/A'}\n"
                    "――――――――――――――――――――\n"
                )
            await update.message.reply_text(response)
        else:
            await update.message.reply_text("Aucun abonné.")

    async def admin_add(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not is_admin(self.config, update.message.from_user.id):
            await update.message.reply_text("Commande réservée aux admins.")
            return
        try:
            args = context.args
            user_id, plan, duration = args[0], args[1].lower(), int(args[2])
            if plan not in ["free", "premium"]:
                raise ValueError
            # Récupérer le nom complet de l'utilisateur
            full_name = " ".join(args[3:]) if len(args) > 3 else "Inconnu"
            database.add_or_update_subscriber(
                user_id, plan, duration if plan == "premium" else None, full_name
            )
            await update.message.reply_text(
                f"Abonnement ajouté pour {user_id} ({full_name}) : {plan} ({duration} jours si premium)."
            )
        except (IndexError, ValueError):
            await update.message.reply_text(
                "Usage: /admin_add <user_id> <plan> <duration> <full_name>"
            )
    async def admin_remove(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Commande /admin_remove"""
        if not is_admin(self.config, update.message.from_user.id):
            await update.message.reply_text("Commande réservée aux admins.")
            return
        try:
            user_id = context.args[0]
            database.remove_subscriber(user_id)
            await update.message.reply_text(f"Abonnement supprimé pour {user_id}.")
        except IndexError:
            await update.message.reply_text("Usage: /admin_remove <user_id>")


    
    async def check_allowed_and_within_budget(self, update: Update, context: ContextTypes.DEFAULT_TYPE, is_inline=False) -> bool:
        name = update.inline_query.from_user.name if is_inline else update.message.from_user.name
        user_id = update.inline_query.from_user.id if is_inline else update.message.from_user.id
        subscription_result = await self.check_subscription(update, context, is_inline)
        if subscription_result is None:
            return False
        plan, _ = subscription_result

        if not await is_allowed(self.config, update, context, is_inline=is_inline):
            logging.warning(f'User {name} (id: {user_id}) is not allowed to use the bot')
            await self.send_disallowed_message(update, context, is_inline)
            return False
    
        if not is_within_budget(self.config, self.usage, update, is_inline=is_inline):
            logging.warning(f'User {name} (id: {user_id}) reached their usage limit')
            await self.send_budget_reached_message(update, context, is_inline)
            return False

        if not is_inline:
            message_text = update.message.text if update.message.text else "(non-text message)"
            excluded_commands = ["/status", "/upgrade", "/help", "/start", "/subscribe", "/admin_list", "/admin_add"]
            print(f"Checking message: '{message_text}' for user {user_id}")
            if message_text not in excluded_commands:
                print(f"Incrementing message count for user {user_id} with message: '{message_text}'")
                database.increment_message_count(str(user_id))
                if user_id not in self.usage:
                    self.usage[user_id] = UsageTracker(user_id, name)
                self.usage[user_id].add_message(plan)
            else:
                print(f"Message '{message_text}' from user {user_id} is an excluded command, not incrementing.")
    
        return True

    async def inline_query(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.inline_query.query
        if len(query) < 3 or not await self.check_allowed_and_within_budget(
            update, context, is_inline=True
        ):
            return
        subscription_result = await self.check_subscription(
            update, context, is_inline=True
        )
        if subscription_result is None:
            return
        plan, _ = subscription_result
        # Optimisation : limiter les inline queries pour les gratuits
        if (
            plan == "free" and len(query) > 50
        ):  # Limite arbitraire pour réduire les tokens
            result_id = str(uuid4())
            await self.send_inline_query_result(
                update,
                result_id,
                "Query too long for free plan. Upgrade with /upgrade.",
            )
            return
        callback_data_suffix = "gpt:"
        result_id = str(uuid4())
        self.inline_queries_cache[result_id] = query
        callback_data = f"{callback_data_suffix}{result_id}"
        await self.send_inline_query_result(
            update, result_id, message_content=query, callback_data=callback_data
        )

    async def send_inline_query_result(
        self, update: Update, result_id, message_content, callback_data=""
    ):
        try:
            reply_markup = None
            bot_language = self.config["bot_language"]
            if callback_data:
                reply_markup = InlineKeyboardMarkup(
                    [
                        [
                            InlineKeyboardButton(
                                text=f'🤖 {localized_text("answer_with_chatgpt", bot_language)}',
                                callback_data=callback_data,
                            )
                        ]
                    ]
                )
            inline_query_result = InlineQueryResultArticle(
                id=result_id,
                title=localized_text("ask_chatgpt", bot_language),
                input_message_content=InputTextMessageContent(message_content),
                description=message_content[
                    :100
                ],  # Limiter la description pour économiser
                thumbnail_url="https://user-images.githubusercontent.com/11541888/223106202-7576ff11-2c8e-408d-94ea-b02a7a32149a.png",
                reply_markup=reply_markup,
            )
            await update.inline_query.answer([inline_query_result], cache_time=0)
        except Exception as e:
            logging.error(f"Error generating inline query result: {e}")

    async def handle_callback_inline_query(
        self, update: Update, context: CallbackContext
    ):
        callback_data = update.callback_query.data
        user_id = update.callback_query.from_user.id
        inline_message_id = update.callback_query.inline_message_id
        name = update.callback_query.from_user.name
        callback_data_suffix = "gpt:"
        bot_language = self.config["bot_language"]
        answer_tr = localized_text("answer", bot_language)
        loading_tr = localized_text("loading", bot_language)

        if not callback_data.startswith(callback_data_suffix):
            return

        unique_id = callback_data.split(":")[1]
        query = self.inline_queries_cache.get(unique_id)
        if not query:
            await edit_message_with_retry(
                context,
                chat_id=None,
                message_id=inline_message_id,
                text=f"Error: Query not found. Try again.",
                is_inline=True,
            )
            return
        self.inline_queries_cache.pop(unique_id)

        subscription_result = await self.check_subscription(
            update, context, is_inline=True
        )
        if subscription_result is None:
            return
        plan, _ = subscription_result

        total_tokens = 0
        unavailable_message = localized_text(
            "function_unavailable_in_inline_mode", bot_language
        )
        if self.config["stream"]:
            stream_response = self.openai.get_chat_response_stream(
                chat_id=user_id, query=query, plan=plan
            )
            i = 0
            prev = ""
            backoff = 0
            async for content, tokens in stream_response:
                if is_direct_result(content):
                    await edit_message_with_retry(
                        context,
                        chat_id=None,
                        message_id=inline_message_id,
                        text=f"{query}\n\n_{answer_tr}:_\n{unavailable_message}",
                        is_inline=True,
                    )
                    return
                if not content.strip():
                    continue
                cutoff = get_stream_cutoff_values(update, content) + backoff
                if i == 0:
                    await edit_message_with_retry(
                        context,
                        chat_id=None,
                        message_id=inline_message_id,
                        text=f"{query}\n\n{answer_tr}:\n{content}",
                        is_inline=True,
                    )
                elif abs(len(content) - len(prev)) > cutoff or tokens != "not_finished":
                    prev = content
                    use_markdown = tokens != "not_finished"
                    text = f"{query}\n\n_{answer_tr}_:\n{content}"[
                        :4096
                    ]  # Limite Telegram
                    try:
                        await edit_message_with_retry(
                            context,
                            chat_id=None,
                            message_id=inline_message_id,
                            text=text,
                            markdown=use_markdown,
                            is_inline=True,
                        )
                    except RetryAfter as e:
                        backoff += 5
                        await asyncio.sleep(e.retry_after)
                    except TimedOut:
                        backoff += 5
                        await asyncio.sleep(0.5)
                i += 1
                if tokens != "not_finished":
                    total_tokens = int(tokens)
        else:

            async def _send_inline():
                nonlocal total_tokens
                await context.bot.edit_message_text(
                    inline_message_id=inline_message_id,
                    text=f"{query}\n\n_{answer_tr}:_\n{loading_tr}",
                    parse_mode=constants.ParseMode.MARKDOWN,
                )
                response, total_tokens = await self.openai.get_chat_response(
                    chat_id=user_id, query=query, plan=plan
                )
                if is_direct_result(response):
                    await edit_message_with_retry(
                        context,
                        chat_id=None,
                        message_id=inline_message_id,
                        text=f"{query}\n\n_{answer_tr}:_\n{unavailable_message}",
                        is_inline=True,
                    )
                    return
                text_content = f"{query}\n\n_{answer_tr}:_\n{response}"[:4096]
                await edit_message_with_retry(
                    context,
                    chat_id=None,
                    message_id=inline_message_id,
                    text=text_content,
                    is_inline=True,
                )

            await wrap_with_indicator(
                update,
                context,
                _send_inline,
                constants.ChatAction.TYPING,
                is_inline=True,
            )
        add_chat_request_to_usage_tracker(
            self.usage, self.config, user_id, total_tokens
        )

    async def send_disallowed_message(
        self, update: Update, _: ContextTypes.DEFAULT_TYPE, is_inline=False
    ):
        if not is_inline:
            await update.effective_message.reply_text(
                message_thread_id=get_thread_id(update),
                text=self.disallowed_message,
                disable_web_page_preview=True,
            )
        else:
            result_id = str(uuid4())
            await self.send_inline_query_result(
                update, result_id, message_content=self.disallowed_message
            )

    async def send_budget_reached_message(
        self, update: Update, _: ContextTypes.DEFAULT_TYPE, is_inline=False
    ):
        if not is_inline:
            await update.effective_message.reply_text(
                message_thread_id=get_thread_id(update), text=self.budget_limit_message
            )
        else:
            result_id = str(uuid4())
            await self.send_inline_query_result(
                update, result_id, message_content=self.budget_limit_message
            )

    async def post_init(self, application: Application):
        await application.bot.set_my_commands(
            self.group_commands, scope=BotCommandScopeAllGroupChats()
        )
        await application.bot.set_my_commands(self.commands)

    def run(self):
        application = (
            ApplicationBuilder()
            .token(self.config["token"])
            .proxy_url(self.config["proxy"])
            .get_updates_proxy_url(self.config["proxy"])
            .post_init(self.post_init)
            .concurrent_updates(True)
            .build()
        )

        application.add_handler(CommandHandler("reset", self.reset))
        application.add_handler(CommandHandler("help", self.help))
        application.add_handler(CommandHandler("myid", self.my_id))
        application.add_handler(CommandHandler("image", self.image))
        application.add_handler(CommandHandler("tts", self.tts))
        application.add_handler(CommandHandler("start", self.help))
        application.add_handler(CommandHandler("stats", self.stats))
        application.add_handler(CommandHandler("resend", self.resend))
        application.add_handler(CommandHandler("subscribe", self.subscribe))
        application.add_handler(CommandHandler("status", self.status))
        application.add_handler(CommandHandler("upgrade", self.upgrade))
        application.add_handler(CommandHandler("admin_list", self.admin_list))
        application.add_handler(CommandHandler("admin_add", self.admin_add))
        application.add_handler(CommandHandler("admin_remove", self.admin_remove))
        application.add_handler(
            CommandHandler(
                "chat",
                self.prompt,
                filters=filters.ChatType.GROUP | filters.ChatType.SUPERGROUP,
            )
        )
        application.add_handler(
            MessageHandler(filters.PHOTO | filters.Document.IMAGE, self.vision)
        )
        application.add_handler(
            MessageHandler(
                filters.AUDIO
                | filters.VOICE
                | filters.Document.AUDIO
                | filters.VIDEO
                | filters.VIDEO_NOTE
                | filters.Document.VIDEO,
                self.transcribe,
            )
        )
        application.add_handler(
            MessageHandler(filters.TEXT & (~filters.COMMAND), self.prompt)
        )
        application.add_handler(
            InlineQueryHandler(
                self.inline_query,
                chat_types=[
                    constants.ChatType.GROUP,
                    constants.ChatType.SUPERGROUP,
                    constants.ChatType.PRIVATE,
                ],
            )
        )
        application.add_handler(CallbackQueryHandler(self.handle_callback_inline_query))
        application.add_error_handler(error_handler)

        application.run_polling()
