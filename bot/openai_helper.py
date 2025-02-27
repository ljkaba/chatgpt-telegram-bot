from __future__ import annotations
import datetime
import logging
import os
import json
import httpx
import io
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
from utils import is_direct_result, encode_image, decode_image
from plugin_manager import PluginManager
import openai
import tiktoken

# Modèles OpenAI
GPT_3_MODELS = ("gpt-3.5-turbo", "gpt-3.5-turbo-0301", "gpt-3.5-turbo-0613")
GPT_3_16K_MODELS = ("gpt-3.5-turbo-16k", "gpt-3.5-turbo-16k-0613", "gpt-3.5-turbo-1106", "gpt-3.5-turbo-0125")
GPT_4_MODELS = ("gpt-4", "gpt-4-0314", "gpt-4-0613", "gpt-4-turbo-preview")
GPT_4_32K_MODELS = ("gpt-4-32k", "gpt-4-32k-0314", "gpt-4-32k-0613")
GPT_4_VISION_MODELS = ("gpt-4o",)
GPT_4_128K_MODELS = ("gpt-4-1106-preview", "gpt-4-0125-preview", "gpt-4-turbo-preview", "gpt-4-turbo", "gpt-4-turbo-2024-04-09")
GPT_4O_MODELS = ("gpt-4o", "gpt-4o-mini", "chatgpt-4o-latest")
O_MODELS = ("o1", "o1-mini", "o1-preview")
GPT_ALL_MODELS = GPT_3_MODELS + GPT_3_16K_MODELS + GPT_4_MODELS + GPT_4_32K_MODELS + GPT_4_VISION_MODELS + GPT_4_128K_MODELS + GPT_4O_MODELS + O_MODELS

def default_max_tokens(model: str) -> int:
    base = 1200
    if model in GPT_3_MODELS:
        return base
    elif model in GPT_4_MODELS:
        return base * 2
    elif model in GPT_3_16K_MODELS:
        return base * 4 if model != "gpt-3.5-turbo-1106" else 4096
    elif model in GPT_4_32K_MODELS:
        return base * 8
    elif model in GPT_4_VISION_MODELS or model in GPT_4_128K_MODELS or model in GPT_4O_MODELS or model in O_MODELS:
        return 4096

def are_functions_available(model: str) -> bool:
    if model in ("gpt-3.5-turbo-0301", "gpt-4-0314", "gpt-4-32k-0314", "gpt-3.5-turbo-0613", "gpt-3.5-turbo-16k-0613") or model in O_MODELS:
        return False
    return True

# Chargement des traductions
parent_dir_path = os.path.join(os.path.dirname(__file__), os.pardir)
translations_file_path = os.path.join(parent_dir_path, 'translations.json')
with open(translations_file_path, 'r', encoding='utf-8') as f:
    translations = json.load(f)

def localized_text(key, bot_language):
    try:
        return translations[bot_language][key]
    except KeyError:
        logging.warning(f"No translation available for bot_language code '{bot_language}' and key '{key}'")
        return translations['en'].get(key, key)

class OpenAIHelper:
    def __init__(self, config: dict, plugin_manager: PluginManager):
        http_client = httpx.AsyncClient(proxy=config['proxy']) if 'proxy' in config else None
        self.client = openai.AsyncOpenAI(api_key=config['api_key'], http_client=http_client)
        self.config = config
        self.plugin_manager = plugin_manager
        self.conversations: dict[int: list] = {}
        self.conversations_vision: dict[int: bool] = {}
        self.last_updated: dict[int: datetime] = {}

    def adjust_settings_for_plan(self, plan: str):
        """Ajuste les paramètres OpenAI selon le plan d'abonnement."""
        if plan == "premium":
            self.config['model'] = "gpt-4o"
            self.config['max_tokens'] = 2400
            self.config['image_model'] = "dall-e-3"
        else:
            self.config['model'] = "gpt-3.5-turbo"
            self.config['max_tokens'] = 1200
            self.config['image_model'] = "dall-e-2"

    def get_conversation_stats(self, chat_id: int) -> tuple[int, int]:
        if chat_id not in self.conversations:
            self.reset_chat_history(chat_id)
        return len(self.conversations[chat_id]), self.__count_tokens(self.conversations[chat_id])

    async def get_chat_response(self, chat_id: int, query: str, plan: str) -> tuple[str, str]:
        self.adjust_settings_for_plan(plan)
        plugins_used = ()
        response = await self.__common_get_chat_response(chat_id, query)
        if self.config['enable_functions'] and not self.conversations_vision[chat_id]:
            response, plugins_used = await self.__handle_function_call(chat_id, response)
            if is_direct_result(response):
                return response, '0'

        answer = ''
        if len(response.choices) > 1 and self.config['n_choices'] > 1:
            for index, choice in enumerate(response.choices):
                content = choice.message.content.strip()
                if index == 0:
                    self.__add_to_history(chat_id, role="assistant", content=content)
                answer += f'{index + 1}\u20e3\n{content}\n\n'
        else:
            answer = response.choices[0].message.content.strip()
            self.__add_to_history(chat_id, role="assistant", content=answer)

        bot_language = self.config['bot_language']
        show_plugins_used = len(plugins_used) > 0 and self.config['show_plugins_used']
        plugin_names = tuple(self.plugin_manager.get_plugin_source_name(plugin) for plugin in plugins_used)
        if self.config['show_usage']:
            answer += f"\n\n---\n💰 {response.usage.total_tokens} {localized_text('stats_tokens', bot_language)} " \
                      f"({response.usage.prompt_tokens} {localized_text('prompt', bot_language)}, " \
                      f"{response.usage.completion_tokens} {localized_text('completion', bot_language)})"
            if show_plugins_used:
                answer += f"\n🔌 {', '.join(plugin_names)}"
        elif show_plugins_used:
            answer += f"\n\n---\n🔌 {', '.join(plugin_names)}"
        return answer, response.usage.total_tokens

    async def get_chat_response_stream(self, chat_id: int, query: str, plan: str):
        self.adjust_settings_for_plan(plan)
        plugins_used = ()
        response = await self.__common_get_chat_response(chat_id, query, stream=True)
        if self.config['enable_functions'] and not self.conversations_vision[chat_id]:
            response, plugins_used = await self.__handle_function_call(chat_id, response, stream=True)
            if is_direct_result(response):
                yield response, '0'
                return

        answer = ''
        async for chunk in response:
            if len(chunk.choices) == 0:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                answer += delta.content
                yield answer, 'not_finished'
        answer = answer.strip()
        self.__add_to_history(chat_id, role="assistant", content=answer)
        tokens_used = str(self.__count_tokens(self.conversations[chat_id]))

        show_plugins_used = len(plugins_used) > 0 and self.config['show_plugins_used']
        plugin_names = tuple(self.plugin_manager.get_plugin_source_name(plugin) for plugin in plugins_used)
        if self.config['show_usage']:
            answer += f"\n\n---\n💰 {tokens_used} {localized_text('stats_tokens', self.config['bot_language'])}"
            if show_plugins_used:
                answer += f"\n🔌 {', '.join(plugin_names)}"
        elif show_plugins_used:
            answer += f"\n\n---\n🔌 {', '.join(plugin_names)}"
        yield answer, tokens_used

    @retry(reraise=True, retry=retry_if_exception_type(openai.RateLimitError), wait=wait_fixed(20), stop=stop_after_attempt(3))
    async def __common_get_chat_response(self, chat_id: int, query: str, stream=False):
        bot_language = self.config['bot_language']
        try:
            if chat_id not in self.conversations or self.__max_age_reached(chat_id):
                self.reset_chat_history(chat_id)
            self.last_updated[chat_id] = datetime.datetime.now()
            self.__add_to_history(chat_id, role="user", content=query)

            token_count = self.__count_tokens(self.conversations[chat_id])
            exceeded_max_tokens = token_count + self.config['max_tokens'] > self.__max_model_tokens()
            exceeded_max_history_size = len(self.conversations[chat_id]) > self.config['max_history_size']

            if exceeded_max_tokens or exceeded_max_history_size:
                logging.info(f'Chat history for chat ID {chat_id} is too long. Summarising...')
                try:
                    summary = await self.__summarise(self.conversations[chat_id][:-1])
                    self.reset_chat_history(chat_id, self.conversations[chat_id][0]['content'])
                    self.__add_to_history(chat_id, role="assistant", content=summary)
                    self.__add_to_history(chat_id, role="user", content=query)
                except Exception as e:
                    logging.warning(f'Error while summarising chat history: {str(e)}. Popping elements instead...')
                    self.conversations[chat_id] = self.conversations[chat_id][-self.config['max_history_size']:]

            max_tokens_str = 'max_completion_tokens' if self.config['model'] in O_MODELS else 'max_tokens'
            common_args = {
                'model': self.config['model'] if not self.conversations_vision[chat_id] else self.config['vision_model'],
                'messages': self.conversations[chat_id],
                'temperature': self.config['temperature'],
                'n': self.config['n_choices'],
                max_tokens_str: self.config['max_tokens'],
                'presence_penalty': self.config['presence_penalty'],
                'frequency_penalty': self.config['frequency_penalty'],
                'stream': stream
            }

            if self.config['enable_functions'] and not self.conversations_vision[chat_id]:
                functions = self.plugin_manager.get_functions_specs()
                if len(functions) > 0:
                    common_args['functions'] = functions
                    common_args['function_call'] = 'auto'
            return await self.client.chat.completions.create(**common_args)
        except openai.RateLimitError as e:
            raise e
        except openai.BadRequestError as e:
            raise Exception(f"⚠️ _{localized_text('openai_invalid', bot_language)}._ ⚠️\n{str(e)}") from e
        except Exception as e:
            raise Exception(f"⚠️ _{localized_text('error', bot_language)}._ ⚠️\n{str(e)}") from e

    async def __handle_function_call(self, chat_id, response, stream=False, times=0, plugins_used=()):
        function_name = ''
        arguments = ''
        if stream:
            async for item in response:
                if len(item.choices) > 0:
                    first_choice = item.choices[0]
                    if first_choice.delta and first_choice.delta.function_call:
                        if first_choice.delta.function_call.name:
                            function_name += first_choice.delta.function_call.name
                        if first_choice.delta.function_call.arguments:
                            arguments += first_choice.delta.function_call.arguments
                    elif first_choice.finish_reason and first_choice.finish_reason == 'function_call':
                        break
                    else:
                        return response, plugins_used
                else:
                    return response, plugins_used
        else:
            if len(response.choices) > 0:
                first_choice = response.choices[0]
                if first_choice.message.function_call:
                    function_name = first_choice.message.function_call.name
                    arguments = first_choice.message.function_call.arguments
                else:
                    return response, plugins_used
            else:
                return response, plugins_used

        logging.info(f'Calling function {function_name} with arguments {arguments}')
        function_response = await self.plugin_manager.call_function(function_name, self, arguments)
        if function_name not in plugins_used:
            plugins_used += (function_name,)

        if is_direct_result(function_response):
            self.__add_function_call_to_history(chat_id=chat_id, function_name=function_name,
                                                content=json.dumps({'result': 'Done, the content has been sent to the user.'}))
            return function_response, plugins_used

        self.__add_function_call_to_history(chat_id=chat_id, function_name=function_name, content=function_response)
        response = await self.client.chat.completions.create(
            model=self.config['model'],
            messages=self.conversations[chat_id],
            functions=self.plugin_manager.get_functions_specs(),
            function_call='auto' if times < self.config['functions_max_consecutive_calls'] else 'none',
            stream=stream
        )
        return await self.__handle_function_call(chat_id, response, stream, times + 1, plugins_used)

    async def generate_image(self, prompt: str, plan: str) -> tuple[str, str]:
        self.adjust_settings_for_plan(plan)
        if plan == "free" and self.config['image_model'] == "dall-e-3":
            raise Exception("Image generation with DALL·E 3 is only available for Premium users.")
        bot_language = self.config['bot_language']
        try:
            response = await self.client.images.generate(
                prompt=prompt,
                n=1,
                model=self.config['image_model'],
                quality=self.config['image_quality'],
                style=self.config['image_style'],
                size=self.config['image_size']
            )
            if not response.data:
                raise Exception(f"⚠️ _{localized_text('error', bot_language)}._ ⚠️\nNo response from GPT.")
            return response.data[0].url, self.config['image_size']
        except Exception as e:
            raise Exception(f"⚠️ _{localized_text('error', bot_language)}._ ⚠️\n{str(e)}") from e

    async def generate_speech(self, text: str) -> tuple[any, int]:
        bot_language = self.config['bot_language']
        try:
            response = await self.client.audio.speech.create(
                model=self.config['tts_model'],
                voice=self.config['tts_voice'],
                input=text,
                response_format='opus'
            )
            temp_file = io.BytesIO()
            temp_file.write(response.read())
            temp_file.seek(0)
            return temp_file, len(text)
        except Exception as e:
            raise Exception(f"⚠️ _{localized_text('error', bot_language)}._ ⚠️\n{str(e)}") from e

    async def transcribe(self, filename):
        try:
            with open(filename, "rb") as audio:
                prompt_text = self.config['whisper_prompt']
                result = await self.client.audio.transcriptions.create(model="whisper-1", file=audio, prompt=prompt_text)
                return result.text
        except Exception as e:
            raise Exception(f"⚠️ _{localized_text('error', self.config['bot_language'])}._ ⚠️\n{str(e)}") from e

    @retry(reraise=True, retry=retry_if_exception_type(openai.RateLimitError), wait=wait_fixed(20), stop=stop_after_attempt(3))
    async def __common_get_chat_response_vision(self, chat_id: int, content: list, stream=False):
        bot_language = self.config['bot_language']
        try:
            if chat_id not in self.conversations or self.__max_age_reached(chat_id):
                self.reset_chat_history(chat_id)
            self.last_updated[chat_id] = datetime.datetime.now()

            if self.config['enable_vision_follow_up_questions']:
                self.conversations_vision[chat_id] = True
                self.__add_to_history(chat_id, role="user", content=content)
            else:
                for message in content:
                    if message['type'] == 'text':
                        query = message['text']
                        break
                self.__add_to_history(chat_id, role="user", content=query)

            token_count = self.__count_tokens(self.conversations[chat_id])
            exceeded_max_tokens = token_count + self.config['max_tokens'] > self.__max_model_tokens()
            exceeded_max_history_size = len(self.conversations[chat_id]) > self.config['max_history_size']

            if exceeded_max_tokens or exceeded_max_history_size:
                logging.info(f'Chat history for chat ID {chat_id} is too long. Summarising...')
                try:
                    last = self.conversations[chat_id][-1]
                    summary = await self.__summarise(self.conversations[chat_id][:-1])
                    self.reset_chat_history(chat_id, self.conversations[chat_id][0]['content'])
                    self.__add_to_history(chat_id, role="assistant", content=summary)
                    self.conversations[chat_id] += [last]
                except Exception as e:
                    logging.warning(f'Error while summarising chat history: {str(e)}. Popping elements instead...')
                    self.conversations[chat_id] = self.conversations[chat_id][-self.config['max_history_size']:]

            message = {'role': 'user', 'content': content}
            common_args = {
                'model': self.config['vision_model'],
                'messages': self.conversations[chat_id][:-1] + [message],
                'temperature': self.config['temperature'],
                'n': 1,
                'max_tokens': self.config['vision_max_tokens'],
                'presence_penalty': self.config['presence_penalty'],
                'frequency_penalty': self.config['frequency_penalty'],
                'stream': stream
            }
            return await self.client.chat.completions.create(**common_args)
        except openai.RateLimitError as e:
            raise e
        except openai.BadRequestError as e:
            raise Exception(f"⚠️ _{localized_text('openai_invalid', bot_language)}._ ⚠️\n{str(e)}") from e
        except Exception as e:
            raise Exception(f"⚠️ _{localized_text('error', bot_language)}._ ⚠️\n{str(e)}") from e

    async def interpret_image(self, chat_id, fileobj, prompt=None, plan: str = "free"):
        self.adjust_settings_for_plan(plan)
        if plan == "free" and not self.config.get('enable_vision', True):
            raise Exception("Vision features are only available for Premium users.")
        prompt = self.config['vision_prompt'] if prompt is None else prompt
        content = [{'type': 'text', 'text': prompt}, {'type': 'image_url', 'image_url': {'url': encode_image(fileobj), 'detail': 'low' if plan == "free" else self.config['vision_detail']}}]  # Détail 'low' pour gratuits
        response = await self.__common_get_chat_response_vision(chat_id, content)

        answer = response.choices[0].message.content.strip()
        self.__add_to_history(chat_id, role="assistant", content=answer)
        bot_language = self.config['bot_language']
        if self.config['show_usage']:
            answer += f"\n\n---\n💰 {response.usage.total_tokens} {localized_text('stats_tokens', bot_language)} " \
                      f"({response.usage.prompt_tokens} {localized_text('prompt', bot_language)}, " \
                      f"{response.usage.completion_tokens} {localized_text('completion', bot_language)})"
        return answer, response.usage.total_tokens

    async def interpret_image_stream(self, chat_id, fileobj, prompt=None, plan: str = "free"):
        self.adjust_settings_for_plan(plan)
        if plan == "free" and not self.config.get('enable_vision', True):
            raise Exception("Vision streaming is only available for Premium users.")
        prompt = self.config['vision_prompt'] if prompt is None else prompt
        content = [{'type': 'text', 'text': prompt}, {'type': 'image_url', 'image_url': {'url': encode_image(fileobj), 'detail': 'low' if plan == "free" else self.config['vision_detail']}}]
        response = await self.__common_get_chat_response_vision(chat_id, content, stream=True)

        answer = ''
        async for chunk in response:
            if len(chunk.choices) == 0:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                answer += delta.content
                yield answer, 'not_finished'
        answer = answer.strip()
        self.__add_to_history(chat_id, role="assistant", content=answer)
        tokens_used = str(self.__count_tokens(self.conversations[chat_id]))
        if self.config['show_usage']:
            answer += f"\n\n---\n💰 {tokens_used} {localized_text('stats_tokens', self.config['bot_language'])}"
        yield answer, tokens_used

    def reset_chat_history(self, chat_id, content=''):
        if not content:
            content = self.config['assistant_prompt']
        self.conversations[chat_id] = [{"role": "assistant" if self.config['model'] in O_MODELS else "system", "content": content}]
        self.conversations_vision[chat_id] = False

    def __max_age_reached(self, chat_id) -> bool:
        if chat_id not in self.last_updated:
            return False
        last_updated = self.last_updated[chat_id]
        now = datetime.datetime.now()
        return last_updated < now - datetime.timedelta(minutes=self.config['max_conversation_age_minutes'])

    def __add_function_call_to_history(self, chat_id, function_name, content):
        self.conversations[chat_id].append({"role": "function", "name": function_name, "content": content})

    def __add_to_history(self, chat_id, role, content):
        self.conversations[chat_id].append({"role": role, "content": content})

    async def __summarise(self, conversation) -> str:
        # Optimisation : éviter la summarisation pour les utilisateurs gratuits
        if self.config.get('plan', 'free') == 'free':
            return "Previous conversation truncated to save tokens."
        messages = [
            {"role": "assistant", "content": "Summarize this conversation in 700 characters or less"},
            {"role": "user", "content": str(conversation)}
        ]
        response = await self.client.chat.completions.create(
            model=self.config['model'],
            messages=messages,
            temperature=1 if self.config['model'] in O_MODELS else 0.4
        )
        return response.choices[0].message.content

    def __max_model_tokens(self):
        base = 4096
        if self.config['model'] in GPT_3_MODELS:
            return base
        elif self.config['model'] in GPT_3_16K_MODELS:
            return base * 4
        elif self.config['model'] in GPT_4_MODELS:
            return base * 2
        elif self.config['model'] in GPT_4_32K_MODELS:
            return base * 8
        elif self.config['model'] in GPT_4_VISION_MODELS or self.config['model'] in GPT_4_128K_MODELS or self.config['model'] in GPT_4O_MODELS:
            return base * 31
        elif self.config['model'] in O_MODELS:
            return 100_000 if self.config['model'] == "o1" else 32_768 if self.config['model'] == "o1-preview" else 65_536
        raise NotImplementedError(f"Max tokens for model {self.config['model']} is not implemented yet.")



    def __count_tokens(self, messages) -> int:
        model = self.config['model']
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("o200k_base")
        if model in GPT_ALL_MODELS:
            tokens_per_message = 3
            tokens_per_name = 1
        else:
            raise NotImplementedError(f"num_tokens_from_messages() is not implemented for model {model}.")
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                if key == 'content':
                    if isinstance(value, str):
                        num_tokens += len(encoding.encode(value))
                    else:
                        for message1 in value:
                            if message1['type'] == 'image_url':
                                image = decode_image(message1['image_url']['url'])
                                num_tokens += self.__count_tokens_vision(image)
                            else:
                                num_tokens += len(encoding.encode(message1['text']))
                else:
                    num_tokens += len(encoding.encode(value))
                    if key == "name":
                        num_tokens += tokens_per_name
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
        return num_tokens

    def __count_tokens_vision(self, image_bytes: bytes) -> int:
        image_file = io.BytesIO(image_bytes)
        image = Image.open(image_file)
        model = self.config['vision_model']
        if model not in GPT_4_VISION_MODELS:
            raise NotImplementedError(f"count_tokens_vision() is not implemented for model {model}.")
        w, h = image.size
        if w > h:
            w, h = h, w
        base_tokens = 85
        detail = self.config['vision_detail']
        if detail == 'low':
            return base_tokens
        elif detail in ('high', 'auto'):  # assuming worst cost for auto
            f = max(w / 768, h / 2048)
            if f > 1:
                w, h = int(w / f), int(h / f)
            tw, th = (w + 511) // 512, (h + 511) // 512
            tiles = tw * th
            num_tokens = base_tokens + tiles * 170
            return num_tokens
        else:
            raise NotImplementedError(f"unknown parameter detail={detail} for model {model}.")



