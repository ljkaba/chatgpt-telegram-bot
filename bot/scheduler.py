# FICHIER NOUVEAU :

import schedule
import time
from datetime import datetime, timedelta
from database import get_expiring_subscriptions
from telegram_bot import ChatGPTTelegramBot

def start_scheduler(bot: ChatGPTTelegramBot):
    """Planifie les tâches automatiques"""
    # Notification 3 jours avant expiration
    schedule.every().day.at("09:00").do(
        lambda: check_expiring_subscriptions(bot, days_before=3)
    )

    # Boucle d'exécution
    while True:
        schedule.run_pending()
        time.sleep(1)

def check_expiring_subscriptions(bot: ChatGPTTelegramBot, days_before: int):
    """Vérifie les abonnements expirants"""
    expiring_users = get_expiring_subscriptions(days_before)
    for user_id, end_date in expiring_users:
        bot.send_notification(
            user_id=user_id,
            message=f"⚠️ Votre abonnement expire le {end_date}. Renouvelez avec /upgrade."
        )