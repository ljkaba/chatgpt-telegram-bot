import os
import sqlite3
import psycopg2
from datetime import datetime, timedelta
from urllib.parse import urlparse

# Fonction pour obtenir une connexion à la base de données
def get_db_connection():
    """Retourne une connexion à la base de données (SQLite ou PostgreSQL)"""
    if os.getenv("DATABASE_URL"):  # Production (Heroku)
        url = urlparse(os.getenv("DATABASE_URL"))
        return psycopg2.connect(
            dbname=url.path[1:],
            user=url.username,
            password=url.password,
            host=url.hostname,
            port=url.port,
            sslmode="require"
        )
    else:  # Développement local (SQLite)
        return sqlite3.connect("subscribers.db")

# Fonction pour initialiser la base de données
def init_db():
    """Initialise la base de données (création de la table si elle n'existe pas)"""
    conn = get_db_connection()
    c = conn.cursor()
    if os.getenv("DATABASE_URL"):  # PostgreSQL
        c.execute(
            """CREATE TABLE IF NOT EXISTS subscribers (
                user_id TEXT PRIMARY KEY,
                plan TEXT NOT NULL,
                start_date TEXT,
                end_date TEXT,
                message_count INTEGER DEFAULT 0,
                last_reset_date TEXT,
                full_name TEXT
            )"""
        )
    else:  # SQLite
        c.execute(
            """CREATE TABLE IF NOT EXISTS subscribers (
                user_id TEXT PRIMARY KEY,
                plan TEXT NOT NULL,
                start_date TEXT,
                end_date TEXT,
                message_count INTEGER DEFAULT 0,
                last_reset_date TEXT,
                full_name TEXT
            )"""
        )
    conn.commit()
    conn.close()

# Fonction pour ajouter ou mettre à jour un abonné
def add_or_update_subscriber(user_id, plan, duration_days=None, full_name=None):
    """Ajoute ou met à jour un abonné dans la base de données"""
    conn = get_db_connection()
    c = conn.cursor()
    start_date = datetime.now().isoformat()
    end_date = (
        (datetime.now() + timedelta(days=duration_days)).isoformat()
        if duration_days
        else None
    )
    if os.getenv("DATABASE_URL"):  # PostgreSQL
        c.execute(
            """INSERT INTO subscribers 
               (user_id, plan, start_date, end_date, message_count, last_reset_date, full_name)
               VALUES (%s, %s, %s, %s, %s, %s, %s)
               ON CONFLICT (user_id) DO UPDATE SET
               plan = EXCLUDED.plan,
               start_date = EXCLUDED.start_date,
               end_date = EXCLUDED.end_date,
               message_count = EXCLUDED.message_count,
               last_reset_date = EXCLUDED.last_reset_date,
               full_name = EXCLUDED.full_name""",
            (str(user_id), plan, start_date, end_date, 0, datetime.now().date().isoformat(), full_name),
        )
    else:  # SQLite
        c.execute(
            """INSERT OR REPLACE INTO subscribers 
               (user_id, plan, start_date, end_date, message_count, last_reset_date, full_name)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (str(user_id), plan, start_date, end_date, 0, datetime.now().date().isoformat(), full_name),
        )
    conn.commit()
    conn.close()

# Fonction pour récupérer le statut d'un abonné
def get_subscriber_status(user_id):
    """Récupère le statut d'un abonné"""
    conn = get_db_connection()
    c = conn.cursor()
    if os.getenv("DATABASE_URL"):  # PostgreSQL
        c.execute(
            "SELECT plan, end_date, message_count, last_reset_date, full_name FROM subscribers WHERE user_id = %s",
            (str(user_id),),
        )
    else:  # SQLite
        c.execute(
            "SELECT plan, end_date, message_count, last_reset_date, full_name FROM subscribers WHERE user_id = ?",
            (str(user_id),),
        )
    result = c.fetchone()
    conn.close()
    return result

# Fonction pour incrémenter le compteur de messages
def increment_message_count(user_id, full_name=None):
    """Incrémente le compteur de messages d'un utilisateur"""
    conn = get_db_connection()
    c = conn.cursor()
    today = datetime.now().date().isoformat()
    if os.getenv("DATABASE_URL"):  # PostgreSQL
        c.execute(
            "SELECT message_count, last_reset_date FROM subscribers WHERE user_id = %s",
            (str(user_id),),
        )
    else:  # SQLite
        c.execute(
            "SELECT message_count, last_reset_date FROM subscribers WHERE user_id = ?",
            (str(user_id),),
        )
    result = c.fetchone()
    if result:
        message_count, last_reset_date = result
        if last_reset_date != today:
            print(f"Reset du compteur de messages pour {user_id} ({full_name}) : {message_count} -> 0")
            message_count = 0
            if os.getenv("DATABASE_URL"):  # PostgreSQL
                c.execute(
                    "UPDATE subscribers SET message_count = %s, last_reset_date = %s WHERE user_id = %s",
                    (0, today, str(user_id)),
                )
            else:  # SQLite
                c.execute(
                    "UPDATE subscribers SET message_count = ?, last_reset_date = ? WHERE user_id = ?",
                    (0, today, str(user_id)),
                )
        message_count += 1
        if os.getenv("DATABASE_URL"):  # PostgreSQL
            c.execute(
                "UPDATE subscribers SET message_count = %s WHERE user_id = %s",
                (message_count, str(user_id)),
            )
        else:  # SQLite
            c.execute(
                "UPDATE subscribers SET message_count = ? WHERE user_id = ?",
                (message_count, str(user_id)),
            )
    else:
        print(f"Création d'un nouvel utilisateur {user_id} ({full_name}) avec count 1")
        if os.getenv("DATABASE_URL"):  # PostgreSQL
            c.execute(
                "INSERT INTO subscribers (user_id, plan, start_date, end_date, message_count, last_reset_date, full_name) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                (str(user_id), "free", today, None, 1, today, full_name),
            )
        else:  # SQLite
            c.execute(
                "INSERT INTO subscribers (user_id, plan, start_date, end_date, message_count, last_reset_date, full_name) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (str(user_id), "free", today, None, 1, today, full_name),
            )
    conn.commit()
    conn.close()

# Fonction pour lister tous les abonnés
def list_subscribers():
    """Retourne la liste de tous les abonnés"""
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT user_id, plan, end_date, full_name FROM subscribers")
    result = c.fetchall()
    conn.close()
    return result

def get_expiring_subscriptions(days_before: int = 3):
    """Récupère les abonnements expirant dans X jours"""
    conn = get_db_connection()
    c = conn.cursor()
    target_date = (datetime.now() + timedelta(days=days_before)).date().isoformat()
    
    if os.getenv("DATABASE_URL"):  # PostgreSQL
        c.execute(
            "SELECT user_id, end_date, full_name FROM subscribers WHERE end_date <= %s",
            (target_date,)
        )
    else:  # SQLite
        c.execute(
            "SELECT user_id, end_date, full_name FROM subscribers WHERE end_date <= ?",
            (target_date,)
        )
    
    result = c.fetchall()
    conn.close()
    return result

# Fonction pour supprimer un abonné
def remove_subscriber(user_id):
    """Supprime un abonné de la base de données"""
    conn = get_db_connection()
    c = conn.cursor()
    if os.getenv("DATABASE_URL"):  # PostgreSQL
        c.execute("DELETE FROM subscribers WHERE user_id = %s", (str(user_id),))
    else:  # SQLite
        c.execute("DELETE FROM subscribers WHERE user_id = ?", (str(user_id),))
    conn.commit()
    conn.close()