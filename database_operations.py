from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
import sqlite3
import streamlit as st
from utils import load_config
import threading
import os # Added this import for directory operations

# Constants
DEFAULT_CHAT_MEMORY_LENGTH = 2
DEFAULT_RETRIEVED_DOCUMENTS = 3
DEFAULT_CHUNK_SIZE = 1024
DEFAULT_CHUNK_OVERLAP = 50

class DatabaseConnection:
    """Handles database connection management with thread safety."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._connection = None
        self._lock = threading.Lock()

    @property
    def connection(self) -> sqlite3.Connection:
        with self._lock:
            if not self._connection:
                # Ensure the directory for the database file exists before connecting
                db_dir = os.path.dirname(self.db_path)
                if db_dir: # Only create directory if a path is specified (not just a filename)
                    os.makedirs(db_dir, exist_ok=True)
                
                self._connection = sqlite3.connect(self.db_path, check_same_thread=False)
                # It's good practice to set a row_factory for easier data access
                self._connection.row_factory = sqlite3.Row 
            return self._connection

    def close(self) -> None:
        with self._lock:
            if self._connection:
                self._connection.close()
                self._connection = None

class BaseRepository(ABC):
    """Abstract base class for all repositories."""
    
    def __init__(self, db_connection: DatabaseConnection):
        self.db = db_connection

    @abstractmethod
    def create_table(self) -> None:
        pass

class MessageRepository(BaseRepository):
    """Handles all message-related database operations."""
    
    def create_table(self) -> None:
        with self.db.connection as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    message_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_history_id TEXT NOT NULL,
                    sender_type TEXT NOT NULL,
                    message_type TEXT NOT NULL,
                    text_content TEXT,
                    blob_content BLOB
                );
            """)
            conn.commit()

    def save_message(self, chat_history_id: str, sender_type: str, 
                     message_type: str, content: Union[str, bytes]) -> None:
        with self.db.connection as conn:
            cursor = conn.cursor()
            if message_type == 'text':
                cursor.execute(
                    'INSERT INTO messages (chat_history_id, sender_type, message_type, text_content) '
                    'VALUES (?, ?, ?, ?)',
                    (chat_history_id, sender_type, message_type, content)
                )
            else:
                cursor.execute(
                    'INSERT INTO messages (chat_history_id, sender_type, message_type, blob_content) '
                    'VALUES (?, ?, ?, ?)',
                    (chat_history_id, sender_type, message_type, sqlite3.Binary(content))
                )
            conn.commit()

    def load_messages(self, chat_history_id: str) -> List[Dict[str, Any]]:
        with self.db.connection as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT message_id, sender_type, message_type, text_content, blob_content "
                "FROM messages WHERE chat_history_id = ?",
                (chat_history_id,)
            )
            return [
                {
                    'message_id': row['message_id'], # Use row as dict due to row_factory
                    'sender_type': row['sender_type'],
                    'message_type': row['message_type'],
                    'content': row['text_content'] if row['message_type'] == 'text' else row['blob_content']
                }
                for row in cursor.fetchall()
            ]

    def load_last_k_text_messages(self, chat_history_id: str, k: int) -> List[Dict[str, Any]]:
        with self.db.connection as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT message_id, sender_type, message_type, text_content
                FROM messages
                WHERE chat_history_id = ? AND message_type = 'text'
                ORDER BY message_id DESC
                LIMIT ?
            """, (chat_history_id, k))
            
            # Fetch all and reverse to maintain chronological order
            return [
                {
                    'message_id': row['message_id'],
                    'sender_type': row['sender_type'],
                    'message_type': row['message_type'],
                    'content': row['text_content']
                }
                for row in reversed(cursor.fetchall())
            ]

    def delete_chat_history(self, chat_history_id: str) -> None:
        with self.db.connection as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM messages WHERE chat_history_id = ?", (chat_history_id,))
            conn.commit()

    def get_all_chat_history_ids(self) -> List[str]:
        """Get all unique chat history IDs from the messages table."""
        with self.db.connection as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT chat_history_id FROM messages ORDER BY chat_history_id ASC")
            return [row['chat_history_id'] for row in cursor.fetchall()]

class SettingsRepository(BaseRepository):
    """Handles all settings-related database operations."""
    
    def create_table(self) -> None:
        with self.db.connection as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS settings (
                    setting_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    setting_name TEXT NOT NULL UNIQUE,
                    setting_value TEXT NOT NULL
                );
            """)
            conn.commit()

    def get_setting(self, setting_name: str, default_value: Any) -> Any:
        with self.db.connection as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT setting_value FROM settings WHERE setting_name = ?",
                (setting_name,)
            )
            result = cursor.fetchone()
            
            if result:
                return result['setting_value'] # Use dict-like access
            
            self.update_setting(setting_name, default_value)
            return default_value

    def update_setting(self, setting_name: str, setting_value: Any) -> None:
        with self.db.connection as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO settings (setting_name, setting_value) VALUES (?, ?)",
                (setting_name, str(setting_value))
            )
            conn.commit()

class DatabaseManager:
    """Main database manager that coordinates all database operations."""
    
    def __init__(self, db_path: str):
        self.db_connection = DatabaseConnection(db_path)
        self.message_repo = MessageRepository(self.db_connection)
        self.settings_repo = SettingsRepository(self.db_connection)
        self._initialize_database()

    def _initialize_database(self) -> None:
        self.message_repo.create_table()
        self.settings_repo.create_table()

    def close(self) -> None:
        self.db_connection.close()

# Initialize the database manager with configuration
config = load_config()
db_manager = DatabaseManager(config["chat_sessions_database_path"])

# Streamlit session state management
def get_db_manager():
    if 'db_manager' not in st.session_state:
        st.session_state.db_manager = db_manager
    return st.session_state.db_manager

def close_db_manager():
    if 'db_manager' in st.session_state:
        st.session_state.db_manager.close()
        st.session_state.db_manager = None

if __name__ == "__main__":
    # This block is for direct execution of this script, not when imported by app.py
    # It ensures the database is initialized and then closed properly if run standalone.
    temp_db_manager = DatabaseManager(config["chat_sessions_database_path"])
    print(f"Database initialized at: {config['chat_sessions_database_path']}")
    # You can add some test operations here if needed
    temp_db_manager.close()
    print("Database manager closed.")