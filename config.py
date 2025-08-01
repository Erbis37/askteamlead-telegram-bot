import os
import sys
import logging
from aiogram.fsm.storage.memory import MemoryStorage

# Load .env file if available
try:
    from dotenv import load_dotenv
    load_dotenv()
    logging.info("Loaded environment variables from .env file")
except ImportError:
    logging.warning("python-dotenv not installed, skipping .env loading")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Bot token configuration
API_TOKEN = os.getenv('BOT_TOKEN')

# Validate essential environment variables
if not API_TOKEN:
    logging.error("BOT_TOKEN is not set.")
    sys.exit(1)

# Anthropic API configuration
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
if not ANTHROPIC_API_KEY:
    logging.error("ANTHROPIC_API_KEY is not set.")
    sys.exit(1)

# Trello API configuration
TRELLO_API_KEY = os.getenv('TRELLO_API_KEY')
TRELLO_TOKEN = os.getenv('TRELLO_TOKEN')
TRELLO_BOARD_ID = os.getenv('TRELLO_BOARD_ID') or os.getenv('TRELLO_KANBAN_TEMPLATE_ID')

# Validate Trello configuration
if not all([TRELLO_API_KEY, TRELLO_TOKEN, TRELLO_BOARD_ID]):
    logging.warning("Trello configuration incomplete. Some features may not work.")

# Webhook configuration
WEBHOOK_HOST = os.getenv('WEBHOOK_HOST')
WEBHOOK_PATH = f'/webhook/{API_TOKEN}'
WEBHOOK_URL = f"{WEBHOOK_HOST}{WEBHOOK_PATH}"

# Global dictionaries and sets
message_timers = {}  # Store message processing timers for each user
processing_users = set()  # Track users with ongoing processing

# Unified content collection storage
collected_content = {}  # Store all types of content (documents, URLs, etc.) for each user

# Threshold for when to add summary to response (in characters)
MAX_SUMMARY_THRESHOLD = 300  # If message or sum of messages is longer than this value, summary is added

# Google Speech API configuration (for voice messages)
GOOGLE_SPEECH_ENABLED = True  # Using free Google Speech Recognition
MAX_VOICE_DURATION = 60  # Maximum voice message duration in seconds
