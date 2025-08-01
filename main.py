import asyncio
import logging
import os
from aiohttp import web

from aiogram import Bot, Dispatcher, types
from aiogram.enums import ParseMode
from aiogram.filters import Command, StateFilter
from aiogram.webhook.aiohttp_server import SimpleRequestHandler, setup_application
from aiogram.client.default import DefaultBotProperties

# Import modules
from config import API_TOKEN
from states import BotStates
from commands import (
    set_commands, 
    handle_start, 
    handle_help, 
    handle_new, 
    handle_rules_command, 
    handle_clear, 
    handle_rules_input,
    handle_changelog
)
from message_handlers import (
    handle_message,
    handle_analyze_callback
)

# Configure logging (in addition to config.py setup)
logger = logging.getLogger(__name__)

# Import Claude API utilities
try:
    import claude_utils
    USING_CLAUDE_API = True
    logger.info("Claude API utilities loaded successfully")
except Exception as e:
    logger.error(f"Failed to load Claude API utilities: {e}")
    USING_CLAUDE_API = False

async def on_startup(bot: Bot):
    """Execute when bot starts"""
    # Set webhook
    webhook_url = os.getenv('WEBHOOK_HOST', 'https://feedmeoil-telegram-bot.fly.dev')
    webhook_path = f'/webhook/{API_TOKEN}'
    webhook_url = f"{webhook_url}{webhook_path}"
    
    # Remove any existing webhook
    await bot.delete_webhook(drop_pending_updates=True)
    
    # Set new webhook
    await bot.set_webhook(webhook_url)
    logger.info(f"Webhook set to: {webhook_url}")
    
    # Set bot commands
    await set_commands(bot)
    logger.info("Bot commands set")
    
    # Start Claude API cleanup task if available
    if USING_CLAUDE_API:
        try:
            # Create a task that will run in the background
            asyncio.create_task(claude_utils.start_cleanup_task())
            logger.info("Started Claude API cleanup task")
        except Exception as e:
            logger.error(f"Failed to start Claude API cleanup task: {e}")

async def on_shutdown(bot: Bot):
    """Execute when bot shuts down"""
    logger.warning("Shutting down...")
    await bot.delete_webhook()

async def health_check(request):
    """Health check endpoint"""
    return web.Response(text="OK")

async def main():
    """Main function to start the bot"""
    # Enhanced error handling and diagnostics
    try:
        print("===== STARTING BOT INITIALIZATION =====")
        logger.info("Starting bot initialization...")
        
        # Debug environment variables
        print(f"Environment variables:")
        print(f"BOT_TOKEN: {'Set' if os.getenv('BOT_TOKEN') else 'NOT SET'}")
        print(f"WEBHOOK_HOST: {os.getenv('WEBHOOK_HOST')}")
        print(f"REDIS_URL: {'Set' if os.getenv('REDIS_URL') else 'NOT SET'}")
        print(f"ANTHROPIC_API_KEY: {'Set' if os.getenv('ANTHROPIC_API_KEY') else 'NOT SET'}")
        logger.info("Environment variables checked")
        
        # Import configuration first to ensure all variables are initialized properly
        from config import API_TOKEN, collected_content, message_timers
        logger.info("Config loaded successfully")
        
        # Initialize bot and dispatcher with memory storage for FSM
        from aiogram.fsm.storage.memory import MemoryStorage
        
        bot = Bot(token=API_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
        storage = MemoryStorage()
        dp = Dispatcher(storage=storage)
        logger.info("Bot and dispatcher initialized successfully")
    
        # Register startup and shutdown handlers
        @dp.startup()
        async def startup_handler():
            await on_startup(bot)
            
        @dp.shutdown()
        async def shutdown_handler():
            await on_shutdown(bot)

        # Register command handlers
        dp.message.register(handle_start, Command("start"))  # Initialize the bot
        dp.message.register(handle_help, Command("help"))    # Get help information
        dp.message.register(handle_clear, Command("clean"))  # Clear user rules
        dp.message.register(handle_rules_command, Command("rules"))  # Set context for responses
        dp.message.register(handle_changelog, Command("changelog"))  # Show recent updates
        dp.message.register(handle_new, Command("new"))    # Start a new conversation
    
        # Register state-specific handlers
        dp.message.register(handle_rules_input, StateFilter(BotStates.waiting_for_rules))
        
        # Register callback query handlers
        dp.callback_query.register(handle_analyze_callback, lambda c: c.data == "analyze_forwards")
        # Button handler has been removed as button functionality is not required
        
        # Register message handler for all other messages
        dp.message.register(handle_message)
        
        logger.info("All message handlers registered successfully")
        
        # Setup web application for webhook
        app = web.Application()
        SimpleRequestHandler(dispatcher=dp, bot=bot).register(app, path=f'/webhook/{API_TOKEN}')
        setup_application(app, dp, bot=bot)
        app.router.add_get('/health', health_check)
        logger.info("Web application setup complete")

        # Create downloads directory if it doesn't exist
        os.makedirs("downloads", exist_ok=True)
        
        # Start the web server
        runner = web.AppRunner(app)
        await runner.setup()
        
        port = int(os.getenv('PORT', 8080))
        site = web.TCPSite(runner, '0.0.0.0', port)
        await site.start()

        logger.info(f"Bot started on port {port}, listening on 0.0.0.0:{port}")

        # Keep the server running
        try:
            while True:
                await asyncio.sleep(3600)
        except (KeyboardInterrupt, SystemExit):
            logger.info("Shutting down bot")
        finally:
            await runner.cleanup()
            
    except Exception as e:
        import traceback
        logger.error(f"Critical error during bot initialization: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        import traceback
        logging.error(f"Failed to run bot: {e}")
        logging.error(f"Traceback: {traceback.format_exc()}")
        import sys
        sys.exit(1)
