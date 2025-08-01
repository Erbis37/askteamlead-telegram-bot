import os
import logging
import time
import json
from typing import Dict, List, Optional, Any, Tuple
import asyncio
import uuid

# Import the official Anthropic client
try:
    from anthropic import AsyncAnthropic
    IMPORT_ERROR = None
except ImportError as e:
    logging.error(f"Error importing Anthropic client: {e}")
    IMPORT_ERROR = str(e)

# Claude API base settings
CLAUDE_API_KEY = os.getenv("ANTHROPIC_API_KEY")
CLAUDE_MODEL = "claude-3-5-sonnet-20241022"  # Using Claude 3.5 Sonnet - latest stable version
# Note: Claude 4 is not yet available in the API. When it becomes available, update to:
# CLAUDE_MODEL = "claude-4-sonnet" or similar model name

# Check if API key is available or if there was an import error
if not CLAUDE_API_KEY or IMPORT_ERROR:
    if not CLAUDE_API_KEY:
        logging.warning("ANTHROPIC_API_KEY not set. Claude API functionality will be disabled.")
    if IMPORT_ERROR:
        logging.warning(f"Anthropic client could not be imported: {IMPORT_ERROR}. Claude API functionality will be disabled.")
    CLAUDE_AVAILABLE = False
    claude_client = None
else:
    CLAUDE_AVAILABLE = True
    # Initialize the Anthropic client
    claude_client = AsyncAnthropic(api_key=CLAUDE_API_KEY)

# Constants
SESSION_TIMEOUT = 30 * 60  # 30 minutes of inactivity before session expiry

# Conversation history storage
# user_id -> {'messages': List[Dict], 'last_active': timestamp, 'content_analysis_mode': bool, 'system': str}
active_sessions: Dict[int, Dict[str, Any]] = {}


async def create_session(user_id: int, user_rules: Optional[str] = None, content_analysis_mode: bool = False) -> bool:
    """
    Initializes a new conversation history for a user
    
    Args:
        user_id: User's Telegram ID
        user_rules: User's personal rules/context
        content_analysis_mode: Whether to use content analysis mode
        
    Returns:
        True if session was created successfully, False otherwise
    """
    # Check if Claude API is available
    if not CLAUDE_AVAILABLE:
        logging.warning(f"Claude API is not available. Cannot create session for user {user_id}")
        return False
    
    logging.info(f"Creating new conversation for user {user_id} with content_analysis_mode={content_analysis_mode}")
    
    # Choose prompt based on mode
    if content_analysis_mode:
        # System prompt for content analysis mode (documents, URLs)
        system_prompt = (
            "You are an AI Team Lead helping teams work more effectively. You analyze documents, links, and other content "
            "to help with project management, task estimation, and team coordination. "
            "Your task is to provide comprehensive analysis and actionable insights. "
            "Focus on identifying key points, risks, dependencies, and opportunities for optimization. "
            "When users ask about specific information in documents - find and present it clearly. "
            "Be structured and informative. Use bullet points and clear formatting. "
            "IMPORTANT: Never use emojis in your responses. "
            "If the user asks to create files, calendar events, or other documents based on the analysis - "
            "acknowledge the request and generate appropriate content with professional quality. "
            "Always think from a team lead perspective - consider timelines, resources, and team dynamics."
        )
    else:
        # Standard prompt for AI Team Lead conversations
        system_prompt = (
            "You are an AI Team Lead helping teams work more effectively. Your role is to help with: "
            "1. Task decomposition and estimation "
            "2. Creating Gantt charts and project timelines "
            "3. Managing tasks in Trello (create, move, prioritize) "
            "4. Generating CSV reports with work plans and cost estimates "
            "5. Creating presentations for the team "
            "6. Estimating development costs and resources "
            "7. Tracking deadlines and highlighting urgent tasks "
            "8. Processing voice messages (transcribed to text) "
            "\n\n"
            "Be professional but approachable. Provide clear, actionable advice. "
            "When estimating tasks, consider complexity, dependencies, and risks. "
            "Always think about team efficiency and project success. "
            "Structure your responses clearly. IMPORTANT: Never use emojis in your responses. "
            "When working with Trello, mention the specific columns: "
            "Backlog, Design, To Do, Doing, Code Review, Testing, Done"
        )
    
    # If user has personal rules, add them to the prompt
    if user_rules:
        system_prompt += f"""

Важные правила для этого чата:
{user_rules}
"""
    
    # Verify system prompt with a small Claude API call
    try:
        # Just test that the system prompt is valid
        await claude_client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=100,  # Reasonable number of tokens for prompt verification
            system=system_prompt,
            messages=[{"role": "user", "content": "Hello"}]
        )
        
        # Initialize a new conversation history for this user
        active_sessions[user_id] = {
            "system": system_prompt,
            "messages": [],  # Empty message history
            "last_active": time.time(),
            "content_analysis_mode": False  # content analysis mode is off by default
        }
        
        logging.info(f"Created new session entry for user {user_id} with content_analysis_mode=False")
        return True
    except Exception as e:
        logging.error(f"Error creating Claude session: {str(e)}")
        return False


async def get_session(user_id: int, user_rules: Optional[str] = None, content_analysis_mode: bool = False) -> bool:
    """
    Gets an existing session or creates a new one
    
    Args:
        user_id: User's Telegram ID
        user_rules: User's personal rules/context
        content_analysis_mode: Whether to use content analysis mode
        
    Returns:
        True if session exists or was created successfully, False otherwise
    """
    # Check if Claude API is available
    if not CLAUDE_AVAILABLE:
        logging.warning(f"Claude API is not available. Cannot get/create session for user {user_id}")
        return False
        
    # Check if the current session is active
    if user_id in active_sessions:
        session_data = active_sessions[user_id]
        last_active = session_data.get("last_active", 0)
        
        # If the session has expired, create a new one
        if time.time() - last_active > SESSION_TIMEOUT:
            logging.info(f"Session for user {user_id} expired, creating new one")
            return await create_session(user_id, user_rules, content_analysis_mode)
        
        # Update the content analysis mode if specified
        current_mode = session_data.get("content_analysis_mode", False)
        if content_analysis_mode != current_mode:
            await set_content_analysis_mode(user_id, content_analysis_mode)
            
        # Update the last active time
        session_data["last_active"] = time.time()
        return True
    
    # If there is no session, create a new one
    return await create_session(user_id, user_rules, content_analysis_mode)


async def close_session(user_id: int) -> None:
    """
    Closes a Claude conversation session for a user
    
    Args:
        user_id: User's Telegram ID
    """
    logging.info(f"Closing Claude session for user {user_id}")
    # Remove the session from the active sessions dictionary
    if user_id in active_sessions:
        del active_sessions[user_id]


async def get_claude_completion(messages, max_tokens=2000, message=None, user_id=None, temperature=0.7, save_in_history=True):
    """
    Get completion from Claude API using the official Anthropic library
    
    Args:
        messages: List of message objects for the API
        max_tokens: Maximum tokens for response
        message: Original message object for error handling
        user_id: User ID for context preservation (if None, will extract from message)
        temperature: Temperature parameter for Claude (0.0 to 1.0), default 0.7
    
    Returns:
        String response from API
    """
    try:
        # Check if Claude API is available
        if not CLAUDE_AVAILABLE:
            error_msg = "Claude API is not available. Please set ANTHROPIC_API_KEY environment variable."
            logging.error(error_msg)
            if message:
                await message.answer(error_msg)
            return error_msg
        
        # Get user_id from message if not provided
        if user_id is None and message:
            user_id = message.from_user.id
        
        # If still no user_id, create a unique one for this request
        if user_id is None:
            user_id = f"temp_{uuid.uuid4()}"
            logging.warning(f"No user_id provided for Claude completion, using temp ID: {user_id}")
        
        # Extract system message and user message from messages
        system_content = ""
        user_content = ""
        
        for msg in messages:
            if msg['role'] == 'system':
                system_content = msg['content']
            elif msg['role'] == 'user':
                user_content = msg['content']
        
        # Get user rules if needed
        from redis_storage import RedisStorage
        redis_storage = RedisStorage()
        user_rules = await redis_storage.get_user_rules(user_id)
        
        # Get or create a session entry
        session_exists = await get_session(user_id, user_rules)
        
        if not session_exists:
            error_msg = "Не удалось создать сессию Claude"
            logging.error(error_msg)
            if message:
                await message.answer(error_msg)
            return error_msg
            
        # Update content analysis mode for this session
        if user_id in active_sessions:
            # Check if content analysis mode has changed
            current_mode = active_sessions[user_id].get("content_analysis_mode", False)
            
            # If content analysis mode is explicitly specified in the request, update the prompt
            for msg in messages:
                if msg['role'] == 'system' and 'анализирует документы' in msg['content']:
                    # If in system message, define prompt for document analysis,
                    # set content analysis mode and update system prompt
                    active_sessions[user_id]["content_analysis_mode"] = True
                    active_sessions[user_id]["system"] = msg['content']
                    logging.info("Detected document analysis prompt in message, updated system prompt")
                    break
        
        # Process system prompt - always use what was provided in the messages
        system_prompt = system_content if system_content else ""
            
        # Log system prompt for debugging
        logging.info(f"Using system prompt (first 100 chars): {system_prompt[:100]}...")
        if user_id in active_sessions:
            logging.info(f"Session content_analysis_mode: {active_sessions[user_id].get('content_analysis_mode', False)}")
            
        # Process messages for the Anthropic API
        anthropic_messages = []
        
        # IMPORTANT: Always use the provided messages as-is, regardless of type
        # This ensures calendar events and other special requests maintain proper context
        if len(messages) > 1:  # If messages were explicitly provided (standard case)
            # Copy all non-system messages (system messages are handled separately by the API)
            for msg in messages:
                if msg['role'] != 'system':  # System messages go through the separate system parameter
                    anthropic_messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
        else:
            # If only system message was provided, we need to add the user message
            anthropic_messages.append({"role": "user", "content": user_content})
        
        # Check if this is a new session or existing one
        is_new_session = user_id not in active_sessions or "messages" not in active_sessions[user_id]
        
        # For existing sessions, always update the session's last active time
        if user_id in active_sessions:
            active_sessions[user_id]["last_active"] = time.time()
            
            # Update session history based on the messages provided
            # Only store messages in history if save_in_history is True
            if save_in_history:
                if len(messages) > 1:
                    # Store all non-system messages in the session history
                    non_system_messages = [msg for msg in messages if msg['role'] != 'system']
                    active_sessions[user_id]["messages"] = non_system_messages
                    logging.info(f"Saved {len(non_system_messages)} messages to conversation history for user {user_id}")
                else:
                    # If only a system message was provided, append just the user message
                    if "messages" in active_sessions[user_id]:
                        active_sessions[user_id]["messages"].append({"role": "user", "content": user_content})
                        logging.info(f"Saved user message to conversation history for user {user_id}")
            
        try:
            # Log request parameters for debugging
            logging.info(f"Sending request to Claude: model={CLAUDE_MODEL}, max_tokens={max_tokens}")
            logging.info(f"System prompt length: {len(system_prompt)} chars")
            logging.info(f"Number of messages: {len(anthropic_messages)}")
            
            # Use the official Anthropic client to make the request
            # Extract system content from messages if present
            final_system_prompt = system_prompt
            final_messages = []
            
            # Process messages to extract system prompt and format user/assistant messages
            for msg in anthropic_messages:
                if msg['role'] == 'system':
                    # Use the most recent system prompt from messages
                    final_system_prompt = msg['content']
                else:
                    # Copy non-system messages
                    final_messages.append(msg)
            
            # Log the API call parameters
            logging.info(f"Calling Claude API with {len(final_messages)} messages")
            logging.info(f"System prompt length: {len(final_system_prompt) if final_system_prompt else 0}")
            
            # Make the API call with system as a separate parameter
            message_response = await claude_client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=max_tokens,
                temperature=temperature,
                system=final_system_prompt,
                messages=final_messages
            )
            
            # Extract the assistant's response
            assistant_response = message_response.content[0].text
            
            # Store the assistant response in the conversation history if save_in_history is True
            # This allows us to skip certain technical prompts without losing the main conversation
            if save_in_history and user_id in active_sessions:
                # Add assistant's response to session history
                if "messages" in active_sessions[user_id]:
                    active_sessions[user_id]["messages"].append({"role": "assistant", "content": assistant_response})
                else:
                    active_sessions[user_id]["messages"] = [{"role": "assistant", "content": assistant_response}]
                    
                logging.info(f"Saved assistant response to conversation history for user {user_id}")
                
            return assistant_response
            
        except Exception as e:
            logging.error(f"Error from Claude API: {str(e)}")
            if message:
                await message.answer(f"Произошла ошибка при отправке сообщения в Claude API: {str(e)}")
            return f"Ошибка Claude API: {str(e)}"
                
    except Exception as e:
        error_msg = f"Исключение при использовании Claude API: {str(e)}"
        logging.error(error_msg)
        if message:
            await message.answer(error_msg)
        return error_msg


async def start_cleanup_task():
    """
    Starts a background task that periodically cleans up expired sessions
    """
    asyncio.create_task(cleanup_expired_sessions())
    

async def cleanup_expired_sessions():
    """
    Periodically cleans up expired sessions
    """
    while True:
        try:
            current_time = time.time()
            expired_users = []
            
            for user_id, session_data in active_sessions.items():
                last_active = session_data.get("last_active", 0)
                if current_time - last_active > SESSION_TIMEOUT:
                    expired_users.append(user_id)
            
            for user_id in expired_users:
                logging.info(f"Removing expired session for user {user_id}")
                if user_id in active_sessions:
                    del active_sessions[user_id]
                    
        except Exception as e:
            logging.error(f"Error in cleanup_expired_sessions: {e}")
            
        # Wait for 5 minutes before next cleanup
        await asyncio.sleep(5 * 60)


async def ensure_session(user_id: int) -> bool:
    """
    Ensures that a user has an active session. If not, creates one.
    
    Args:
        user_id: User's Telegram ID
        
    Returns:
        True if a session exists or was successfully created, False otherwise
    """
    global active_sessions
    
    # Check if user already has an active session
    if user_id in active_sessions:
        # Update the last active timestamp
        active_sessions[user_id]['last_active'] = time.time()
        return True
    
    # Create a new session if one doesn't exist
    return await create_session(user_id)


async def is_file_generation_request(text: str) -> Tuple[bool, Optional[str]]:
    """
    Determine if the user request is related to file generation
    
    Args:
        text: User message
        
    Returns:
        Tuple of (is_file_request, file_type)
        Where file_type can be 'excel', 'pptx', 'doc', 'pdf', 'calendar' or None
    """
    if not text:
        return False, None
        
    text_lower = text.lower()
    
    # File generation keywords and patterns
    file_patterns = {
        'excel': [
            'создай эксель', 'сделай эксель', 'создай таблицу excel', 'сделай excel', 
            'создай xlsx', 'сгенерируй excel', 'создай таблицу в excel', 'xlsx файл',
            'таблицу в excel', 'таблицу эксель', 'excel таблицу'
        ],
        'pptx': [
            'создай презентацию', 'сделай презентацию', 'создай слайды', 'сделай слайды',
            'создай powerpoint', 'сделай powerpoint', 'создай pptx', 'сгенерируй pptx',
            'создай powerpoint презентацию', 'презентацию powerpoint', 'презентацию в powerpoint'
        ],
        'calendar': [
            'добавь в календарь', 'доабвь в календарь', 'доб в календарь', 'создай событие', 'создай календарное событие', 
            'добавь встречу', 'добавь событие', 'создай встречу', 'сохрани в календарь', 'в календарь',
            'сделай календарное событие', 'создай ics', 'создай ics файл', 'новое событие', 'новая встреча',
            'запиши в календарь', 'календарь', 'запланируй встречу', 'запланируй событие'
        ],
        'doc': [
            'создай документ', 'сделай документ', 'создай word', 'сделай word', 
            'создай docx', 'сгенерируй документ', 'создай текстовый документ',
            'документ word', 'документ в word'
        ],
        'pdf': [
            'создай pdf', 'сделай pdf', 'сгенерируй pdf', 'создай pdf документ',
            'создай pdf файл', 'сделай pdf документ', 'создай документ в pdf'
        ]
    }
    
    # Check for direct pattern matches first (optimization)
    for file_type, patterns in file_patterns.items():
        for pattern in patterns:
            if pattern in text_lower:
                logging.info(f"Direct match with pattern for type {file_type}: '{pattern}'")
                return True, file_type
    
    # If no direct match is found, use Claude for intent analysis
    try:
        logging.info(f"Checking with Claude for file creation intent in request: '{text}'")
        intent_result = await check_file_creation_intent(text)
        
        if intent_result:
            is_file_request, file_type = intent_result
            logging.info(f"Claude identified intent to create file: {is_file_request}, type: {file_type}")
            return is_file_request, file_type
    except Exception as e:
        logging.error(f"Error checking file creation intent: {str(e)}")
                
    return False, None


async def check_file_creation_intent(text: str, user_id: Optional[int] = None) -> Optional[Tuple[bool, Optional[str]]]:
    """
    Checks user's intent to create a file using Claude
    Args:
        text: User message
        user_id: Optional user ID to use existing session if available
        
    Returns:
        Tuple (has_file_creation_intent, file_type) or None on error
    """
    # Prompt for detecting file creation intent
    system_prompt = (
        "Проанализируй намерение пользователя создать файл. "
        "Определи: 1) содержит ли текст запрос на создание файла (ДА или НЕТ), "
        "2) если ДА, определи тип файла из списка: excel, pptx, doc, pdf, calendar. "
        "Отвечай в формате JSON: {'is_file_request': true/false, 'file_type': 'excel'/'pptx'/'doc'/'pdf'/'calendar'/null}. "
        "Вот примеры возможных типов файлов: "
        "- excel: таблица, расчеты, xls, xlsx, эксель, spreadsheet "
        "- pptx: презентация, слайды, доклад, powerpoint, ppt "
        "- doc: документ, текст, word, docx, текстовый файл "
        "- pdf: pdf-документ, пдф "
        "- calendar: событие, встреча, напоминание, календарная запись, ics, icalendar"
    )
    
    try:
        response_text = ""
        original_system = None
        
        # Use the existing user session if available
        if user_id and user_id in active_sessions:
            logging.info(f"Using existing session for file intent check with user_id={user_id}")
            try:
                # Save the original system prompt
                if 'system' in active_sessions[user_id]:
                    original_system = active_sessions[user_id]['system']
                    logging.info("Saved original system prompt")
                
                # Create a special request to check intent WITHOUT modifying the main session
                # Important: Do NOT save this request to the session history
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}
                ]
                
                # Create a separate request without saving to session history
                client = AsyncAnthropic(api_key=CLAUDE_API_KEY)
                message_response = await client.messages.create(
                    model=CLAUDE_MODEL,
                    max_tokens=100,
                    temperature=0.1,
                    system=system_prompt,
                    messages=[{"role": "user", "content": text}]
                )
                response_text = message_response.content[0].text
                logging.info(f"Got file intent check response: {response_text[:50]}...")
            except Exception as e:
                logging.error(f"Error checking file intent with existing session: {str(e)}")
                # Continue with the fallback method
        
        # If we couldn't use an existing session or there isn't one
        if not response_text:
            logging.info("Using standalone request for file intent check")
            try:
                # Standalone request without connection to any session
                client = AsyncAnthropic(api_key=CLAUDE_API_KEY)
                message_response = await client.messages.create(
                    model=CLAUDE_MODEL,
                    max_tokens=100,
                    temperature=0.1,
                    system=system_prompt,
                    messages=[{"role": "user", "content": text}]
                )
                response_text = message_response.content[0].text
            except Exception as e:
                logging.error(f"Error in standalone file intent check: {str(e)}")
        
        # Extract JSON from the response
        if response_text:
            try:
                import json
                import re
                # Search for JSON object in the response using regular expression
                json_match = re.search(r'\{.*?\}', response_text, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group(0))
                    is_file_request = result.get('is_file_request', False)
                    file_type = result.get('file_type', None) if is_file_request else None
                    return is_file_request, file_type
            except Exception as json_err:
                logging.error(f"Error parsing JSON from file intent check: {str(json_err)}. Response: {response_text[:100]}")
    except Exception as e:
        logging.error(f"Error in check_file_creation_intent: {str(e)}")
    
    return None


async def is_content_analysis_mode(user_id: int) -> bool:
    """
    Check if the session is in content analysis mode
    
    Args:
        user_id: User's Telegram ID
        
    Returns:
        True if content analysis mode is enabled, False otherwise
    """
    if user_id in active_sessions:
        return active_sessions[user_id].get("content_analysis_mode", False)
    return False


async def set_content_analysis_mode(user_id: int, enabled: bool = True) -> None:
    """
    Set content analysis mode for a session
    
    Args:
        user_id: User's Telegram ID
        enabled: Whether to enable or disable content analysis mode
    """
    if user_id in active_sessions:
        active_sessions[user_id]["content_analysis_mode"] = enabled


async def set_system_prompt(user_id: int, system_prompt: str) -> None:
    """
    Set system prompt for a session
    
    Args:
        user_id: User's Telegram ID
        system_prompt: System prompt to set
    """
    if user_id in active_sessions:
        active_sessions[user_id]["system"] = system_prompt
        logging.info(f"Set custom system prompt for user {user_id}")


async def close_session(user_id: int) -> bool:
    """
    Closes an active Claude session for a user
    
    Args:
        user_id: User's Telegram ID
        
    Returns:
        True if session was closed, False if no session existed
    """
    if user_id in active_sessions:
        # Remove the session
        del active_sessions[user_id]
        logging.info(f"Closed Claude session for user {user_id}")
        return True
    else:
        logging.info(f"No active Claude session found for user {user_id} to close")
        return False
        

async def check_if_new_conversation(user_id: int, message_text: str, user_rules: Optional[str] = None) -> bool:
    """
    Check if the message should start a new conversation rather than continue the existing one.
    
    According to simplified rules, a new conversation is started only in specific cases:
    1. If there's no active session for the user
    2. If the message starts with the /new command
    3. If message contains forwarded content (handled in message_handlers.py)
    4. If message contains documents/links (handled in message_handlers.py)
    
    Args:
        user_id: User's Telegram ID
        message_text: The message text to analyze
        user_rules: User's personal rules/context
        
    Returns:
        True if this should be treated as a new conversation, False otherwise
    """
    # If there's no active session, it's definitely a new conversation
    if user_id not in active_sessions:
        logging.info(f"No active session for user {user_id}, starting new conversation")
        return True
        
    # If there are no previous messages, treat as new conversation
    if not active_sessions[user_id].get("messages", []):
        logging.info(f"No previous messages for user {user_id}, starting new conversation")
        return True
        
    # Check if message starts with /new command
    if message_text.strip().startswith("/new"):
        logging.info(f"User {user_id} used /new command, starting new conversation")
        return True
        
    # For all other cases, consider it a continuation of the existing conversation
    return False
        
        
async def send_message(user_id: int, message: str, content_analysis_mode: bool = False, user_rules: Optional[str] = None) -> Tuple[str, bool]:
    """
    Sends a message to Claude and gets a response using the official Anthropic library
    
    Args:
        user_id: User's Telegram ID
        message: Message text to send
        content_analysis_mode: Content analysis mode for documents/URLs
        user_rules: User's personal rules/context
        
    Returns:
        Tuple of (response text, is_new_chat flag)
    """
    try:
        if not CLAUDE_AVAILABLE:
            return "Claude API is not available. Please check your API key.", False

        # Check if conversation is new or existing
        is_new_chat = await check_if_new_conversation(user_id, message, user_rules)

        # If it's a new conversation, close old session and create new one
        if is_new_chat:
            await close_session(user_id)
            session_created = await create_session(user_id, user_rules, content_analysis_mode)
            if not session_created:
                return "Failed to create new Claude session", True
        else:
            # For existing conversations, just get or update session
            session_exists = await get_session(user_id, user_rules, content_analysis_mode)
            if not session_exists:
                return "Failed to retrieve Claude session", False
            
        # Set content analysis mode
        await set_content_analysis_mode(user_id, content_analysis_mode)
        
        # Get system prompt from session
        system_prompt = ""
        if user_id in active_sessions and "system" in active_sessions[user_id]:
            system_prompt = active_sessions[user_id]["system"]
            
        # Prepare existing conversation history
        anthropic_messages = []
        if user_id in active_sessions and "messages" in active_sessions[user_id]:
            anthropic_messages = active_sessions[user_id]["messages"]
            
        # Add current user message
        anthropic_messages.append({"role": "user", "content": message})
        
        # Save updated conversation history
        if user_id in active_sessions:
            active_sessions[user_id]["messages"] = anthropic_messages
            active_sessions[user_id]["last_active"] = time.time()
        
        try:
            # Use the official Anthropic client
            response = await claude_client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=4000,
                system=system_prompt,
                messages=anthropic_messages
            )
            
            # Extract the assistant's response
            content = response.content[0].text
            
            # Store the assistant response in conversation history
            if user_id in active_sessions:
                active_sessions[user_id]["messages"].append({"role": "assistant", "content": content})
                
            return content, is_new_chat
            
        except Exception as e:
            logging.error(f"Error from Claude API in send_message: {str(e)}")
            
            # Try to recreate the session and try again
            logging.info(f"Trying to recreate session and send message again")
            await close_session(user_id)
            await create_session(user_id, user_rules)
            
            try:
                # Try one more time with a fresh session
                if user_id in active_sessions:
                    system_prompt = active_sessions[user_id]["system"]
                    
                    # Create a new request with only the user message
                    retry_messages = [{"role": "user", "content": message}]
                    
                    # Make API call using the same system prompt
                    response = await claude_client.messages.create(
                        model=CLAUDE_MODEL,
                        max_tokens=4000,
                        system=system_prompt,  # System prompt as a separate parameter
                        messages=retry_messages
                    )
                    
                    # Extract and store the assistant's response
                    content = response.content[0].text
                    
                    # Store in conversation history
                    if user_id in active_sessions:
                        active_sessions[user_id]["messages"] = [
                            {"role": "user", "content": message},
                            {"role": "assistant", "content": content}
                        ]
                    
                    return content, True  # True indicates this is a new conversation
                else:
                    logging.error("Second attempt failed: Failed to create session")
                    return "Failed to create Claude session on second attempt", False
                    
            except Exception as e2:
                logging.error(f"Second attempt failed: {str(e2)}")
                return f"Error from Claude API (second attempt): {str(e2)}", False
                
    except Exception as e:
        logging.error(f"Exception in send_message: {str(e)}")
        return f"Error: {str(e)}", False
