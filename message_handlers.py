import logging
import time
import os
from datetime import datetime
import asyncio
from typing import Set

# Global set to track users currently being processed to prevent duplicate processing
processing_users: Set[int] = set()

from aiogram import types, F
from aiogram.types import Message, CallbackQuery, InlineKeyboardMarkup, InlineKeyboardButton, FSInputFile
from aiogram.fsm.context import FSMContext
from aiogram.exceptions import TelegramAPIError

from claude_utils import active_sessions, set_content_analysis_mode, get_claude_completion, check_if_new_conversation, check_file_creation_intent, ensure_session, close_session, send_message
from redis_storage import RedisStorage
from states import BotStates
from config import message_timers, collected_content
# Import content handling functions
from content_handlers import handle_document, handle_url, extract_url_from_text
# Import voice handler
from voice_handler import handle_voice_message
# Import teamlead handlers
from teamlead_handlers import process_teamlead_command

async def process_and_send_ai_response(message: Message, response: str, summary_needed: bool = False, state=None, content_analysis_mode: bool = False):
    """Process AI response and send to user with appropriate formatting"""
    if response:
        # Extract user_id for tracking and logging
        user_id = message.from_user.id
        
        # Default modes
        variant_mode = True
        
        # Get modes from state if provided
        if state:
            data = await state.get_data()
            variant_mode = data.get('variant_mode', True)
            
        # If content_analysis_mode is not passed as a parameter, try to get it from state
        if state and not content_analysis_mode:
            data = await state.get_data()
            content_analysis_mode = data.get('content_analysis_mode', False)
            
        # Check mode with Claude API
        if not content_analysis_mode:
            try:
                from claude_utils import is_content_analysis_mode
                content_analysis_mode = await is_content_analysis_mode(user_id)
            except Exception as e:
                logging.error(f"Error checking content_analysis_mode from Claude API: {e}")
                
        # Process and send AI response using message_processing module
        from message_processing import process_and_send_ai_response as process_response
        await process_response(message, response, summary_needed=summary_needed, state=state, content_analysis_mode=content_analysis_mode)

async def handle_file_generation(message: Message, file_type: str, user_id: int, reformulated_text: str = None):
    """
    Handles file generation requests while maintaining conversation flow
    
    Args:
        message: User message
        file_type: Type of file to generate
        user_id: User's Telegram ID
        reformulated_text: Optional reformulated version of the user's request (for AI-detected requests)
    """
    logging.info(f"Handling file generation request: {file_type}")
    from file_generator import generate_file
    
    # Send message to user about file generation - more natural messaging
    creation_messages = {
        'excel': "Создаю таблицу для вас...",
        'pptx': "Готовлю презентацию...",
        'doc': "Создаю документ...",
        'pdf': "Формирую PDF файл...",
        'calendar': "Создаю событие для вашего календаря..."
    }
    
    creation_message = creation_messages.get(file_type, f"Создаю файл в формате {file_type.upper()}...")
    working_msg = await message.answer(creation_message)
    
    # Send loading indicator while generating
    await message.bot.send_chat_action(chat_id=message.chat.id, action="upload_document")
    
    try:
        # Generate file using current user session or reformulated version
        input_text = reformulated_text if reformulated_text else message.text
        logging.info(f"Generating {file_type} file using text: {input_text}")
        
        # Get conversation history from the user's active session
        from claude_utils import active_sessions
        conversation_history = []
        if user_id in active_sessions and "messages" in active_sessions[user_id]:
            conversation_history = active_sessions[user_id]["messages"]
            logging.info(f"Retrieved conversation history with {len(conversation_history)} messages for file generation")
        else:
            logging.warning("No conversation history found, generating file without context")
        
        try:
            # Generate the file - this will raise an exception if it fails
            result = await generate_file(input_text, file_type, user_id, conversation_history)
            
            # Check if the result is a list of files (for calendar events)
            if isinstance(result, list) and all(os.path.exists(path) for path in result):
                # For calendars, send multiple files
                logging.info(f"Sending {len(result)} calendar files")
                
                # Send files directly without additional messages
                for file_path in result:
                    file_name = os.path.basename(file_path)
                    platform = "iOS" if "ios" in file_name.lower() else "Android" if "android" in file_name.lower() else "Outlook"
                    
                    # Create FSInputFile object for the file from filesystem
                    file_obj = FSInputFile(path=file_path)
                    
                    # Send file without caption
                    try:
                        await message.answer_document(
                            document=file_obj
                            # No caption to avoid extra text
                        )
                    except Exception as err:
                        logging.error(f"Error sending file {file_name}: {str(err)}")
                    
                    # Delete file regardless of success or failure
                    try:
                        os.remove(file_path)
                    except Exception as cleanup_err:
                        # Error message when deleting a file
                        logging.error(f"Error deleting file {file_path}: {str(cleanup_err)}")
                
                # Add files to a single document to avoid separate instruction message
                # After sending all files, send a brief message
                await message.answer("Файлы календаря готовы. Выберите файл для вашего устройства (iOS, Android или Outlook), скачайте и откройте его, чтобы импортировать событие в календарь.")
                
                # Delete the working message after sending all calendar files
                try:
                    await working_msg.delete()
                    logging.info("Deleted working message after sending calendar files")
                except Exception as e:
                    logging.error(f"Failed to delete working message: {e}")
                    
                # We've modified our approach to maintain the conversation context automatically
                # The API calls now don't disrupt the chat history by using save_in_history=False for technical requests
                # This ensures the main conversation stays intact while generating files
            
            # For regular files - standard processing
            elif result and os.path.exists(result):
                file_name = os.path.basename(result)
                file_obj = FSInputFile(path=result)
                
                # Send document without additional text
                await message.answer_document(
                    document=file_obj
                )
                
                # Delete temporary file
                os.remove(result)
                
                # Delete the working message after sending the file
                try:
                    await working_msg.delete()
                    logging.info(f"Deleted working message after sending file")
                except Exception as e:
                    logging.error(f"Failed to delete working message: {e}")
            else:
                # This shouldn't happen now that we're using exceptions for error handling
                # but keeping as a fallback
                raise Exception(f"File generation failed for {file_type}")
                
        except Exception as file_error:
            # Import error handler from file_generator
            from file_generator import handle_file_generation_error
            error_message = await handle_file_generation_error(file_type, str(file_error))
            
            # Return user-friendly error message
            await message.answer(error_message)
            
            # Delete the working message
            try:
                await working_msg.delete()
            except Exception:
                pass
                
    except Exception as e:
        logging.error(f"Error in handle_file_generation: {str(e)}")
        # Use a more general error without details to avoid issues with displaying objects in the message
        await message.answer("Произошла ошибка при создании файла. Пожалуйста, уточните запрос и попробуйте снова.")
        
        # Delete the working message if it exists
        try:
            await working_msg.delete()
        except Exception:
            pass

async def process_claude_message(message: Message, state: FSMContext, user_id: int, user_rules: str):
    """Send message to Claude and process response"""
    try:
        # Get content analysis mode from Claude API
        from claude_utils import is_content_analysis_mode, send_message, set_system_prompt
        content_analysis_mode = await is_content_analysis_mode(user_id)
        
        # Use different system prompts based on mode
        if content_analysis_mode:
            # For content analysis mode (documents, URLs)
            system_prompt = "You are an analytical assistant analyzing documents and links. "
            system_prompt += "Your task is to provide a BRIEF, SUBSTANTIVE SUMMARY of the provided content. "
            system_prompt += "DO NOT split your response into variants. Give ONE complete substantive response. "
            system_prompt += "If the user asks a question about the document, give a direct answer without unnecessary words. "
            system_prompt += "If the document contains key facts (dates, names, places, numbers) - highlight them. "
            system_prompt += "DO NOT use the ### separator. The entire response should be a single text. "
            system_prompt += "THIS IS CRITICALLY IMPORTANT: You MUST carefully analyze the tone and level of formality in the analyzed documents/messages. "
            system_prompt += "Pay special attention to the use of formal or informal address forms. "
            system_prompt += "If the user's message or content uses an informal tone (for example, casual expressions or informal language), "
            system_prompt += "you MUST use THE SAME INFORMAL STYLE in your responses. "
            system_prompt += "If the user's message or content uses a formal tone (for example, business-like expressions or formal language), "
            system_prompt += "you MUST use THE SAME FORMAL STYLE in your responses. "
            system_prompt += "ALSO EXACTLY MATCH other style elements: conversational/formal tone, specialized terminology, slang, professional jargon."
            
            # Set this system prompt for the session
            await set_system_prompt(user_id, system_prompt)
        
        # Send typing indicator
        await message.bot.send_chat_action(chat_id=message.chat.id, action="typing")
        
        # Send message to Claude
        response, is_new_chat = await send_message(user_id, message.text, content_analysis_mode, user_rules)
        
        # Get state data
        data = await state.get_data()
        variant_mode = data.get('variant_mode', True)
        summary_needed = data.get('summary_needed', False)
        working_message_id = data.get('working_message_id')
        
        # Delete working message if it exists
        if working_message_id:
            try:
                await message.bot.delete_message(chat_id=message.chat.id, message_id=working_message_id)
                await state.update_data(working_message_id=None)
            except Exception as e:
                logging.error(f"Error deleting working message: {str(e)}")
        
        # Log the settings being used
        logging.info(f"Processing message with: summary_needed={summary_needed}, variant_mode={variant_mode}, content_analysis_mode={content_analysis_mode}")
        
        # Use processing function from message_processing
        from message_processing import process_and_send_ai_response as process_response
        # Variant mode is taken from the state, no need to pass it to the function
        await process_response(message, response, summary_needed=summary_needed, state=state, content_analysis_mode=content_analysis_mode)
        
        # Set continuation flag to False for next message
        await state.update_data(is_continuation=False)
        
    except Exception as e:
        logging.error(f"Error in process_claude_message: {e}")
        await message.answer(f"Произошла ошибка при обработке сообщения: {str(e)}")

async def cancel_timer(user_id):
    """Cancel any existing timers for a user"""
    if user_id in message_timers and message_timers[user_id]:
        try:
            # Cancel the timer task
            message_timers[user_id].cancel()
            message_timers[user_id] = None
        except Exception as e:
            logging.error(f"Error canceling timer for user {user_id}: {e}")

async def process_forwarded_messages_after_timeout(user_id, message, state):
    """Process collected messages after a timeout"""
    # Wait for 1 second
    await asyncio.sleep(1)
    
    # Clear timer reference
    message_timers[user_id] = None
    
    # Get state data
    data = await state.get_data()
    forwarded_messages = data.get('forwarded_messages', [])
    
    # Check if we have messages to process
    if forwarded_messages:
        msg_count = len(forwarded_messages)
        
        # Inform user
        if msg_count > 1:
            await message.answer(f"Получено {msg_count} сообщений. Начинаю анализ...")
        
        # Process the forwarded messages
        await analyze_forwarded_messages(message, state)
    else:
        logging.warning(f"No forwarded messages found for user {user_id} after timeout")

async def collect_forwarded_message(message: Message, state: FSMContext):
    """
    Collect a forwarded message and start/reset timer for batch processing
    
    Args:
        message: Forwarded message object
        state: FSM context
    """
    # Get sender information
    user_id = message.from_user.id
    
    # Get state data
    data = await state.get_data()
    forwarded_messages = data.get('forwarded_messages', [])
    
    # Create a simplified message object for storage
    forwarded_from = None
    if message.forward_from:
        forwarded_from = message.forward_from.first_name
    elif message.forward_sender_name:
        forwarded_from = message.forward_sender_name
    else:
        forwarded_from = "Unknown"
    
    # Create a dictionary with essential information
    simplified_message = {
        "from": forwarded_from,
        "text": message.text or message.caption or "[Нетекстовое сообщение]"
    }
    
    # Add to the list
    forwarded_messages.append(simplified_message)
    
    # Calculate total length of all collected messages
    from config import MAX_SUMMARY_THRESHOLD
    total_content_length = 0
    for msg in forwarded_messages:
        total_content_length += len(msg.get('text', ''))
    
    # Set summary_needed flag if total length exceeds threshold or multiple messages
    summary_needed = total_content_length > MAX_SUMMARY_THRESHOLD or len(forwarded_messages) > 1
    
    # Update state with messages and summary flag
    await state.update_data(
        forwarded_messages=forwarded_messages,
        summary_needed=summary_needed
    )
    
    # Log the determination
    if summary_needed:
        logging.info(f"Summary needed for forwarded messages. Total length: {total_content_length}, message count: {len(forwarded_messages)}")
    
    # Cancel any existing timer
    await cancel_timer(user_id)
    
    # Set a new timer (5 seconds)
    message_timers[user_id] = asyncio.create_task(
        process_forwarded_messages_after_timeout(user_id, message, state)
    )
    
    # Do not send a notification about the received message
    # Simply collect forwarded messages in a batch
    pass

async def analyze_forwarded_messages(message: Message, state: FSMContext, from_callback=False, from_direct_message=False):
    """
    Analyze a batch of collected forwarded messages and URLs
    
    Args:
        message: Original message that triggered the analysis
        state: FSM context
        from_callback: Whether this was triggered by a callback button
        from_direct_message: Whether this was triggered by a direct message
    """
    user_id = message.from_user.id
    
    # Prevent duplicate processing
    if user_id in processing_users:
        return
    
    processing_users.add(user_id)
    
    try:
        # Get state data
        data = await state.get_data()
        forwarded_messages = data.get('forwarded_messages', [])
        
        if not forwarded_messages:
            if not from_direct_message:
                await message.answer("Нет сообщений для анализа.")
            processing_users.discard(user_id)
            return
            
        # Close any existing session for new forwarded messages
        from claude_utils import close_session, active_sessions
        if user_id in active_sessions:
            logging.info(f"New messages detected - closing existing session for user {user_id}")
            await close_session(user_id)
        
        # Show working message
        working_msg = None
        if not from_callback:
            working_msg = await message.answer("Анализирую сообщения...")
        
        # Process messages and URLs
        formatted_messages = []
        total_content_length = 0
        urls = []
        
        for msg in forwarded_messages:
            # Handle URL messages
            if msg.get('contains_url') and 'url' in msg:
                url = msg['url']
                urls.append(url)
                formatted_message = f"[Ссылка] {url}"
            # Handle regular text messages
            else:
                sender = msg.get('from_user', 'Неизвестно')
                text = msg.get('text', '')
                formatted_message = f"{sender}: {text}"
            
            formatted_messages.append(formatted_message)
            total_content_length += len(str(msg.get('text', '')))
        
        all_content = "\n\n".join(formatted_messages)
        
        # Check if we need to generate a summary first
        from config import MAX_SUMMARY_THRESHOLD
        needs_summary = total_content_length > MAX_SUMMARY_THRESHOLD or len(forwarded_messages) > 1 or bool(urls)
        
        # Get user-specific rules
        redis_storage = RedisStorage()
        user_rules = await redis_storage.get_user_rules(user_id) or ""
        
        # Build system prompt
        summary_instruction = ""
        if needs_summary:
            summary_instruction = "Сначала предоставь краткое резюме (1-3 предложения). Затем, после двойного переноса строки, "
            logging.info(f"Adding summary instruction for user {user_id} (content length: {total_content_length}, messages: {len(forwarded_messages)})")
        
        # Different prompts based on content type
        has_text = any('text' in msg and msg['text'].strip() for msg in forwarded_messages)
        has_documents = any('document' in msg for msg in forwarded_messages)
        mixed_content = (urls and has_text) or (has_documents and (has_text or urls))
        
        if urls and not has_text and not has_documents:
            # Only URLs - use content analysis mode without variants
            system_prompt = (
                "Ты - ассистент, который анализирует веб-страницы по предоставленным URL. "
                "Предоставь краткий, но информативный анализ содержания. "
                "Не предлагай варианты ответов, просто дай анализ."
            )
            content_analysis_mode = True
            variant_mode = False
        elif has_documents and not has_text and not urls:
            # Only documents - already handled by document processor
            processing_users.discard(user_id)
            return
        else:
            # Text messages or mixed content
            system_prompt = (
                "Ты - ассистент, который анализирует сообщения и предлагает варианты ответов. "
                f"{summary_instruction}предложи 2-4 варианта ответа, разделенных символами ###. "
                "ОБЯЗАТЕЛЬНО предоставь минимум 2 разных варианта. "
                "Анализируй тон и уровень формальности в сообщениях. "
                "Если в сообщениях используется неформальное обращение на 'ТЫ', используй такой же стиль. "
                "Если используется формальное обращение на 'ВЫ', сохраняй формальность. "
                "Ответы должны быть краткими, но информативными, и соответствовать тону беседы. "
                "Не нумеруй и не отмечай варианты ответов. "
                "Просто предоставь сами ответы, разделенные символами ###."
            )
            content_analysis_mode = False
            variant_mode = True
        
        # Add user rules if available
        if user_rules:
            system_prompt += f"\n\nПравила пользователя: {user_rules}"
        
        # Create messages for AI
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Проанализируй следующее и предложи варианты ответа (не нумеруй их и не добавляй префиксы):\n\n{all_content}"}
        ]
        
        # Set flags for response processing based on content type
        await state.update_data({
            'content_analysis_mode': content_analysis_mode,
            'variant_mode': variant_mode,
            'needs_summary': needs_summary,
            'mixed_content_mode': mixed_content if 'mixed_content' in locals() else False
        })
        
        # Get AI completion
        response = await get_claude_completion(messages, max_tokens=2000, message=message)
        
        # Delete working message if it exists
        if working_msg:
            await working_msg.delete()
        
        # Store conversation in state for context
        await state.update_data(conversation=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": all_content},
            {"role": "assistant", "content": response}
        ])
        
        # Get summary needed flag from state
        summary_needed = data.get('summary_needed', False)
        logging.info(f"Processing forwarded messages with summary_needed={summary_needed}")
        
        # Process and send AI response with summary if needed
        await process_and_send_ai_response(message, response, summary_needed=summary_needed, state=state)
        
    except Exception as e:
        logging.error(f"Error analyzing forwarded messages: {e}")
        await message.answer(f"Произошла ошибка при анализе сообщений: {str(e)}")
    
    finally:
        # Remove from processing users
        processing_users.discard(user_id)
        
        # Reset the collected messages
        await state.update_data(forwarded_messages=[])

async def handle_clarification(message: Message, state: FSMContext):
    """Handle clarification or follow-up messages, including file generation requests"""
    # Get user ID and prevent duplicate processing
    user_id = message.from_user.id
    
    if user_id in processing_users:
        return
    
    processing_users.add(user_id)
    
    try:
        # Get state data with conversation history
        data = await state.get_data()
        conversation = data.get('conversation', [])
        
        if not conversation:
            processing_users.discard(user_id)
            return
        
        # Set continuation flag to prevent summary
        await state.update_data(is_continuation=True)
            
        # Add user's clarification to the conversation
        clarification_text = message.text
        
        # Check if this is a file creation request
        from file_generator import process_file_request, create_file, generate_with_fallback
        
        # Two ways to detect file creation requests
        file_creation_result = await check_file_creation_intent(clarification_text, user_id)
        file_creation_request = None
        if file_creation_result and file_creation_result[0]:
            file_creation_request = {'file_type': file_creation_result[1]}
        file_info = await process_file_request(clarification_text, conversation)
        
        # Use any successful detection method
        if file_creation_request and not file_info:
            file_info = {
                'file_type': file_creation_request['file_type'],
                'file_name': f"generated_{file_creation_request['file_type']}.{file_creation_request['file_type']}",
                'user_request': file_creation_request['request_text']
            }
            
        if file_info:  # If this is a file creation request
            # Show that we're preparing the file
            working_msg = await message.answer(
                f"Подготавливаю файл {file_info['file_type']} по вашему запросу... Это может занять некоторое время."
            )
            
            # Update conversation history
            conversation.append({"role": "user", "content": clarification_text})
            
            # Special handling for calendar files
            if file_info['file_type'] == 'calendar':
                assistant_response = "Подготавливаю файлы события для добавления в календарь."
                
                # Add conversation context for event details extraction
                # (conversation history уже содержит всю информацию, включая пересланные сообщения)
                file_info['conversation'] = conversation
            else:
                assistant_response = f"Конечно, я подготовлю для вас файл {file_info['file_type']} по указанной теме."
                
            conversation.append({"role": "assistant", "content": assistant_response})
            await state.update_data(conversation=conversation)
            
            # Try to create the file
            file_path = await create_file(file_info)
            
            # If script creation failed, try fallback generation
            if not file_path:
                file_path = await generate_with_fallback(file_info, user_id=user_id)
            
            # Delete temporary message
            await working_msg.delete()
            
            # If file was created successfully, send it
            if file_path and os.path.exists(file_path):
                # Special handling for calendar files
                if file_info['file_type'] == 'calendar' and 'calendar_files' in file_info:
                    # First send a message with event details
                    event = file_info.get('event_details', {})
                    
                    event_summary = event.get('summary', 'Событие')
                    event_start = event.get('start_time', '')
                    event_end = event.get('end_time', '')
                    event_location = event.get('location', 'Не указано')
                    
                    event_message = f"Я создал файлы события для календаря:\n"
                    event_message += f"\nНазвание: {event_summary}"
                    if event_start:
                        event_message += f"\nНачало: {event_start}"
                    if event_end:
                        event_message += f"\nОкончание: {event_end}"
                    if event_location:
                        event_message += f"\nМесто: {event_location}"
                    
                    await message.answer(event_message)
                    
                    # Send each calendar file
                    total_calendar_files = len(file_info['calendar_files'])
                    logging.info(f"Sending {total_calendar_files} calendar files to user {message.from_user.id}")
                    
                    for i, cal_file in enumerate(file_info['calendar_files'], 1):
                        # Write to temporary file
                        temp_path = f"/tmp/telegram_bot_files/{cal_file['filename']}"
                        logging.info(f"Creating calendar file {i}/{total_calendar_files}: {temp_path}")
                        
                        # Save the file to disk
                        with open(temp_path, 'wb') as f:
                            f.write(cal_file['content'])
                        
                        # Send the file
                        try:
                            with open(temp_path, 'rb') as f:
                                await message.answer_document(
                                    FSInputFile(temp_path, filename=cal_file['filename'])
                                )
                            logging.info(f"Successfully sent calendar file {i}/{total_calendar_files} to user {message.from_user.id}")
                        except Exception as e:
                            logging.error(f"Failed to send calendar file {i}/{total_calendar_files}: {str(e)}")
                        
                        # Delete temporary file
                        try:
                            os.remove(temp_path)
                        except Exception as e:
                            logging.error(f"Failed to delete temporary file {temp_path}: {str(e)}")
                            
                # Delete the status message after successful file generation
                try:
                    await working_msg.delete()
                    logging.info(f"Deleted working message for calendar generation")
                except Exception as e:
                    logging.error(f"Failed to delete working message: {e}")
            else:
                # If file creation failed, notify the user
                logging.error(f"Failed to create file of type {file_info['file_type']}")
                await message.answer(
                    "К сожалению, не удалось создать файл. Пожалуйста, уточните ваш запрос или попробуйте позже."
                )
            
            # File creation request handling complete
            return
        
        # Standard clarification handling
        working_msg = await message.answer("Analyzing your follow-up...")
        
        # Get user rules if any
        redis_storage = RedisStorage()
        user_rules = await redis_storage.get_user_rules(user_id) or ""
        
        # Check dialog mode (with variants or without)
        # Strict logic to avoid losing content analysis mode
        content_analysis_mode = data.get('content_analysis_mode', False)
        
        # After analyzing document without explanations we always respond without variants
        if content_analysis_mode:
            variant_mode = False
            logging.info("Content analysis mode detected - forcing dialog WITHOUT variants")
        else:
            variant_mode = data.get('variant_mode', True)
            logging.info(f"Standard mode detected - using variant_mode={variant_mode}")
            
        # Save flag values in state
        await state.update_data(content_analysis_mode=content_analysis_mode, variant_mode=variant_mode)
        
        # Add system message with rules if we have them
        system_message = "You are an assistant that answers user questions, continuing the previous dialog. Answer in Russian."
        system_message += "\n\nIf documents or texts were mentioned in the previous context, answer user questions based on their content."
        system_message += "\n\nResponses should be complete but concise, based on information from the previous context."
        
        # Add instruction about response format based on dialog mode
        if variant_mode:
            system_message += "\n\nSuggest 2-4 response options, separated by ### symbols. Do not number the options or add prefixes to them."
        else:
            system_message += "\n\nProvide only one concise and informative response. DO NOT suggest response options and do not use ### separators."
        
        if user_rules:
            system_message += f"\n\nПравила пользователя: {user_rules}"
        
        # Create new conversation with history + clarification
        conversation.append({"role": "user", "content": clarification_text})
        
        # Add system message at the beginning
        full_conversation = [
            {"role": "system", "content": system_message},
        ] + conversation
        
        # Get AI completion
        response = await get_claude_completion(full_conversation, max_tokens=2000, message=message)
        
        # Update conversation history
        conversation.append({"role": "assistant", "content": response})
        await state.update_data(conversation=conversation)
        
        # Delete working message
        await working_msg.delete()
        
        # Process and send AI response without summary (we're in continuation mode)
        logging.info(f"Processing clarification response without summary for user {user_id}")
        await process_and_send_ai_response(message, response, state=state)
        
    except Exception as e:
        logging.error(f"Error handling clarification: {e}")
        await message.answer(f"Произошла ошибка при обработке уточнения: {str(e)}")
    
    finally:
        # Remove from processing users
        processing_users.discard(user_id)

async def handle_message(message: Message, state: FSMContext):
    """Main handler for all messages (forwarded, text, documents, voice)"""
    user_id = message.from_user.id
    # Initialize Redis storage
    redis_storage = RedisStorage()
    user_rules = await redis_storage.get_user_rules(user_id)
    
    # Get the bot instance
    bot = message.bot
    
    # Handle voice messages
    if message.voice:
        logging.info(f"Received voice message from user {user_id}")
        transcribed_text = await handle_voice_message(message, bot)
        if transcribed_text:
            # Process transcribed text without modifying frozen message object
            logging.info(f"Processing transcribed voice text: {transcribed_text[:50]}...")
            
            # Create a text-like message data for processing
            # Try to process as teamlead command first
            class VoiceMessage:
                def __init__(self, original_message, text):
                    self._original_message = original_message
                    self.text = text
                    self.from_user = original_message.from_user
                    self.chat = original_message.chat
                    self.message_id = original_message.message_id
                    self.date = original_message.date
                    self.bot = original_message.bot
                    self.reply_to_message = original_message.reply_to_message
                    
                async def answer(self, *args, **kwargs):
                    return await self._original_message.answer(*args, **kwargs)
                    
                async def answer_photo(self, *args, **kwargs):
                    return await self._original_message.answer_photo(*args, **kwargs)
                    
                async def answer_document(self, *args, **kwargs):
                    return await self._original_message.answer_document(*args, **kwargs)
            
            voice_msg = VoiceMessage(message, transcribed_text)
            
            # Try to process as teamlead command
            if await process_teamlead_command(voice_msg, state):
                logging.info(f"Processed voice as teamlead command: {transcribed_text}")
                return
                
            # If not a command, process as regular message
            await process_claude_message(voice_msg, state, user_id, user_rules)
            return
        else:
            # Voice processing failed
            return

    # Handle forwarded messages
    if message.forward_from or message.forward_sender_name:
        # Check if it's a forwarded document
        if message.document or message.photo:
            await handle_document(message, state, bot)
        # Handle forwarded text messages
        elif message.text or message.caption:
            # Check if forwarded message contains just a URL
            message_text = message.text or message.caption or ""
            url = await extract_url_from_text(message_text)
            
            if url and (message_text.strip() == url or message_text.strip() == f"{url} "):
                # Handle forwarded URL as a document
                logging.info(f"Processing forwarded URL: {url}")
                content_info = {
                    'type': 'url',
                    'url': url,
                    'message_text': message_text.strip(),
                    'is_plain_url': True,
                    'is_forwarded': True,
                    'from_user': message.forward_from.first_name if message.forward_from else "Unknown"
                }
                # Collect for batched processing (1-second timer)
                from content_handlers import collect_content
                await collect_content(message, state, content_info)
            else:
                # Handle as regular forwarded text message
                data = await state.get_data()
                forwarded_message = {
                    'text': message_text,
                    'from_user': message.forward_from.first_name if message.forward_from else "Unknown",
                    'date': message.forward_date or datetime.now(),
                    'is_forwarded': True,
                    'contains_url': bool(url),
                    'url': url if url else None
                }
                
                # Add to collected messages
                if 'forwarded_messages' not in data:
                    data['forwarded_messages'] = []
                data['forwarded_messages'].append(forwarded_message)
                await state.set_data(data)
                
                # Process after a short delay to collect multiple messages
                await asyncio.sleep(1)
                await analyze_forwarded_messages(message, state)
    elif message.document or message.photo:
        logging.info(f"Received document or photo from user {user_id}")
        await handle_document(message, state, message.bot)
        return
        
    # Handle URLs
    elif message.text:
        # Check if message contains URL
        url = await extract_url_from_text(message.text)
        if url:
            logging.info(f"Received URL from user {user_id}")
            
            # If message is just a URL (or URL with whitespace), handle it as a document
            if message.text.strip() == url or message.text.strip() == f"{url} ":
                logging.info(f"Processing plain URL: {url}")
                # Create content info for URL (same as in handle_document)
                content_info = {
                    'type': 'url',
                    'url': url,
                    'message_text': message.text.strip(),
                    'is_plain_url': True  # Flag to indicate this is a plain URL
                }
                # Collect for batched processing (1-second timer)
                from content_handlers import collect_content
                await collect_content(message, state, content_info)
                return
            else:
                # Process URL as part of a text message
                forwarded_message = {
                    'text': message.text,
                    'from_user': message.from_user.first_name,
                    'date': datetime.now(),
                    'is_forwarded': True,
                    'contains_url': True,
                    'url': url
                }
                
                # Add to collected messages
                data = await state.get_data()
                if 'forwarded_messages' not in data:
                    data['forwarded_messages'] = []
                data['forwarded_messages'].append(forwarded_message)
                await state.set_data(data)
                
                # Process after a short delay to collect multiple messages
                await asyncio.sleep(1)
                await analyze_forwarded_messages(message, state)
                return
            
        # Get state data
        data = await state.get_data()
        
        # Try to process as teamlead command first
        if await process_teamlead_command(message, state):
            logging.info(f"Processed as teamlead command: {message.text}")
            return
        
        # Check if in any special states
        current_state = await state.get_state()
        if current_state == BotStates.waiting_for_rules.state:
            # Save user rules and finish processing
            await redis_storage.save_user_rules(user_id, message.text)
            await message.answer("Правила сохранены! Теперь я буду использовать этот контекст при генерации ответов.")
            await state.finish()
            return
            
        elif current_state == BotStates.collecting_forwards.state:
            # Inform user we're still analyzing previous messages
            await message.answer("Я все еще анализирую предыдущие сообщения. Подождите немного...")
            return
            
        elif current_state == BotStates.waiting_for_clarification.state:
            # Handle clarification
            await handle_clarification(message, state)
            return
        
        # Load necessary functions
        from claude_utils import is_content_analysis_mode, check_if_new_conversation, is_file_generation_request, close_session, set_content_analysis_mode
        content_analysis_mode = await is_content_analysis_mode(user_id)
        
        # Check if message is a file creation request
        is_file_request, file_type = await is_file_generation_request(message.text)
        
        # Special handling for file creation requests
        if is_file_request and file_type:
            logging.info(f"Detected file creation request: {file_type} from message: {message.text}")
            # For non-standard requests (detected via AI analysis), reformulate the request
            from file_generator import reformulate_file_request
            
            # Check if the request was detected via direct pattern matching
            direct_match = False
            text_lower = message.text.lower()
            
            # Check all patterns for this file type
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
                    'добавь в календарь', 'создай событие', 'создай календарное событие', 
                    'добавь встречу', 'добавь событие', 'создай встречу', 'сохрани в календарь',
                    'сделай календарное событие', 'создай ics', 'создай ics файл', 
                    'закинь в календарь', 'в календарь', 'календарь', 'событие в календарь'
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
            
            if file_type in file_patterns:
                for pattern in file_patterns[file_type]:
                    if pattern in text_lower:
                        direct_match = True
                        break
            
            # If the request doesn't match standard patterns, reformulate it
            if not direct_match:
                reformulated_text = await reformulate_file_request(message.text, file_type, user_id)
                logging.info(f"Reformulated request: '{message.text}' -> '{reformulated_text}'")
                
                # Create a file with the reformulated request
                # Instead of checking for an existing session, create one if needed
                from claude_utils import ensure_session
                await ensure_session(user_id)
                await handle_file_generation(message, file_type, user_id, reformulated_text)
                return
            else:
                # Process standard file creation request
                # Instead of checking for an existing session, create one if needed
                from claude_utils import ensure_session
                await ensure_session(user_id)
                await handle_file_generation(message, file_type, user_id)
                return
            
        # If message is short (less than 5 words), treat it as a continuation
        if len(message.text.split()) < 5 or data.get('is_continuation', False):
            logging.info(f"Short message, treating as continuation: {message.text}")
                
            # Send message to Claude and get response
            await process_claude_message(message, state, user_id, user_rules)
            return
        
        # For longer messages, check if it's a new conversation
        is_new_conversation = await check_if_new_conversation(user_id, message.text, user_rules)
        
        # Determine if we need to add a summary based on message length
        from config import MAX_SUMMARY_THRESHOLD
        summary_needed = len(message.text) > MAX_SUMMARY_THRESHOLD
        if summary_needed:
            logging.info(f"Long message detected ({len(message.text)} chars), summary will be added.")
            # Set this in state for later processing
            await state.update_data(summary_needed=summary_needed)
        
        if is_new_conversation:
            logging.info(f"Detected new conversation: {message.text[:50]}...")
            # Close current session and create new one
            await close_session(user_id)
            
            # Reset content analysis mode
            await set_content_analysis_mode(user_id, False)
        
        # For file creation requests, use special handling
        if is_file_request and file_type and user_id in active_sessions:
            await handle_file_generation(message, file_type, user_id)
            return
        
        # For new conversation or long message, show processing feedback
        if is_new_conversation or summary_needed:
            # Send acknowledgment that we're working on it
            working_msg = await message.answer("Обрабатываю сообщение...")  # Processing message...
            await state.update_data(working_message_id=working_msg.message_id)
        
        # Send typing indicator
        await message.bot.send_chat_action(chat_id=message.chat.id, action="typing")
        
        # Send message to Claude and get response
        await process_claude_message(message, state, user_id, user_rules)
    else:
        # Unsupported message type
        logging.info("Unsupported message type")
        await message.answer(
            "Я могу работать с текстовыми сообщениями, документами и ссылками. Перешлите мне сообщения из любого чата или напишите /help для примеров.",
            parse_mode=ParseMode.MARKDOWN
        )

async def handle_analyze_callback(callback_query: types.CallbackQuery, state: FSMContext):
    """Handle analyze callback button click"""
    # Answer the callback to remove the loading indicator
    await callback_query.answer()
    
    # Get the message to use for replies
    message = callback_query.message
    
    # Process the collected forwarded messages
    await analyze_forwarded_messages(message, state, from_callback=True)
