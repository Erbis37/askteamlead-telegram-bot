"""Common message processing functions to avoid circular imports between
content_handlers.py and message_handlers.py

This module contains shared functionality for processing AI responses and 
sending them to users with appropriate formatting.
"""

import logging
import json
import re
import asyncio
from aiogram.types import Message, InlineKeyboardMarkup, InlineKeyboardButton, CallbackQuery
from aiogram.fsm.context import FSMContext
from aiogram.enums import ParseMode


def remove_markdown_formatting(text):
    """Remove markdown formatting but preserve ### separators for response variants"""
    if not text:
        return ""
        
    # Save existing ### separators (they're used for variant splitting)
    # Replace them temporarily with a unique marker that won't be in the text
    unique_marker = "__TRIPLE_HASH_SEPARATOR__"
    text = text.replace("###", unique_marker)
    
    # Remove markdown headers (# Header, ## Subheader, etc.) but keep the text
    # The regex matches 1-6 # at the beginning of a line followed by space, capturing the rest of the line
    text = re.sub(r'^#{1,6}\s+(.+)$', r'\1', text, flags=re.MULTILINE)
    
    # Remove bold formatting (**text**)
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    
    # Remove italic formatting (*text*)
    text = re.sub(r'\*([^\*]+)\*', r'\1', text)
    
    # Remove underscore italic (_text_)
    text = re.sub(r'_([^_]+)_', r'\1', text)
    
    # Remove backtick formatting (`text`)
    text = re.sub(r'`([^`]+)`', r'\1', text)
    
    # Remove link formatting [text](url)
    text = re.sub(r'\[(.+?)\]\(.+?\)', r'\1', text)
    
    # Remove horizontal rules (---, ***, ___)
    text = re.sub(r'^(---+|\*\*\*+|___+)$', '', text, flags=re.MULTILINE)
    
    # Restore the ### separators
    text = text.replace(unique_marker, "###")
    
    return text


def process_response_parts(response, content_analysis_mode=False):
    """Process response by splitting into parts and cleaning up formatting"""
    # If this is content analysis mode, don't split the response into parts
    # This is important for documents and URLs without accompanying text
    if content_analysis_mode or "###" not in response:
        clean_response = remove_markdown_formatting(response)
        
        # Remove prefixes like "Резюме содержимого документа" if present
        clean_response = re.sub(r'^\s*Резюме\s+содержимого\s+документа[:\s]*', '', clean_response, flags=re.IGNORECASE)
        
        return {
            "summary": clean_response,
            "options": []
        }
        
    # In variant mode, split by delimiter
    parts = response.split("###")
    
    # First element is summary/introduction, rest are variants
    summary = parts[0].strip()
    
    # Clean summary - remove "Summary:" prefix if exists
    if summary.startswith("Резюме:"):
        summary = summary[len("Резюме:"):].strip()
    
    # Clean summary formatting
    clean_summary = remove_markdown_formatting(summary)
    
    # Clean options - remove numbering and prefixes
    options = []
    raw_options = [opt.strip() for opt in parts[1:] if opt.strip()]
    
    for opt in raw_options:
        # Remove "Response options:" if it's the first option
        if opt.startswith("Варианты ответа:"):
            opt = opt[len("Варианты ответа:"):].strip()
            
        # Remove numbering (1., 2., 3., etc.)
        opt = re.sub(r'^\d+\.\s*', '', opt).strip()
        
        # Clean option formatting
        options.append(remove_markdown_formatting(opt))
    
    return {
        "summary": clean_summary,
        "options": options
    }


async def process_and_send_ai_response(message: Message, response: str, summary_needed: bool = False, state=None, content_analysis_mode: bool = False):
    """
    Process and send AI response to the user.
    Splits response variants by ### marker and sends them as text messages.
    
    Args:
        message: Message object
        response: Response from AI
        summary_needed: Whether to send summary separately
        state: FSMContext object (optional)
        content_analysis_mode: Whether the response is in content analysis mode for documents/URLs
    """
    try:
        if not response:
            await message.answer("Извините, не удалось сформировать ответ. Пожалуйста, попробуйте еще раз.")
            return
        
        # Get modes from state if available
        variant_mode = True
        needs_summary = summary_needed
        mixed_content_mode = False
        
        if state:
            data = await state.get_data()
            variant_mode = data.get('variant_mode', True)
            needs_summary = data.get('needs_summary', summary_needed)
            mixed_content_mode = data.get('mixed_content_mode', False)
            
            # Log modes for debugging
            logging.info(f"Response processing - variant_mode: {variant_mode}, needs_summary: {needs_summary}, "
                        f"content_analysis_mode: {content_analysis_mode}, mixed_content_mode: {mixed_content_mode}")
        
        # Clean up response
        response = response.strip()
        
        # Process the response into parts
        processed = process_response_parts(response, content_analysis_mode)
        
        # Check if we have any content to send
        if not processed["summary"].strip() and not processed["options"]:
            logging.warning("No valid response content found - sending fallback message")
            fallback_msg = "Извините, не удалось получить ответ по вашему запросу. "
            
            if content_analysis_mode:
                fallback_msg += "Попробуйте задать более конкретный вопрос о содержимом документа."
            else:
                fallback_msg += "Пожалуйста, попробуйте переформулировать вопрос или задать его иначе."
            
            await message.answer(fallback_msg)
            return
        
        # If we have a summary and it's needed, send it first
        if needs_summary and processed["summary"].strip():
            await message.answer(processed["summary"])
        
        # Handle variant responses if in variant mode and we have options
        if variant_mode and processed["options"]:
            for option in processed["options"]:
                if option.strip():
                    await message.answer(option.strip())
        # If not in variant mode or no options, just send the summary if we haven't already
        elif not needs_summary and processed["summary"].strip():
            await message.answer(processed["summary"])
        
        # Save the response in conversation history only if state is provided
        if state:
            # Get current state data
            current_data = await state.get_data()
            current_mode = current_data.get('content_analysis_mode', False)
            current_mixed = current_data.get('mixed_content_mode', False)
            
            # Only update the mode if it's not already set or if we're explicitly changing it
            if current_mode != content_analysis_mode or 'content_analysis_mode' not in current_data:
                if content_analysis_mode:
                    if current_mixed or ('mixed_content_mode' in current_data and current_data['mixed_content_mode']):
                        # In mixed content mode, we want to show variants after the summary
                        await state.update_data(
                            variant_mode=True, 
                            content_analysis_mode=True, 
                            mixed_content_mode=True
                        )
                        logging.info("Set mixed content mode: variant_mode=True, content_analysis_mode=True")
                    else:
                        # In pure content analysis mode, don't show variants
                        await state.update_data(
                            variant_mode=False, 
                            content_analysis_mode=True,
                            mixed_content_mode=False
                        )
                        logging.info("Set content analysis mode: variant_mode=False")
                else:
                    # In standard mode, show variants
                    await state.update_data(
                        variant_mode=True, 
                        content_analysis_mode=False,
                        mixed_content_mode=False
                    )
                    logging.info("Set standard mode: variant_mode=True")

    
    except Exception as e:
        logging.error(f"Error in process_and_send_ai_response: {e}")
        await message.answer(f"Произошла ошибка при обработке ответа: {str(e)}")
