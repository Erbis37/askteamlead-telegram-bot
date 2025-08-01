import os
import logging
import tempfile
import asyncio
from typing import Optional
from aiogram import Bot, types
from pydub import AudioSegment
import speech_recognition as sr
from config import MAX_VOICE_DURATION, GOOGLE_SPEECH_ENABLED

# Initialize recognizer
recognizer = sr.Recognizer()

async def process_voice_message(message: types.Message, bot: Bot) -> Optional[str]:
    """
    Process voice message: download, convert to WAV, trim to 1 minute, and transcribe
    
    Args:
        message: Telegram message containing voice
        bot: Bot instance
        
    Returns:
        Transcribed text or None if failed
    """
    if not GOOGLE_SPEECH_ENABLED:
        logging.warning("Google Speech recognition is disabled")
        return None
        
    try:
        # Get voice file info
        voice = message.voice
        if not voice:
            logging.error("No voice object in message")
            return None
            
        duration = voice.duration
        logging.info(f"Voice message duration: {duration} seconds")
        
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Download voice file
            voice_path = os.path.join(temp_dir, f"voice_{message.message_id}.ogg")
            file = await bot.get_file(voice.file_id)
            await bot.download_file(file.file_path, voice_path)
            
            # Convert OGG to WAV for better compatibility
            wav_path = os.path.join(temp_dir, f"voice_{message.message_id}.wav")
            audio = AudioSegment.from_ogg(voice_path)
            
            # Trim to first minute if longer
            if duration > MAX_VOICE_DURATION:
                logging.info(f"Trimming voice message from {duration}s to {MAX_VOICE_DURATION}s")
                audio = audio[:MAX_VOICE_DURATION * 1000]  # pydub works in milliseconds
                await message.answer(
                    f"Голосовое сообщение обрезано до {MAX_VOICE_DURATION} секунд для бесплатного распознавания."
                )
            
            # Export as WAV
            audio.export(wav_path, format="wav")
            
            # Transcribe using Google Speech Recognition
            with sr.AudioFile(wav_path) as source:
                # Adjust for ambient noise
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                # Record audio data
                audio_data = recognizer.record(source)
                
                try:
                    # Try to recognize in Russian first
                    text = recognizer.recognize_google(audio_data, language="ru-RU")
                    logging.info(f"Successfully transcribed voice message: {text[:50]}...")
                    return text
                except sr.UnknownValueError:
                    # Try English if Russian fails
                    try:
                        text = recognizer.recognize_google(audio_data, language="en-US")
                        logging.info(f"Successfully transcribed voice message (EN): {text[:50]}...")
                        return text
                    except sr.UnknownValueError:
                        logging.warning("Google Speech Recognition could not understand audio")
                        await message.answer(
                            "Не удалось распознать голосовое сообщение. "
                            "Попробуйте говорить четче или отправьте текстовое сообщение."
                        )
                        return None
                except sr.RequestError as e:
                    logging.error(f"Google Speech Recognition error: {e}")
                    await message.answer(
                        "Ошибка сервиса распознавания речи. "
                        "Пожалуйста, отправьте текстовое сообщение."
                    )
                    return None
                    
    except Exception as e:
        logging.error(f"Error processing voice message: {e}", exc_info=True)
        await message.answer(
            "Произошла ошибка при обработке голосового сообщения. "
            "Пожалуйста, отправьте текстовое сообщение."
        )
        return None

async def handle_voice_message(message: types.Message, bot: Bot) -> Optional[str]:
    """
    Handle voice message with user feedback
    
    Args:
        message: Telegram message containing voice
        bot: Bot instance
        
    Returns:
        Transcribed text or None if failed
    """
    # Send processing notification
    processing_msg = await message.answer("Обрабатываю голосовое сообщение...")
    
    try:
        # Process voice
        text = await process_voice_message(message, bot)
        
        # Delete processing message
        await processing_msg.delete()
        
        if text:
            # Send transcription confirmation
            await message.answer(f"Распознанный текст:\n_{text}_", parse_mode="Markdown")
            return text
        else:
            return None
            
    except Exception as e:
        logging.error(f"Error in handle_voice_message: {e}")
        try:
            await processing_msg.delete()
        except:
            pass
        return None