from aiogram import Bot, types
from aiogram.types import BotCommand, BotCommandScopeDefault
from aiogram.enums import ParseMode
from aiogram.fsm.context import FSMContext
from redis_storage import RedisStorage
import logging
from states import BotStates

async def set_commands(bot: Bot):
    """Setup bot commands that are shown in the menu"""
    commands = [
        BotCommand(command="start", description="Познакомиться с вашим AI тимлидом"),
        BotCommand(command="help", description="Как этим пользоваться"),
        BotCommand(command="rules", description="Настроить характер вашего тимлида"),
        BotCommand(command="new", description="Начать новый чат с чистого листа"),
        BotCommand(command="clean", description="Вернуть стандартный характер тимлида"),
        BotCommand(command="changelog", description="Мои суперспособности"),
    ]
    await bot.set_my_commands(commands, scope=BotCommandScopeDefault())

async def handle_start(message: types.Message, bot: Bot):
    """Handler for /start command"""
    # Send welcome message
    await message.answer(
        f"Привет, {message.from_user.first_name}!\n\n"
        "Я ваш AI тимлид - помогаю командам работать эффективнее и веселее!\n\n"
        "Что я умею:\n\n"
        "• Превращу ваши идеи в четкие задачи с оценками\n"
        "• Организую работу в Trello - чтобы ничего не потерялось\n"
        "• Построю красивые графики и диаграммы Ганта\n"
        "• Подготовлю отчеты и презентации для команды\n"
        "• Посчитаю бюджет и сроки - все будет прозрачно\n"
        "• Напомню о важных дедлайнах\n"
        "• Понимаю голосовые сообщения\n\n"
        "Начнем? Просто расскажите, что нужно сделать!\n\n"
        "Хотите настроить мой характер? Используйте /rules\n"
        "Нужна инструкция? Жмите /help"
    )
    
    # Send security message in a separate message
    await message.answer(
        "Кстати о безопасности: все ваши данные остаются только в вашем Telegram. "
        "Я ничего не сохраняю!"
    )
    
    # Set up bot menu commands
    await set_commands(bot)

async def handle_help(message: types.Message):
    """Handler for /help command - provides detailed help information"""
    help_text = (
        "**Как со мной работать**\n\n"
        
        "**Управляем задачами:**\n"
        "Просто расскажите, что нужно сделать - я сама разберу на части и оценю сроки.\n"
        "Могу сразу закинуть в Trello, показать что в работе или передвинуть по колонкам.\n\n"
        
        "**Планируем работу:**\n"
        "Скажите «нужен план спринта» или «сделай Гант» - будет вам красивая картинка.\n"
        "Хотите цифры? Попросите CSV с оценками или расчет бюджета.\n\n"
        
        "**Готовим материалы:**\n"
        "Нужна презентация для команды? Архитектурная схема? Легко!\n"
        "Кидайте документы - проанализирую и подскажу, что можно улучшить.\n\n"
        
        "**Говорите голосом:**\n"
        "Лень печатать? Отправьте голосовое до минуты - я все пойму.\n\n"
        
        "**Мои команды:**\n"
        "/rules - научите меня общаться как вам удобно\n"
        "/new - начнем новый чат с чистого листа\n"
        "/clean - вернусь к стандартному поведению\n"
        "/changelog - покажу все свои фишки\n\n"
        
        "**Лайфхак:** Я помню наш разговор, так что можете ссылаться на то, что обсуждали раньше!"
    )
    
    await message.answer(help_text, parse_mode=ParseMode.MARKDOWN)

async def handle_new(message: types.Message, state: FSMContext):
    """Handler for /new command - starts a new chat by clearing conversation history, but preserves user rules"""
    # Get user ID
    user_id = message.from_user.id
    
    try:
        # Save current user rules if they exist
        data = await state.get_data()
        user_rules = data.get('rules')
        
        # If no rules in state, try to get from Redis
        if not user_rules:
            redis_storage = RedisStorage()
            user_rules = await redis_storage.get_user_rules(user_id)
        
        # Fully clear state (including conversation history)
        await state.clear()
        
        # Restore user rules to the state if they existed
        if user_rules:
            await state.update_data(rules=user_rules)
        
        # Reset to default mode
        await state.update_data(variant_mode=True, content_analysis_mode=False)
        
        # Close Claude session if available
        try:
            from claude_utils import close_session
            await close_session(user_id)
            logging.info(f"Closed Claude session for user {user_id} via /new command")
        except Exception as e:
            logging.error(f"Error closing Claude session: {e}")
        
        # Inform user
        await message.answer(
            "Отлично! Начинаем с чистого листа. История нашего разговора очищена.\n\n"
            "Кстати, ваши настройки характера сохранились. Если хотите их тоже сбросить - используйте /clean.",
            parse_mode=ParseMode.MARKDOWN
        )
    except Exception as e:
        logging.error(f"Error starting new chat for user {user_id}: {e}")
        await message.answer(
            "Упс! Что-то пошло не так. Попробуйте еще раз через пару секунд.", 
            parse_mode=ParseMode.MARKDOWN
        )

async def handle_clear(message: types.Message, state: FSMContext):
    """Handler for /clean command - clears only user rules"""
    # Get user ID
    user_id = message.from_user.id
    
    # Get current state data to preserve conversation history
    data = await state.get_data()
    
    try:
        # Clear the user rules in Redis
        redis_storage = RedisStorage()
        success = await redis_storage.delete_user_rules(user_id)
        
        # Remove rules from state if they exist
        if 'rules' in data:
            del data['rules']
            await state.set_data(data)
        
        if success:
            await message.answer("Готово! Теперь я веду себя как обычно.")
        else:
            await message.answer("Что-то не так. Попробуйте еще разок.")
    except Exception as e:
        logging.error(f"Error clearing user rules: {e}")
        await message.answer("Что-то не так. Попробуйте еще разок.")

async def handle_changelog(message: types.Message):
    """Handler for /changelog command - shows recent updates"""
    changelog_text = (
        "**Мои суперспособности**\n\n"
        
        "**Планирование как искусство:**\n"
        "Разложу любую задачу на понятные кусочки и нарисую красивую диаграмму Ганта.\n"
        "Знаю сколько времени и денег нужно на каждый этап.\n\n"
        
        "**Дружу с Trello:**\n"
        "Создаю задачи, двигаю по колонкам, слежу за дедлайнами.\n"
        "Могу показать кто чем занят и что горит красным.\n\n"
        
        "**Генерирую документы:**\n"
        "CSV-отчеты с детальными оценками? Легко!\n"
        "Презентация для инвесторов? Без проблем!\n"
        "Календарное событие? Уже в вашем календаре!\n\n"
        
        "**Понимаю с полуслова:**\n"
        "Говорите голосом - я распознаю.\n"
        "Кидайте документы - проанализирую.\n"
        "Ссылайтесь на прошлые обсуждения - я все помню!\n\n"
        
        "**P.S.** Постоянно учусь новому, так что следите за обновлениями!"
    )
    
    await message.answer(changelog_text, parse_mode=ParseMode.MARKDOWN)


async def handle_rules_command(message: types.Message, state: FSMContext):
    """Handler for /rules command to set conversation rules"""
    # Set state to waiting for rules
    await state.set_state(BotStates.waiting_for_rules)
    
    # Retrieve current rules if they exist
    user_id = message.from_user.id
    redis_storage = RedisStorage()
    current_rules = await redis_storage.get_user_rules(user_id)
    
    # Prepare response message
    rules_message = (
        "**Настройка характера**\n\n"
        "Расскажите, каким тимлидом мне быть для вашей команды?\n\n"
        "**Вот несколько идей:**\n"
        "• Строгий наставник - буду требовательным и держать всех в тонусе\n"
        "• Добрая подруга - поддержу и помогу, никакого давления\n"
        "• Технический гуру - глубоко копаю в детали, говорю на языке разработчиков\n"
        "• Милая девушка тимлид - мягко, но настойчиво веду к цели\n"
        "• Деловой партнер - только факты, цифры и конкретика\n"
        "• Вдохновляющий коуч - мотивирую и заряжаю энергией\n\n"
        "**Или придумайте свой вариант!** Опишите, как вам удобнее.\n\n"
    )
    
    # Add info about current rules if they exist
    if current_rules:
        rules_message += f"**Сейчас я такая:**\n_{current_rules}_\n\n"
        rules_message += "Хотите изменить? Просто опишите новый характер.\nПередумали? Отправьте /cancel"
    else:
        rules_message += "Пока я веду себя стандартно. Опишите желаемый стиль или отправьте /cancel"
    
    await message.answer(rules_message, parse_mode=ParseMode.MARKDOWN)

async def handle_rules_input(message: types.Message, state: FSMContext):
    """Handler for receiving rules input after /rules command"""
    # Check if the user wants to cancel
    if message.text.startswith('/'):
        command = message.text.lower()
        if command == '/cancel':
            await state.set_state(None)
            await message.answer("Ок, оставляем как есть!", parse_mode=ParseMode.MARKDOWN)
            return
        elif command.startswith('/'):
            await state.set_state(None)
            await message.answer(
                "Ой, вы начали другую команду. Настройка отменена!",
                parse_mode=ParseMode.MARKDOWN
            )
            return
    
    # Get the rules text
    rules_text = message.text.strip()
    
    # Validate rules length
    if len(rules_text) < 3:
        await message.answer(
            "Это слишком коротко! Расскажите подробнее, каким тимлидом мне быть.",
            parse_mode=ParseMode.MARKDOWN
        )
        return
    
    if len(rules_text) > 500:
        await message.answer(
            "Вау! Это целый роман! Давайте покороче - до 500 символов.",
            parse_mode=ParseMode.MARKDOWN
        )
        return
    
    # Store rules in Redis
    user_id = message.from_user.id
    redis_storage = RedisStorage()
    await redis_storage.save_user_rules(user_id, rules_text)
    
    # Also store it in the state for current session
    await state.update_data(rules=rules_text)
    
    # Clear the waiting_for_rules state
    await state.set_state(None)
    
    # Confirm to the user
    await message.answer(
        f"**Отлично! Теперь я буду:**\n\n_{rules_text}_\n\nНравится? Давайте работать!",
        parse_mode=ParseMode.MARKDOWN
    )
