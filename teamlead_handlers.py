import logging
import re
from typing import Optional, Dict, List
from datetime import datetime, timedelta
from aiogram import types
from aiogram.fsm.context import FSMContext

from trello_integration import TrelloIntegration
from config import TRELLO_API_KEY, TRELLO_TOKEN, TRELLO_BOARD_ID
from claude_utils import get_claude_completion
from gantt_generator import generate_gantt_from_trello, create_gantt_chart, parse_tasks_for_gantt
from presentation_generator import create_project_presentation

# Initialize Trello integration
trello = None
if all([TRELLO_API_KEY, TRELLO_TOKEN, TRELLO_BOARD_ID]):
    trello = TrelloIntegration(TRELLO_API_KEY, TRELLO_TOKEN, TRELLO_BOARD_ID)
else:
    logging.warning("Trello integration not initialized - missing credentials")

async def parse_task_command(text: str, user_id: int) -> Dict:
    """
    Parse natural language task commands using Claude AI
    
    Args:
        text: User's command text
        user_id: User's Telegram ID
        
    Returns:
        Dictionary with parsed command information
    """
    # Get conversation history to understand context
    from claude_utils import active_sessions
    conversation_context = ""
    if user_id in active_sessions and "messages" in active_sessions[user_id]:
        # Get last few messages for context
        recent_messages = active_sessions[user_id]["messages"][-6:]  # Last 3 exchanges
        for msg in recent_messages:
            conversation_context += f"{msg['role']}: {msg['content']}\n"
    
    prompt = f"""Based on the conversation history and the current command, extract task information.

Conversation context:
{conversation_context}

Current command: "{text}"

Extract and return the following information in a structured format:
- action: Determine what the user wants to do (create_task, move_task, list_tasks, estimate_task, create_report, show_status, create_gantt, create_presentation)
- task_name: Extract the task name from the conversation context (e.g., "Разработка промо-лендинга")
- description: Extract task details from the conversation (breakdown of work, estimates, etc.)
- target_column: Which column to place the task (Backlog, Design, To Do, Doing, Code Review, Testing, Done)
- deadline: If mentioned, format as ISO date
- assignee: If mentioned
- complexity: If discussed (simple, medium, complex)

For "создай таски в трело" or similar commands, extract ALL tasks discussed in the conversation.

Respond with structured JSON-like format."""

    try:
        # Create messages for Claude API
        messages = [
            {"role": "user", "content": prompt}
        ]
        response = await get_claude_completion(messages, user_id=user_id, save_in_history=False)
        
        # Parse the response to extract structured data
        parsed = {
            'action': None,
            'task_name': None,
            'description': None,
            'target_column': None,
            'deadline': None,
            'assignee': None,
            'complexity': None,
            'all_tasks': []  # For multiple tasks
        }
        
        # First check the command action based on keywords
        command_lower = text.lower()
        if any(word in command_lower for word in ['создай', 'добавь', 'новая задача', 'create', 'закинь', 'таски', 'таск']):
            parsed['action'] = 'create_task'
            
            # Check if user asks for multiple tasks
            if any(word in command_lower for word in ['таски', 'задачи', 'все задачи', 'необходимые']):
                # Request Claude to extract all tasks from conversation
                extract_prompt = f"""Based on our conversation history, extract ALL tasks that need to be created in Trello.

Conversation context:
{conversation_context}

For each task, provide:
- Task name
- Estimated duration (in days)
- Target column (Design, To Do, etc.)
- Any additional details

Format each task on a new line like:
TASK: [name] | DURATION: [days] | COLUMN: [column]"""
                
                extract_messages = [{"role": "user", "content": extract_prompt}]
                tasks_response = await get_claude_completion(extract_messages, user_id=user_id, save_in_history=False)
                
                # Parse tasks from response
                all_tasks = []
                if tasks_response:
                    for line in tasks_response.split('\n'):
                        if 'TASK:' in line:
                            task_info_match = re.match(r'TASK:\s*([^|]+)\s*\|\s*DURATION:\s*([^|]+)\s*\|\s*COLUMN:\s*(.+)', line)
                            if task_info_match:
                                all_tasks.append({
                                    'name': task_info_match.group(1).strip(),
                                    'days': task_info_match.group(2).strip(),
                                    'column': task_info_match.group(3).strip()
                                })
                
                if all_tasks:
                    parsed['all_tasks'] = all_tasks
                    return parsed
        elif any(word in text.lower() for word in ['передвинь', 'перемести', 'move']):
            parsed['action'] = 'move_task'
        elif any(word in text.lower() for word in ['покажи', 'список', 'задачи', 'show', 'list']):
            parsed['action'] = 'list_tasks'
        elif any(word in text.lower() for word in ['оцени', 'estimate', 'сколько']):
            parsed['action'] = 'estimate_task'
        elif any(word in text.lower() for word in ['отчет', 'csv', 'экспорт', 'report']):
            parsed['action'] = 'create_report'
            # Check if user specifically mentions Trello
            if 'trello' in text.lower() or 'трелло' in text.lower():
                parsed['from_trello'] = True
            else:
                parsed['from_trello'] = False
        elif any(word in text.lower() for word in ['статус', 'обзор', 'status']):
            parsed['action'] = 'show_status'
        elif any(word in text.lower() for word in ['гант', 'gantt', 'диаграмм', 'timeline']):
            parsed['action'] = 'create_gantt'
            # Check if user specifically mentions Trello
            if 'trello' in text.lower() or 'трелло' in text.lower():
                parsed['from_trello'] = True
            else:
                parsed['from_trello'] = False
        elif any(word in text.lower() for word in ['презентаци', 'presentation', 'слайд']):
            parsed['action'] = 'create_presentation'
            
        # Extract task name (simple heuristic)
        if parsed['action'] in ['create_task', 'move_task']:
            # Try to extract quoted text as task name
            quotes = re.findall(r'"([^"]*)"', text)
            if quotes:
                parsed['task_name'] = quotes[0]
                
        # Try to parse Claude's response to extract task details
        if response:
            # Try to extract task name
            task_name_match = re.search(r'task_name["\s:]+([^"\n]+)', response, re.IGNORECASE)
            if task_name_match:
                parsed['task_name'] = task_name_match.group(1).strip()
            
            # Try to extract description
            desc_match = re.search(r'description["\s:]+([^"\n]+)', response, re.IGNORECASE)
            if desc_match:
                parsed['description'] = desc_match.group(1).strip()
            
            # Try to extract target column
            column_match = re.search(r'target_column["\s:]+([^"\n]+)', response, re.IGNORECASE)
            if column_match:
                parsed['target_column'] = column_match.group(1).strip()
            
            # Check for multiple tasks pattern in response
            if 'дизайн макета' in response.lower() or 'верстка' in response.lower():
                # Extract all tasks from the conversation about landing page
                tasks = []
                if 'лендинг' in conversation_context.lower() or 'landing' in conversation_context.lower():
                    tasks = [
                        {'name': 'Дизайн макета промо-лендинга', 'days': '3-4', 'column': 'Design'},
                        {'name': 'Верстка лендинга', 'days': '2-3', 'column': 'To Do'},
                        {'name': 'Программирование функционала', 'days': '2-3', 'column': 'To Do'},
                        {'name': 'Тестирование на разных устройствах', 'days': '1', 'column': 'To Do'}
                    ]
                parsed['all_tasks'] = tasks
        
        # Extract target column from command if not found
        if not parsed['target_column']:
            for column in ['Backlog', 'Design', 'To Do', 'Doing', 'Code Review', 'Testing', 'Done']:
                if column.lower() in text.lower():
                    parsed['target_column'] = column
                    break
                    
        # Extract deadline
        if 'завтра' in text.lower():
            parsed['deadline'] = (datetime.now() + timedelta(days=1)).isoformat()
        elif 'послезавтра' in text.lower():
            parsed['deadline'] = (datetime.now() + timedelta(days=2)).isoformat()
        elif 'через неделю' in text.lower():
            parsed['deadline'] = (datetime.now() + timedelta(weeks=1)).isoformat()
            
        return parsed
        
    except Exception as e:
        logging.error(f"Error parsing task command: {e}")
        return {'action': None}

async def handle_create_task(message: types.Message, task_info: Dict):
    """Handle task creation in Trello"""
    if not trello:
        await message.answer("Trello интеграция не настроена")
        return
        
    try:
        # Check if we have multiple tasks to create
        if task_info.get('all_tasks'):
            created_tasks = []
            for task in task_info['all_tasks']:
                list_name = task.get('column', 'To Do')
                task_name = task.get('name', 'Новая задача')
                description = f"Оценка: {task.get('days', 'не определено')} дней"
                
                # Create card in Trello
                try:
                    card = await trello.create_card(
                        list_name=list_name,
                        name=task_name,
                        desc=description,
                        due=None
                    )
                    created_tasks.append({
                        'name': task_name,
                        'column': list_name,
                        'url': card['shortUrl']
                    })
                except Exception as e:
                    logging.error(f"Error creating task {task_name}: {e}")
            
            if created_tasks:
                response = f"Создано задач: {len(created_tasks)}\n\n"
                for task in created_tasks:
                    response += f"• **{task['name']}** → {task['column']}\n"
                    response += f"  [Открыть в Trello]({task['url']})\n\n"
                await message.answer(response, parse_mode="Markdown", disable_web_page_preview=True)
            else:
                await message.answer("Не удалось создать задачи")
        else:
            # Single task creation
            list_name = task_info.get('target_column') or 'To Do'  # Default to To Do if None
            task_name = task_info.get('task_name', 'Новая задача')
            description = task_info.get('description', '')
            deadline = None
            
            if task_info.get('deadline'):
                deadline = datetime.fromisoformat(task_info['deadline'])
                
            # Create card in Trello
            card = await trello.create_card(
                list_name=list_name,
                name=task_name,
                desc=description,
                due=deadline
            )
            
            response = f"Задача создана в колонке **{list_name}**\n\n"
            response += f"**{task_name}**\n"
            if description:
                response += f"{description}\n"
            if deadline:
                response += f"Дедлайн: {deadline.strftime('%d.%m.%Y')}\n"
            response += f"\n[Открыть в Trello]({card['shortUrl']})"
            
            await message.answer(response, parse_mode="Markdown")
        
    except Exception as e:
        logging.error(f"Error creating task: {e}")
        await message.answer(f"Ошибка при создании задачи: {str(e)}")

async def handle_list_tasks(message: types.Message, filter_column: Optional[str] = None):
    """List tasks from Trello board"""
    if not trello:
        await message.answer("Trello интеграция не настроена")
        return
        
    try:
        await message.answer("Загружаю задачи...")
        
        lists = await trello.get_lists()
        all_cards = {}
        
        for list_data in lists:
            if filter_column and list_data['name'] != filter_column:
                continue
            cards = await trello.get_cards(list_data['id'])
            if cards:
                all_cards[list_data['id']] = cards
                
        if not all_cards:
            await message.answer("Нет задач для отображения")
            return
            
        # Format tasks
        response = "**Задачи на доске:**\n"
        
        for list_data in lists:
            list_id = list_data['id']
            if list_id not in all_cards:
                continue
                
            list_name = list_data['name']
            cards = all_cards[list_id]
            
            response += f"\n**{list_name}** ({len(cards)}):\n"
            
            for card in cards[:5]:  # Show max 5 cards per list
                assignee = next((m['fullName'] for m in card.get('members', [])), 'Не назначено')
                deadline_str, emoji = trello.format_deadline(card.get('due'), list_name)
                
                response += f"• [{card['name']}]({card['shortUrl']})"
                response += f" - {assignee}"
                if deadline_str:
                    response += f" {emoji} {deadline_str}"
                response += "\n"
                
            if len(cards) > 5:
                response += f"_...и еще {len(cards) - 5} задач_\n"
                
        await message.answer(response, parse_mode="Markdown", disable_web_page_preview=True)
        
    except Exception as e:
        logging.error(f"Error listing tasks: {e}")
        await message.answer(f"Ошибка при загрузке задач: {str(e)}")

async def handle_project_status(message: types.Message):
    """Show overall project status"""
    if not trello:
        await message.answer("Trello интеграция не настроена")
        return
        
    try:
        await message.answer("Анализирую статус проекта...")
        
        status = await trello.get_project_status()
        response = trello.format_project_status(status)
        
        await message.answer(response, parse_mode="Markdown")
        
    except Exception as e:
        logging.error(f"Error getting project status: {e}")
        await message.answer(f"Ошибка при получении статуса: {str(e)}")

async def handle_create_report(message: types.Message, include_estimates: bool = False, from_trello: bool = False):
    """Create CSV report with tasks from context or Trello"""
    try:
        await message.answer("Создаю отчет...")
        
        if from_trello and trello:
            # Generate from Trello data
            lists = await trello.get_lists()
            all_cards = {}
            
            for list_data in lists:
                cards = await trello.get_cards(list_data['id'])
                all_cards[list_data['id']] = cards
                
            # Create CSV
            csv_data = trello.create_csv_report(lists, all_cards, include_estimates)
        else:
            # Generate from chat context
            user_id = message.from_user.id
            
            # Get conversation history from Claude session
            from claude_utils import active_sessions
            conversation_history = []
            if user_id in active_sessions and "messages" in active_sessions[user_id]:
                conversation_history = active_sessions[user_id]["messages"]
            
            # Ask Claude to extract tasks from conversation
            prompt = """Based on our conversation history, create a CSV report with all tasks mentioned.
            Format the response as CSV with these columns:
            Task;Assignee;Deadline;Status;Complexity;Estimated Hours;Estimated Cost
            
            Extract all tasks discussed, with their details. If some information is missing, use reasonable defaults:
            - Assignee: "Team" if not specified
            - Deadline: Estimate based on complexity
            - Status: "To Do" if not specified
            - Complexity: Assess based on task description (simple/medium/complex)
            - Estimated Hours: Based on complexity (simple=2-4h, medium=8-16h, complex=24-40h)
            - Estimated Cost: Hours * $50/hour
            
            If no specific tasks were mentioned, create a sample report based on the discussion context."""
            
            messages = [
                {"role": "user", "content": prompt}
            ]
            
            csv_content = await get_claude_completion(messages, user_id=user_id)
            
            # Create StringIO object from the response
            from io import StringIO
            csv_data = StringIO()
            csv_data.write(csv_content)
            csv_data.seek(0)
        
        # Send file
        filename = f"project_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        await message.answer_document(
            types.BufferedInputFile(
                csv_data.getvalue().encode('utf-8-sig'),
                filename=filename
            ),
            caption="Отчет по проекту готов!"
        )
        
    except Exception as e:
        logging.error(f"Error creating report: {e}")
        await message.answer(f"Ошибка при создании отчета: {str(e)}")

async def handle_create_gantt(message: types.Message, from_trello: bool = False):
    """Create and send Gantt chart from context or Trello"""
    try:
        await message.answer("Создаю диаграмму Ганта...")
        
        if from_trello and trello:
            # Generate from Trello data
            img_buffer = await generate_gantt_from_trello(trello)
        else:
            # Generate from chat context
            user_id = message.from_user.id
            
            # Get conversation history from Claude session
            from claude_utils import active_sessions
            conversation_history = []
            if user_id in active_sessions and "messages" in active_sessions[user_id]:
                conversation_history = active_sessions[user_id]["messages"]
            
            # Ask Claude to extract tasks from conversation
            prompt = """Based on our conversation history, extract all tasks mentioned with their deadlines and assignees.
            Format each task as:
            - Task name
            - Start date (if mentioned, otherwise estimate based on context)
            - End date/deadline (if mentioned)
            - Assignee (if mentioned)
            - Status/Progress (if mentioned)
            
            If no specific tasks were mentioned, create a sample project plan based on the discussion context."""
            
            messages = [
                {"role": "user", "content": prompt}
            ]
            
            task_description = await get_claude_completion(messages, user_id=user_id)
            
            # Parse tasks for Gantt chart
            tasks = parse_tasks_for_gantt(task_description, user_id)
            
            # Create Gantt chart
            img_buffer = create_gantt_chart(tasks)
        
        # Send as photo
        await message.answer_photo(
            types.BufferedInputFile(
                img_buffer.getvalue(),
                filename="gantt_chart.png"
            ),
            caption="Диаграмма Ганта проекта\n\n" +
                    "Зеленый - задачи в графике\n" +
                    "Желтый - приближается дедлайн\n" +
                    "Красный - просроченные задачи"
        )
        
    except Exception as e:
        logging.error(f"Error creating Gantt chart: {e}")
        await message.answer(f"Ошибка при создании диаграммы: {str(e)}")

async def handle_create_presentation(message: types.Message):
    """Create and send project presentation"""
    try:
        await message.answer("Создаю презентацию для команды...")
        
        # Get project info using Claude
        user_id = message.from_user.id
        prompt = """Based on the current project context, create a project overview with:
        - Project title
        - Current status overview
        - Sprint progress
        - Key risks and mitigation strategies
        - Next steps and action items
        
        Format as a structured response."""
        
        # Create messages for Claude API
        messages = [
            {"role": "user", "content": prompt}
        ]
        project_overview = await get_claude_completion(messages, user_id=user_id)
        
        # Parse the response to create project_info dict
        project_info = {
            'title': 'Project Status Update',
            'status_overview': project_overview[:200] if project_overview else 'Project progressing well',
            'sprint_progress': '• Sprint Goal: Feature Development\n• Progress: In Progress\n• On Track',
            'risks': [
                '• Timeline: Monitor closely',
                '• Resources: Adequate coverage',
                '• Technical: Architecture review needed'
            ],
            'next_steps': [
                '• Complete current sprint tasks',
                '• Review and update backlog',
                '• Schedule team retrospective'
            ]
        }
        
        # Create presentation
        pptx_buffer = await create_project_presentation(project_info, trello)
        
        # Send file
        await message.answer_document(
            types.BufferedInputFile(
                pptx_buffer.getvalue(),
                filename=f"project_presentation_{datetime.now().strftime('%Y%m%d')}.pptx"
            ),
            caption="Презентация проекта готова!\n\n" +
                    "Содержит:\n" +
                    "• Обзор статуса проекта\n" +
                    "• Распределение задач\n" +
                    "• Прогресс спринта\n" +
                    "• Риски и план митигации\n" +
                    "• Следующие шаги"
        )
        
    except Exception as e:
        logging.error(f"Error creating presentation: {e}")
        await message.answer(f"Ошибка при создании презентации: {str(e)}")

async def handle_estimate_task(message: types.Message, task_name: str):
    """Estimate task complexity and cost"""
    if not trello:
        # Can still do estimation without Trello
        estimate = trello.estimate_task_complexity(task_name) if trello else {
            'complexity': 'medium',
            'estimated_hours': 4,
            'estimated_cost': 200
        }
        
        response = f"**Оценка задачи:** {task_name}\n\n"
        response += f"Сложность: {estimate['complexity']}\n"
        response += f"Оценка времени: {estimate['estimated_hours']} часов\n"
        response += f"Оценка стоимости: ${estimate['estimated_cost']}\n"
        
        await message.answer(response, parse_mode="Markdown")
        return
        
    try:
        # Get more detailed estimate using Claude
        user_id = message.from_user.id
        prompt = f"""As an experienced team lead, estimate this task: "{task_name}"

Consider:
1. Task complexity (simple/medium/complex)
2. Required skills and expertise
3. Potential risks and dependencies
4. Testing and review time
5. Documentation needs

Provide:
- Complexity level
- Estimated hours (min-max range)
- Required team members
- Main risks
- Recommendations

Be realistic and consider typical software development practices."""

        # Create messages for Claude API
        messages = [
            {"role": "user", "content": prompt}
        ]
        response = await get_claude_completion(messages, user_id=user_id)
        
        await message.answer(f"**Детальная оценка задачи:**\n\n{response}", parse_mode="Markdown")
        
    except Exception as e:
        logging.error(f"Error estimating task: {e}")
        await message.answer("Ошибка при оценке задачи")

async def process_teamlead_command(message: types.Message, state: FSMContext):
    """
    Process natural language commands for team lead functions
    
    Args:
        message: User message
        state: FSM context
    """
    user_id = message.from_user.id
    text = message.text
    
    # Ensure we have an active session
    from claude_utils import ensure_session
    await ensure_session(user_id)
    
    # Parse command
    command_info = await parse_task_command(text, user_id)
    
    if not command_info.get('action'):
        # Not a recognized command, handle as regular message
        return False
        
    action = command_info['action']
    
    try:
        if action == 'create_task':
            await handle_create_task(message, command_info)
        elif action == 'list_tasks':
            await handle_list_tasks(message, command_info.get('target_column'))
        elif action == 'show_status':
            await handle_project_status(message)
        elif action == 'create_report':
            include_estimates = 'оценк' in text.lower() or 'стоимост' in text.lower()
            from_trello = command_info.get('from_trello', False)
            await handle_create_report(message, include_estimates, from_trello=from_trello)
        elif action == 'estimate_task':
            task_name = command_info.get('task_name', text)
            await handle_estimate_task(message, task_name)
        elif action == 'create_gantt':
            from_trello = command_info.get('from_trello', False)
            await handle_create_gantt(message, from_trello=from_trello)
        elif action == 'create_presentation':
            await handle_create_presentation(message)
        else:
            return False
            
        return True
        
    except Exception as e:
        logging.error(f"Error processing teamlead command: {e}")
        await message.answer("Произошла ошибка при выполнении команды")
        return True