import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta, timezone
import logging
from typing import List, Dict, Tuple
import io

def create_gantt_chart(tasks: List[Dict]) -> io.BytesIO:
    """
    Create a Gantt chart from task data
    
    Args:
        tasks: List of task dictionaries with:
            - name: Task name
            - start: Start date (datetime)
            - end: End date (datetime)
            - assignee: Person assigned (optional)
            - progress: Progress percentage (0-100, optional)
            
    Returns:
        BytesIO object containing the PNG image
    """
    if not tasks:
        raise ValueError("No tasks provided for Gantt chart")
    
    # Sort tasks by start date
    tasks = sorted(tasks, key=lambda x: x['start'])
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, max(6, len(tasks) * 0.5)))
    
    # Color mapping for different assignees
    assignees = list(set(task.get('assignee', 'Unassigned') for task in tasks))
    colors = plt.cm.Set3(range(len(assignees)))
    color_map = dict(zip(assignees, colors))
    
    # Plot tasks
    y_positions = range(len(tasks))
    
    for idx, task in enumerate(tasks):
        start_date = task['start']
        end_date = task['end']
        duration = (end_date - start_date).days + 1
        assignee = task.get('assignee', 'Unassigned')
        progress = task.get('progress', 0)
        
        # Main task bar
        ax.barh(idx, duration, left=start_date, height=0.5,
                color=color_map[assignee], alpha=0.8, 
                label=assignee if assignee not in ax.get_legend_handles_labels()[1] else "")
        
        # Progress bar
        if progress > 0:
            progress_duration = duration * (progress / 100)
            ax.barh(idx, progress_duration, left=start_date, height=0.5,
                    color=color_map[assignee], alpha=1.0)
        
        # Task name
        ax.text(start_date - timedelta(days=1), idx, task['name'], 
                ha='right', va='center', fontsize=9)
        
        # Progress percentage
        if progress > 0:
            ax.text(start_date + timedelta(days=duration/2), idx, 
                    f"{progress}%", ha='center', va='center', 
                    fontsize=8, color='white', weight='bold')
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, duration // 10)))
    
    # Set labels and title
    ax.set_xlabel('Timeline', fontsize=12)
    ax.set_title('Project Gantt Chart', fontsize=16, weight='bold')
    
    # Remove y-axis ticks
    ax.set_yticks([])
    
    # Add grid
    ax.grid(True, axis='x', alpha=0.3)
    
    # Add legend for assignees
    if len(assignees) > 1:
        ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    
    # Add current date line
    today = datetime.now(timezone.utc)
    if tasks[0]['start'] <= today <= tasks[-1]['end']:
        ax.axvline(x=today, color='red', linestyle='--', alpha=0.7, label='Today')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save to BytesIO
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
    plt.close()
    
    img_buffer.seek(0)
    return img_buffer

def parse_tasks_for_gantt(task_description: str, user_id: int) -> List[Dict]:
    """
    Parse task description to extract Gantt chart data
    
    Args:
        task_description: Natural language description of tasks
        user_id: User ID for Claude API
        
    Returns:
        List of task dictionaries ready for Gantt chart
    """
    tasks = []
    today = datetime.now(timezone.utc)
    
    # Parse the response from Claude
    if task_description:
        lines = task_description.split('\n')
        current_date = today
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Look for task patterns
            import re
            
            # Pattern: "Task name" or "- Task name"
            task_match = re.match(r'^[-•*]?\s*(.+?)(?:\s*[-–]\s*(.+?))?$', line)
            
            if task_match:
                task_name = task_match.group(1).strip()
                details = task_match.group(2) if task_match.group(2) else ""
                
                # Skip non-task lines
                if any(skip in task_name.lower() for skip in ['task:', 'задача:', 'start date', 'end date', 'assignee', 'status']):
                    continue
                
                # Default values
                duration = 5  # days
                assignee = 'Team'
                progress = 0
                
                # Extract duration from various patterns
                duration_patterns = [
                    r'(\d+)[-–](\d+)\s*(?:days?|дн|день|дней)',
                    r'(\d+)\s*(?:days?|дн|день|дней)',
                    r'оценка[:\s]*(\d+)[-–](\d+)',
                ]
                
                for pattern in duration_patterns:
                    duration_match = re.search(pattern, line, re.IGNORECASE)
                    if duration_match:
                        if duration_match.group(2) if len(duration_match.groups()) > 1 else None:
                            # Range: take average
                            duration = (int(duration_match.group(1)) + int(duration_match.group(2))) // 2
                        else:
                            duration = int(duration_match.group(1))
                        break
                
                # Extract assignee
                assignee_match = re.search(r'(?:assignee|исполнитель|ответственный)[:\s]*([^,\n]+)', line, re.IGNORECASE)
                if assignee_match:
                    assignee = assignee_match.group(1).strip()
                elif '@' in line:
                    assignee = line.split('@')[1].split()[0]
                
                # Extract progress
                progress_match = re.search(r'(\d+)%', line)
                if progress_match:
                    progress = int(progress_match.group(1))
                
                # Clean task name
                task_name = re.sub(r'\s*\(.*?\)\s*', '', task_name)  # Remove parentheses
                task_name = re.sub(r'\s*[-–]\s*\d+.*', '', task_name)  # Remove duration at end
                task_name = task_name.strip()
                
                if task_name and len(task_name) > 2:  # Valid task name
                    task = {
                        'name': task_name[:50],  # Limit length
                        'start': current_date,
                        'end': current_date + timedelta(days=duration),
                        'assignee': assignee,
                        'progress': progress
                    }
                    tasks.append(task)
                    
                    # Move current date forward
                    current_date = current_date + timedelta(days=max(1, duration // 2))
    
    # If no tasks parsed or too few, create sample tasks
    if len(tasks) < 3:
        sample_tasks = [
            {
                'name': 'Анализ требований',
                'start': today,
                'end': today + timedelta(days=3),
                'assignee': 'Аналитик',
                'progress': 100
            },
            {
                'name': 'Проектирование',
                'start': today + timedelta(days=2),
                'end': today + timedelta(days=7),
                'assignee': 'Архитектор',
                'progress': 80
            },
            {
                'name': 'Разработка MVP',
                'start': today + timedelta(days=5),
                'end': today + timedelta(days=15),
                'assignee': 'Разработчики',
                'progress': 30
            },
            {
                'name': 'Тестирование',
                'start': today + timedelta(days=12),
                'end': today + timedelta(days=18),
                'assignee': 'QA',
                'progress': 0
            },
            {
                'name': 'Деплой',
                'start': today + timedelta(days=17),
                'end': today + timedelta(days=20),
                'assignee': 'DevOps',
                'progress': 0
            }
        ]
        return sample_tasks
    
    return tasks

async def generate_gantt_from_trello(trello_integration) -> io.BytesIO:
    """
    Generate Gantt chart from Trello board data
    
    Args:
        trello_integration: TrelloIntegration instance
        
    Returns:
        BytesIO object containing the PNG image
    """
    try:
        # Get all lists and cards
        lists = await trello_integration.get_lists()
        all_tasks = []
        
        # Map column names to progress percentages
        progress_map = {
            'Backlog': 0,
            'Design': 15,
            'To Do': 25,
            'Doing': 50,
            'Code Review': 75,
            'Testing': 90,
            'Done': 100
        }
        
        today = datetime.now(timezone.utc)
        
        for list_data in lists:
            list_name = list_data['name']
            cards = await trello_integration.get_cards(list_data['id'])
            
            for card in cards:
                # Skip cards without due dates
                if not card.get('due'):
                    continue
                    
                # Convert Trello date to timezone-aware datetime
                due_str = card['due']
                if due_str.endswith('Z'):
                    due_date = datetime.fromisoformat(due_str.replace('Z', '+00:00'))
                else:
                    due_date = datetime.fromisoformat(due_str)
                
                # Ensure it's timezone-aware
                if due_date.tzinfo is None:
                    due_date = due_date.replace(tzinfo=timezone.utc)
                
                # Estimate start date based on column
                if list_name == 'Done':
                    start_date = due_date - timedelta(days=1)
                elif list_name in ['Testing', 'Code Review']:
                    start_date = due_date - timedelta(days=3)
                elif list_name == 'Doing':
                    start_date = today
                else:
                    start_date = today + timedelta(days=1)
                
                assignee = next((m['fullName'] for m in card.get('members', [])), 'Unassigned')
                
                task = {
                    'name': card['name'][:30] + '...' if len(card['name']) > 30 else card['name'],
                    'start': start_date,
                    'end': due_date,
                    'assignee': assignee,
                    'progress': progress_map.get(list_name, 0)
                }
                
                all_tasks.append(task)
        
        if not all_tasks:
            # Create sample tasks if no tasks with deadlines
            logging.warning("No tasks with deadlines found, creating sample Gantt chart")
            all_tasks = parse_tasks_for_gantt("", 0)
        
        return create_gantt_chart(all_tasks)
        
    except Exception as e:
        logging.error(f"Error generating Gantt chart from Trello: {e}")
        raise