import os
import csv
import logging
import aiohttp
from typing import List, Dict, Union, Optional
from aiogram import Bot, types
from datetime import datetime, timezone, timedelta
from io import StringIO

class TrelloIntegration:
    def __init__(self, api_key: str, token: str, board_id: str):
        logging.info("Initializing TrelloIntegration for AI TeamLead")
        self.api_key = api_key
        self.token = token
        self.board_id = board_id
        self.base_url = "https://api.trello.com/1"
        # Project columns
        self.COLUMNS = ['Backlog', 'Design', 'To Do', 'Doing', 'Code Review', 'Testing', 'Done']
        # Columns where deadlines are not critical
        self.NO_DEADLINE_LISTS = {'Backlog', 'Done'}

    async def get_lists(self) -> List[Dict]:
        """Get all lists from the board"""
        logging.info(f"Fetching lists for board {self.board_id}")
        url = f"{self.base_url}/boards/{self.board_id}/lists"
        params = {
            "key": self.api_key,
            "token": self.token
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    lists = await response.json()
                    logging.info(f"Retrieved {len(lists)} lists")
                    return lists
                else:
                    logging.error(f"Failed to fetch lists: {response.status}")
                    return []

    async def get_cards(self, list_id: str) -> List[Dict]:
        """Get all cards from a specific list"""
        logging.info(f"Fetching cards for list {list_id}")
        url = f"{self.base_url}/lists/{list_id}/cards"
        params = {
            "key": self.api_key,
            "token": self.token,
            "members": "true",
            "fields": "name,desc,due,dueComplete,shortUrl,members,idList,labels"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    cards = await response.json()
                    logging.info(f"Retrieved {len(cards)} cards")
                    return cards
                else:
                    logging.error(f"Failed to fetch cards: {response.status}")
                    return []

    async def create_card(self, list_name: str, name: str, desc: str = "", due: Optional[datetime] = None, members: List[str] = None) -> Dict:
        """Create a new card in specified list"""
        # Get lists to find the target list ID
        lists = await self.get_lists()
        target_list = None
        for lst in lists:
            if lst['name'] == list_name:
                target_list = lst
                break
        
        if not target_list:
            raise ValueError(f"List '{list_name}' not found on board")
        
        url = f"{self.base_url}/cards"
        data = {
            "key": self.api_key,
            "token": self.token,
            "idList": target_list['id'],
            "name": name,
            "desc": desc
        }
        
        if due:
            data["due"] = due.isoformat()
        
        if members:
            data["idMembers"] = ",".join(members)
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=data) as response:
                if response.status == 200:
                    card = await response.json()
                    logging.info(f"Created card: {card['name']}")
                    return card
                else:
                    logging.error(f"Failed to create card: {response.status}")
                    raise Exception(f"Failed to create card: {await response.text()}")

    async def move_card(self, card_id: str, target_list_name: str) -> Dict:
        """Move a card to a different list"""
        # Get lists to find the target list ID
        lists = await self.get_lists()
        target_list = None
        for lst in lists:
            if lst['name'] == target_list_name:
                target_list = lst
                break
        
        if not target_list:
            raise ValueError(f"List '{target_list_name}' not found on board")
        
        url = f"{self.base_url}/cards/{card_id}"
        data = {
            "key": self.api_key,
            "token": self.token,
            "idList": target_list['id']
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.put(url, data=data) as response:
                if response.status == 200:
                    card = await response.json()
                    logging.info(f"Moved card to {target_list_name}")
                    return card
                else:
                    logging.error(f"Failed to move card: {response.status}")
                    raise Exception(f"Failed to move card: {await response.text()}")

    async def get_members(self) -> List[Dict]:
        """Get all board members"""
        url = f"{self.base_url}/boards/{self.board_id}/members"
        params = {
            "key": self.api_key,
            "token": self.token
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    members = await response.json()
                    logging.info(f"Retrieved {len(members)} members")
                    return members
                else:
                    logging.error(f"Failed to fetch members: {response.status}")
                    return []

    def format_deadline(self, due_date: str | None, list_name: str) -> tuple[str, str]:
        """Format deadline with appropriate emoji based on urgency"""
        if not due_date:
            if list_name not in self.NO_DEADLINE_LISTS:
                return "No deadline", "WARNING"
            return "", ""

        due = datetime.fromisoformat(due_date.replace('Z', '+00:00'))
        now = datetime.now(timezone.utc)
        days_diff = (due - now).days
        
        months = ['January', 'February', 'March', 'April', 'May', 'June', 
                 'July', 'August', 'September', 'October', 'November', 'December']
        
        date_str = f"{due.day} {months[due.month - 1]}"
        
        # For certain columns show only date without emotions
        if list_name in self.NO_DEADLINE_LISTS:
            return date_str, ""
        
        # Color coding for urgency
        if days_diff < -1:
            return date_str, "OVERDUE"  # Overdue
        elif days_diff == -1:
            return "Yesterday", "OVERDUE"
        elif days_diff == 0:
            return "Today", "URGENT"
        elif days_diff == 1:
            return "Tomorrow", "URGENT"
        elif days_diff <= 3:
            return date_str, "SOON"  # Soon
        else:
            return date_str, "OK"  # OK

    def estimate_task_complexity(self, task_name: str, task_desc: str = "") -> Dict[str, any]:
        """Basic task complexity estimation based on keywords"""
        # This is a simple heuristic - in production would use AI
        keywords = {
            'simple': ['fix', 'update', 'change', 'modify', 'adjust'],
            'medium': ['implement', 'create', 'add', 'integrate', 'design'],
            'complex': ['refactor', 'architecture', 'migrate', 'optimize', 'redesign']
        }
        
        text = (task_name + " " + task_desc).lower()
        
        # Default estimation
        complexity = 'medium'
        hours = 4
        
        for level, words in keywords.items():
            if any(word in text for word in words):
                complexity = level
                if level == 'simple':
                    hours = 2
                elif level == 'complex':
                    hours = 8
                break
        
        return {
            'complexity': complexity,
            'estimated_hours': hours,
            'estimated_cost': hours * 50  # $50/hour default rate
        }

    async def get_project_status(self) -> Dict[str, any]:
        """Get overall project status with metrics"""
        lists = await self.get_lists()
        all_cards = {}
        total_cards = 0
        
        status = {
            'total_tasks': 0,
            'by_column': {},
            'by_assignee': {},
            'overdue_tasks': [],
            'urgent_tasks': [],
            'no_deadline_tasks': []
        }
        
        now = datetime.now(timezone.utc)
        
        for list_data in lists:
            list_name = list_data['name']
            cards = await self.get_cards(list_data['id'])
            all_cards[list_data['id']] = cards
            total_cards += len(cards)
            
            status['by_column'][list_name] = len(cards)
            
            for card in cards:
                # Count by assignee
                assignee = next((m['fullName'] for m in card.get('members', [])), 'Unassigned')
                status['by_assignee'][assignee] = status['by_assignee'].get(assignee, 0) + 1
                
                # Check deadlines
                due_date = card.get('due')
                if due_date:
                    due = datetime.fromisoformat(due_date.replace('Z', '+00:00'))
                    days_diff = (due - now).days
                    
                    if days_diff < 0:
                        status['overdue_tasks'].append({
                            'name': card['name'],
                            'assignee': assignee,
                            'days_overdue': abs(days_diff),
                            'column': list_name
                        })
                    elif days_diff <= 2:
                        status['urgent_tasks'].append({
                            'name': card['name'],
                            'assignee': assignee,
                            'days_left': days_diff,
                            'column': list_name
                        })
                elif list_name not in self.NO_DEADLINE_LISTS:
                    status['no_deadline_tasks'].append({
                        'name': card['name'],
                        'assignee': assignee,
                        'column': list_name
                    })
        
        status['total_tasks'] = total_cards
        return status

    def create_csv_report(self, lists: List[Dict], cards: Dict[str, List[Dict]], include_estimates: bool = False) -> StringIO:
        """Create detailed CSV report with optional cost estimates"""
        output = StringIO()
        writer = csv.writer(output, delimiter=';')
        
        headers = ['Column', 'Task', 'Assignee', 'Deadline', 'Status', 'Link']
        if include_estimates:
            headers.extend(['Complexity', 'Est. Hours', 'Est. Cost ($)'])
        
        writer.writerow(headers)
        
        for list_data in lists:
            list_name = list_data['name']
            list_id = list_data['id']
            list_cards = cards.get(list_id, [])
            
            for card in list_cards:
                assignee = next((m['fullName'] for m in card.get('members', [])), 'Unassigned')
                deadline_str, status_emoji = self.format_deadline(card.get('due'), list_name)
                
                row = [
                    list_name,
                    card['name'],
                    assignee,
                    deadline_str,
                    status_emoji,
                    card['shortUrl']
                ]
                
                if include_estimates:
                    estimate = self.estimate_task_complexity(card['name'], card.get('desc', ''))
                    row.extend([
                        estimate['complexity'],
                        estimate['estimated_hours'],
                        estimate['estimated_cost']
                    ])
                
                writer.writerow(row)
        
        output.seek(0)
        return output

    def format_project_status(self, status: Dict[str, any]) -> str:
        """Format project status for message"""
        lines = ["**Project Status Overview**\n"]
        
        lines.append(f"**Total Tasks:** {status['total_tasks']}\n")
        
        lines.append("**Tasks by Column:**")
        for column in self.COLUMNS:
            count = status['by_column'].get(column, 0)
            lines.append(f"• {column}: {count}")
        
        lines.append("\n**Tasks by Assignee:**")
        for assignee, count in sorted(status['by_assignee'].items(), key=lambda x: x[1], reverse=True):
            lines.append(f"• {assignee}: {count}")
        
        if status['overdue_tasks']:
            lines.append(f"\n**Overdue Tasks ({len(status['overdue_tasks'])}):**")
            for task in status['overdue_tasks'][:5]:  # Show top 5
                lines.append(f"• {task['name']} ({task['assignee']}) - {task['days_overdue']} days overdue")
        
        if status['urgent_tasks']:
            lines.append(f"\n**Urgent Tasks ({len(status['urgent_tasks'])}):**")
            for task in status['urgent_tasks'][:5]:  # Show top 5
                lines.append(f"• {task['name']} ({task['assignee']}) - {task['days_left']} days left")
        
        if status['no_deadline_tasks']:
            lines.append(f"\n**Tasks Without Deadlines ({len(status['no_deadline_tasks'])}):**")
            for task in status['no_deadline_tasks'][:5]:  # Show top 5
                lines.append(f"• {task['name']} ({task['assignee']}) in {task['column']}")
        
        return "\n".join(lines)