from aiogram.fsm.state import State, StatesGroup

class BotStates(StatesGroup):
    waiting_for_clarification = State()
    collecting_forwards = State()
    waiting_for_rules = State()
