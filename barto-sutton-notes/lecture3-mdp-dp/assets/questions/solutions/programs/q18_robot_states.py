"""
Recycling Robot States: Enum class for states.
"""
from enum import Enum

class RobotState(Enum):
    HIGH = 0
    LOW = 1

if __name__ == "__main__":
    current_state = RobotState.HIGH
    print(f"Current Robot State: {current_state}")
    print(f"State Name: {current_state.name}, State Value: {current_state.value}")
