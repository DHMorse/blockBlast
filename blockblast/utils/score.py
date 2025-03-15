"""
Score management utilities for BlockBlast.
"""
import os
from typing import Tuple

from blockblast.utils.constants import HIGH_SCORE_FILE

def loadHighScore() -> int:
    """
    Load the high score from file.
    
    Returns:
        The high score as an integer, or 0 if no high score exists
    """
    try:
        if os.path.exists(HIGH_SCORE_FILE):
            with open(HIGH_SCORE_FILE, 'r') as file:
                return int(file.read().strip())
    except (IOError, ValueError):
        # If there's an error reading the file or parsing the score, return 0
        pass
    
    return 0

def saveHighScore(score: int, currentHighScore: int) -> int:
    """
    Save the high score to file if it's higher than the current high score.
    
    Args:
        score: The score to save
        currentHighScore: The current high score
        
    Returns:
        The new high score
    """
    if score > currentHighScore:
        newHighScore = score
        try:
            with open(HIGH_SCORE_FILE, 'w') as file:
                file.write(str(score))
        except IOError:
            # If there's an error writing to the file, just continue
            print(f"Error: Could not save high score to {HIGH_SCORE_FILE}")
        return newHighScore
    
    return currentHighScore 