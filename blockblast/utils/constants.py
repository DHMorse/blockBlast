"""
Constants used throughout the BlockBlast game.
"""
from typing import List, Tuple

# Colors
BLUE_BG: Tuple[int, int, int] = (59, 84, 152)
DARK_BLUE_GRID: Tuple[int, int, int] = (40, 57, 106)
GRID_LINE: Tuple[int, int, int] = (50, 67, 116)
WHITE: Tuple[int, int, int] = (255, 255, 255)
GOLD: Tuple[int, int, int] = (255, 215, 0)
LIGHT_BLUE: Tuple[int, int, int] = (173, 216, 230)
RED: Tuple[int, int, int] = (215, 66, 66)
GREEN: Tuple[int, int, int] = (66, 215, 74)

# Semi-transparent white (used with pygame.SRCALPHA surfaces)
HIGHLIGHT_COLOR: Tuple[int, int, int] = (255, 255, 255)
HIGHLIGHT_ALPHA: int = 100

# Block colors
BLOCK_COLORS: List[Tuple[int, int, int]] = [
    (66, 114, 215),   # Blue
    (242, 150, 58),   # Orange
    (242, 211, 58),   # Yellow
    (187, 114, 215),  # Purple
    (114, 215, 74),   # Green
    (215, 66, 66),    # Red
    (66, 215, 181),   # Teal
    (215, 137, 66),   # Amber
    (137, 66, 215),   # Violet
    (215, 66, 137)    # Pink
]

# Game dimensions
SCREEN_WIDTH: int = 400
SCREEN_HEIGHT: int = 700
GRID_SIZE: int = 8
CELL_SIZE: int = 40
GRID_PADDING: int = 40

# Number of blocks available at a time
MAX_AVAILABLE_BLOCKS: int = 3

# Game states
STATE_PLAYING: int = 0
STATE_GAME_OVER: int = 1

# High score file
HIGH_SCORE_FILE: str = "highscore.txt" 