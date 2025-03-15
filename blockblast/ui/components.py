"""
UI components for BlockBlast.
"""
import pygame
from typing import Tuple

from blockblast.utils.constants import GOLD

def createCrownIcon(width: int, height: int) -> pygame.Surface:
    """
    Create a crown icon.
    
    Args:
        width: Width of the icon
        height: Height of the icon
        
    Returns:
        A pygame Surface with the crown icon
    """
    icon = pygame.Surface((width, height), pygame.SRCALPHA)
    
    # Draw crown base
    pygame.draw.rect(icon, GOLD, (5, height-10, width-10, 8), border_radius=2)
    
    # Draw crown points
    pygame.draw.polygon(icon, GOLD, [
        (5, height-10),  # Left bottom
        (5, height-15),  # Left middle
        (width//2-5, height//2),  # Left peak
        (width//2, height//2-5),  # Middle peak
        (width//2+5, height//2),  # Right peak
        (width-5, height-15),  # Right middle
        (width-5, height-10)  # Right bottom
    ])
    
    # Draw crown jewels
    pygame.draw.circle(icon, (255, 0, 0), (width//2, height//2+5), 3)  # Middle jewel
    pygame.draw.circle(icon, (0, 0, 255), (width//4, height//2+8), 2)  # Left jewel
    pygame.draw.circle(icon, (0, 255, 0), (3*width//4, height//2+8), 2)  # Right jewel
    
    return icon

def drawBlock(screen: pygame.Surface, x: int, y: int, color: Tuple[int, int, int], size: int) -> None:
    """
    Draw a single block with 3D effect.
    
    Args:
        screen: Pygame surface to draw on
        x: X coordinate
        y: Y coordinate
        color: RGB color tuple
        size: Size of the block
    """
    # Main block face
    pygame.draw.rect(screen, color, (x, y, size, size))
    
    # Highlight (top-left edges)
    lighter = tuple(min(c + 30, 255) for c in color)
    pygame.draw.polygon(screen, lighter, [
        (x, y),
        (x + size, y),
        (x + size - 5, y + 5),
        (x + 5, y + 5),
        (x + 5, y + size - 5),
        (x, y + size)
    ])
    
    # Shadow (bottom-right edges)
    darker = tuple(max(c - 30, 0) for c in color)
    pygame.draw.polygon(screen, darker, [
        (x + size, y),
        (x + size, y + size),
        (x, y + size),
        (x + 5, y + size - 5),
        (x + size - 5, y + size - 5),
        (x + size - 5, y + 5)
    ]) 