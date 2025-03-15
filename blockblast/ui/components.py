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

def createButton(screen: pygame.Surface, text: str, x: int, y: int, width: int, height: int, 
                color: Tuple[int, int, int], textColor: Tuple[int, int, int], 
                fontSize: int = 24) -> pygame.Rect:
    """
    Create and draw a button with text.
    
    Args:
        screen: Pygame surface to draw on
        text: Text to display on the button
        x: X coordinate of the button
        y: Y coordinate of the button
        width: Width of the button
        height: Height of the button
        color: RGB color tuple for the button background
        textColor: RGB color tuple for the text
        fontSize: Font size for the text
        
    Returns:
        A pygame Rect representing the button's area
    """
    # Create button rectangle
    buttonRect = pygame.Rect(x, y, width, height)
    
    # Draw button
    pygame.draw.rect(screen, color, buttonRect, border_radius=5)
    pygame.draw.rect(screen, tuple(max(c - 40, 0) for c in color), buttonRect, width=2, border_radius=5)
    
    # Create font and render text
    font = pygame.font.Font(None, fontSize)
    textSurface = font.render(text, True, textColor)
    
    # Center text on button
    textRect = textSurface.get_rect(center=buttonRect.center)
    
    # Draw text
    screen.blit(textSurface, textRect)
    
    return buttonRect 