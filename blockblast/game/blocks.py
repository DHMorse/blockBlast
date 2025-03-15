"""
Block shapes and block-related functionality for BlockBlast.
"""
from typing import List, Dict, Any, Tuple
import random

from blockblast.utils.constants import BLOCK_COLORS

# Define block shapes
BLOCK_SHAPES: List[List[List[int]]] = [
    # Lines
    [[1, 1]],
    [[1, 1, 1]],
    [[1, 1, 1, 1, 1]],
    [[1], [1], [1], [1]],
    
    # L-shape
    [[1, 1], [1, 0], [1, 0]],
    [[1, 1], [0, 1], [0, 1]],
    [[1, 0, 0], [1, 1, 1]],
    [[1, 1, 1], [0, 0, 1]],

    # T-shape
    [[1, 1, 1], [0, 1, 0]],
    [[0, 1, 0], [1, 1, 1]],
    [[0, 1], [1, 1], [0, 1]],
    [[1, 0], [1, 1], [1, 0]],
    
    # Z-shape
    [[1, 1, 0], [0, 1, 1]],
    [[0, 1, 1], [1, 1, 0]],
    [[1, 0], [1, 1], [0, 1]],
    [[0, 1], [1, 1], [1, 0]],
    
    # Block shapes
    [[1, 1], [1, 1]],
    [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
    [[1, 1, 1], [1, 1, 1]],
    [[1, 1], [1, 1], [1, 1]],

    # Weird shapes
    [[1, 1, 1], [0, 0, 1], [0, 0, 1]],
    [[1, 1], [0, 1]],
    [[1, 1], [1, 0]],
    [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
]

def generateRandomBlocks(maxBlocks: int) -> List[Dict[str, Any]]:
    """
    Generate blocks with random colors and shapes.
    
    Args:
        maxBlocks: The number of blocks to generate
        
    Returns:
        A list of block dictionaries with random colors and shapes
    """
    blocks: List[Dict[str, Any]] = []
    
    # Generate maxBlocks random blocks
    for _ in range(maxBlocks):
        # Choose a random shape
        shapeIndex = random.randint(0, len(BLOCK_SHAPES) - 1)
        shape = BLOCK_SHAPES[shapeIndex]
        
        # Choose a random color
        colorIndex = random.randint(0, len(BLOCK_COLORS) - 1)
        color = BLOCK_COLORS[colorIndex]
        colorName = f"color_{colorIndex}"
        
        blocks.append({
            "shape": shape,
            "color": color,
            "colorName": colorName
        })
    
    return blocks 