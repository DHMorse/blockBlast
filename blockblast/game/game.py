"""
Main game class for BlockBlast.
"""
import pygame
import sys
from typing import List, Tuple, Dict, Optional, Any

from blockblast.utils.constants import (
    BLUE_BG, DARK_BLUE_GRID, GRID_LINE, WHITE, GOLD, 
    LIGHT_BLUE, RED, GREEN, HIGHLIGHT_COLOR, HIGHLIGHT_ALPHA,
    SCREEN_WIDTH, SCREEN_HEIGHT, GRID_SIZE, CELL_SIZE, GRID_PADDING,
    MAX_AVAILABLE_BLOCKS, STATE_PLAYING, STATE_GAME_OVER
)
from blockblast.utils.score import loadHighScore, saveHighScore
from blockblast.game.blocks import generateRandomBlocks
from blockblast.ui.components import createCrownIcon, drawBlock

class BlockBlastGame:
    """
    Main game class for Block Blast puzzle game.
    Handles game logic, rendering, and user interactions.
    """
    
    def __init__(self) -> None:
        """
        Initialize the game, set up the screen, load assets, and set initial game state.
        """
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Block Blast")
        
        # Load fonts
        self.largeFont = pygame.font.Font(None, 100)
        self.smallFont = pygame.font.Font(None, 36)
        self.tinyFont = pygame.font.Font(None, 24)
        
        # Load high score
        self.highScore: int = loadHighScore()
        
        # Initialize game
        self.initGame()
        
        # Create crown icon
        self.crownIcon = createCrownIcon(30, 30)
    
    def initGame(self) -> None:
        """
        Initialize or reset the game state.
        """
        # Game state
        self.score: int = 0
        self.grid: List[List[Optional[str]]] = [[None for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        self.gameState: int = STATE_PLAYING
        
        # Generate available blocks with random colors
        self.availableBlocks = generateRandomBlocks(MAX_AVAILABLE_BLOCKS)
        
        # Track which blocks have been used
        self.usedBlocks: List[bool] = [False] * MAX_AVAILABLE_BLOCKS
        
        # Game interaction state
        self.selectedBlockIndex: Optional[int] = None
        self.hoverCell: Optional[Tuple[int, int]] = None
        self.canPlaceBlock: bool = False
    
    def placeBlock(self, blockIndex: int, row: int, col: int) -> None:
        """
        Place the selected block on the grid.
        
        Args:
            blockIndex: Index of the selected block
            row: Grid row
            col: Grid column
        """
        block = self.availableBlocks[blockIndex]
        shape = block["shape"]
        colorName = block["colorName"]
        
        # Place the block on the grid
        for r in range(len(shape)):
            for c in range(len(shape[0])):
                if shape[r][c]:
                    self.grid[row + r][col + c] = colorName
        
        # Increase score
        self.score += 10
        
        # Check for completed rows or columns
        self.checkCompletedLines()
        
        # Mark this block as used
        self.usedBlocks[blockIndex] = True
        
        # Check if all blocks have been used
        if all(self.usedBlocks):
            # Generate new blocks
            self.availableBlocks = generateRandomBlocks(MAX_AVAILABLE_BLOCKS)
            # Reset used blocks tracker
            self.usedBlocks = [False] * MAX_AVAILABLE_BLOCKS
            
        # Check if there are any valid moves left
        self.checkForGameOver()
    
    def checkForGameOver(self) -> None:
        """
        Check if there are any valid moves left. If not, set game state to game over.
        """
        # Check each available block
        for blockIndex in range(len(self.availableBlocks)):
            # Skip used blocks
            if self.usedBlocks[blockIndex]:
                continue
                
            # Check if this block can be placed anywhere on the grid
            if self.canBlockBePlacedAnywhere(blockIndex):
                return  # Found a valid move, game can continue
        
        # No valid moves found, game over
        self.gameState = STATE_GAME_OVER
        
        # Save high score
        self.highScore = saveHighScore(self.score, self.highScore)
    
    def canBlockBePlacedAnywhere(self, blockIndex: int) -> bool:
        """
        Check if the given block can be placed anywhere on the grid.
        
        Args:
            blockIndex: Index of the block to check
            
        Returns:
            True if the block can be placed somewhere, False otherwise
        """
        # Check every position on the grid
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                if self.canPlaceBlockAtPosition(blockIndex, row, col):
                    return True
        
        return False
    
    def drawGrid(self) -> None:
        """
        Draw the game grid with placed blocks.
        """
        gridX = (SCREEN_WIDTH - GRID_SIZE * CELL_SIZE) // 2
        gridY = GRID_PADDING + 80
        
        # Draw grid background
        pygame.draw.rect(self.screen, DARK_BLUE_GRID, 
                        (gridX, gridY, 
                         GRID_SIZE * CELL_SIZE, 
                         GRID_SIZE * CELL_SIZE))
        
        # Draw grid lines
        for i in range(GRID_SIZE + 1):
            # Vertical lines
            pygame.draw.line(self.screen, GRID_LINE, 
                            (gridX + i * CELL_SIZE, gridY),
                            (gridX + i * CELL_SIZE, gridY + GRID_SIZE * CELL_SIZE))
            # Horizontal lines
            pygame.draw.line(self.screen, GRID_LINE,
                            (gridX, gridY + i * CELL_SIZE),
                            (gridX + GRID_SIZE * CELL_SIZE, gridY + i * CELL_SIZE))
        
        # Draw placed blocks
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                if self.grid[row][col]:
                    blockX = gridX + col * CELL_SIZE
                    blockY = gridY + row * CELL_SIZE
                    
                    # Extract color index from colorName (format: "color_X")
                    colorName = self.grid[row][col]
                    if colorName and colorName.startswith("color_"):
                        colorIndex = int(colorName.split('_')[1])
                        from blockblast.utils.constants import BLOCK_COLORS
                        color = BLOCK_COLORS[colorIndex]
                        drawBlock(self.screen, blockX, blockY, color, CELL_SIZE)
        
        # Draw hover preview if a block is selected and mouse is over the grid
        if self.gameState == STATE_PLAYING and self.selectedBlockIndex is not None and self.hoverCell is not None:
            hoverRow, hoverCol = self.hoverCell
            selectedBlock = self.availableBlocks[self.selectedBlockIndex]
            shape = selectedBlock["shape"]
            color = selectedBlock["color"]
            
            # Create a semi-transparent surface for the preview
            previewSurface = pygame.Surface(
                (len(shape[0]) * CELL_SIZE, len(shape) * CELL_SIZE), 
                pygame.SRCALPHA
            )
            
            # Draw the block shape on the preview surface
            for row in range(len(shape)):
                for col in range(len(shape[0])):
                    if shape[row][col]:
                        # Draw with transparency
                        blockColor = (*color[:3], 150)  # Add alpha value
                        pygame.draw.rect(
                            previewSurface, 
                            blockColor, 
                            (col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                        )
            
            # Calculate position to draw the preview
            previewX = gridX + hoverCol * CELL_SIZE
            previewY = gridY + hoverRow * CELL_SIZE
            
            # Draw the preview
            self.screen.blit(previewSurface, (previewX, previewY))
            
            # Indicate if the block can be placed
            if not self.canPlaceBlock:
                # Draw a red X over the preview
                pygame.draw.line(self.screen, (255, 0, 0), 
                                (previewX, previewY), 
                                (previewX + len(shape[0]) * CELL_SIZE, previewY + len(shape) * CELL_SIZE), 
                                3)
                pygame.draw.line(self.screen, (255, 0, 0), 
                                (previewX + len(shape[0]) * CELL_SIZE, previewY), 
                                (previewX, previewY + len(shape) * CELL_SIZE), 
                                3)
    
    def drawGameOver(self) -> None:
        """
        Draw the game over screen with a replay button.
        """
        # Draw semi-transparent overlay
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))  # Semi-transparent black
        self.screen.blit(overlay, (0, 0))
        
        # Draw "Game Over" text
        gameOverText = self.largeFont.render("GAME OVER", True, WHITE)
        gameOverRect = gameOverText.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 80))
        self.screen.blit(gameOverText, gameOverRect)
        
        # Draw final score
        scoreText = self.smallFont.render(f"Final Score: {self.score}", True, WHITE)
        scoreRect = scoreText.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 20))
        self.screen.blit(scoreText, scoreRect)
        
        # Draw high score
        highScoreText = self.smallFont.render(f"High Score: {self.highScore}", True, GOLD)
        highScoreRect = highScoreText.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 20))
        self.screen.blit(highScoreText, highScoreRect)
        
        # Draw new high score message if applicable
        if self.score >= self.highScore and self.score > 0:
            newHighScoreText = self.tinyFont.render("New High Score!", True, GOLD)
            newHighScoreRect = newHighScoreText.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 50))
            self.screen.blit(newHighScoreText, newHighScoreRect)
        
        # Draw replay button
        buttonWidth, buttonHeight = 150, 50
        buttonX = (SCREEN_WIDTH - buttonWidth) // 2
        buttonY = SCREEN_HEIGHT // 2 + 80
        
        # Button background
        pygame.draw.rect(self.screen, GREEN, (buttonX, buttonY, buttonWidth, buttonHeight), border_radius=10)
        
        # Button text
        replayText = self.smallFont.render("Play Again", True, WHITE)
        replayRect = replayText.get_rect(center=(buttonX + buttonWidth // 2, buttonY + buttonHeight // 2))
        self.screen.blit(replayText, replayRect)
        
        # Store button position for click detection
        self.replayButtonRect = pygame.Rect(buttonX, buttonY, buttonWidth, buttonHeight)
    
    def drawAvailableBlocks(self) -> None:
        """
        Draw the available blocks at the bottom of the screen.
        """
        blockY = SCREEN_HEIGHT - 150
        
        # Calculate spacing based on number of blocks
        totalBlocks = MAX_AVAILABLE_BLOCKS
        blockWidth = 80
        totalWidth = totalBlocks * blockWidth
        spacing = (SCREEN_WIDTH - totalWidth) // (totalBlocks + 1)
        
        for i, block in enumerate(self.availableBlocks):
            blockX = spacing + i * (blockWidth + spacing)
            
            # Skip drawing used blocks
            if self.usedBlocks[i]:
                continue
                
            # Draw selection highlight if this block is selected
            if self.selectedBlockIndex == i:
                highlightSurface = pygame.Surface((blockWidth + 10, blockWidth + 10), pygame.SRCALPHA)
                highlightSurface.fill((*HIGHLIGHT_COLOR, HIGHLIGHT_ALPHA))  # Semi-transparent white
                self.screen.blit(highlightSurface, (blockX - 5, blockY - 5))
                pygame.draw.rect(self.screen, WHITE, (blockX - 5, blockY - 5, blockWidth + 10, blockWidth + 10), 2)
            
            # Draw the block shape
            shape = block["shape"]
            color = block["color"]
            
            # Calculate block size based on shape dimensions
            maxDim = max(len(shape), max(len(row) for row in shape))
            blockSize = min(25, (blockWidth - 10) // maxDim)  # Ensure blocks fit within the space
            
            # Center the shape in its allocated space
            shapeWidth = len(shape[0]) * blockSize
            shapeHeight = len(shape) * blockSize
            offsetX = (blockWidth - shapeWidth) // 2
            offsetY = (blockWidth - shapeHeight) // 2
            
            for row in range(len(shape)):
                for col in range(len(shape[0])):
                    if shape[row][col]:
                        drawBlock(
                            self.screen,
                            blockX + offsetX + col * blockSize, 
                            blockY + offsetY + row * blockSize, 
                            color, blockSize
                        )
    
    def drawScore(self) -> None:
        """
        Draw the score display at the top of the screen.
        """
        # Draw crown icon
        self.screen.blit(self.crownIcon, (20, 20))
        
        # Draw score next to crown
        scoreText = self.smallFont.render(str(self.score), True, WHITE)
        self.screen.blit(scoreText, (60, 25))
        
        # Draw large score in the center top
        largeScoreText = self.largeFont.render(str(self.score), True, WHITE)
        scoreRect = largeScoreText.get_rect(center=(SCREEN_WIDTH // 2, 60))
        self.screen.blit(largeScoreText, scoreRect)
        
        # Draw high score in small text at the top right
        highScoreText = self.tinyFont.render(f"High: {self.highScore}", True, GOLD)
        self.screen.blit(highScoreText, (SCREEN_WIDTH - 100, 25))
    
    def getGridCoordinates(self) -> Tuple[int, int]:
        """
        Get the top-left coordinates of the grid.
        
        Returns:
            A tuple containing the x and y coordinates of the grid
        """
        gridX = (SCREEN_WIDTH - GRID_SIZE * CELL_SIZE) // 2
        gridY = GRID_PADDING + 80
        return gridX, gridY
    
    def getGridCellFromMouse(self, mousePos: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """
        Convert mouse position to grid cell coordinates.
        
        Args:
            mousePos: The mouse position as (x, y)
            
        Returns:
            Grid cell as (row, col) or None if mouse is outside the grid
        """
        gridX, gridY = self.getGridCoordinates()
        
        # Check if mouse is within grid bounds
        if (gridX <= mousePos[0] <= gridX + GRID_SIZE * CELL_SIZE and
            gridY <= mousePos[1] <= gridY + GRID_SIZE * CELL_SIZE):
            
            # Calculate grid cell
            col = (mousePos[0] - gridX) // CELL_SIZE
            row = (mousePos[1] - gridY) // CELL_SIZE
            
            return row, col
        
        return None
    
    def canPlaceBlockAtPosition(self, blockIndex: int, row: int, col: int) -> bool:
        """
        Check if the selected block can be placed at the given position.
        
        Args:
            blockIndex: Index of the selected block
            row: Grid row
            col: Grid column
            
        Returns:
            True if the block can be placed, False otherwise
        """
        if blockIndex is None:
            return False
        
        block = self.availableBlocks[blockIndex]
        shape = block["shape"]
        
        # Check if the block is within grid bounds
        if (row + len(shape) > GRID_SIZE or 
            col + len(shape[0]) > GRID_SIZE):
            return False
        
        # Check if all cells required by the block are empty
        for r in range(len(shape)):
            for c in range(len(shape[0])):
                if shape[r][c] and self.grid[row + r][col + c] is not None:
                    return False
        
        return True
    
    def checkCompletedLines(self) -> None:
        """
        Check for and clear completed rows and columns.
        """
        # Check rows
        rowsToRemove = []
        for row in range(GRID_SIZE):
            if all(self.grid[row][col] is not None for col in range(GRID_SIZE)):
                rowsToRemove.append(row)
        
        # Check columns
        colsToRemove = []
        for col in range(GRID_SIZE):
            if all(self.grid[row][col] is not None for row in range(GRID_SIZE)):
                colsToRemove.append(col)
        
        # Clear completed rows
        for row in rowsToRemove:
            for col in range(GRID_SIZE):
                self.grid[row][col] = None
            self.score += 100
        
        # Clear completed columns
        for col in colsToRemove:
            for row in range(GRID_SIZE):
                self.grid[row][col] = None
            self.score += 100
    
    def handleEvents(self) -> bool:
        """
        Handle pygame events.
        
        Returns:
            False if the game should quit, True otherwise
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left mouse button
                    mousePos = pygame.mouse.get_pos()
                    
                    # Handle game over state
                    if self.gameState == STATE_GAME_OVER:
                        # Check if replay button was clicked
                        if hasattr(self, 'replayButtonRect') and self.replayButtonRect.collidepoint(mousePos):
                            self.initGame()  # Reset the game
                        return True
                    
                    # Handle playing state
                    # Check if all blocks have been used and user clicked to get new blocks
                    if all(self.usedBlocks):
                        # Generate new blocks
                        self.availableBlocks = generateRandomBlocks(MAX_AVAILABLE_BLOCKS)
                        # Reset used blocks tracker
                        self.usedBlocks = [False] * MAX_AVAILABLE_BLOCKS
                        # Deselect any selected block
                        self.selectedBlockIndex = None
                        # Check if there are any valid moves with the new blocks
                        self.checkForGameOver()
                        continue
                    
                    # Check if clicked on an available block
                    blockY = SCREEN_HEIGHT - 150
                    
                    # Calculate spacing based on number of blocks (same as in drawAvailableBlocks)
                    totalBlocks = MAX_AVAILABLE_BLOCKS
                    blockWidth = 80
                    totalWidth = totalBlocks * blockWidth
                    spacing = (SCREEN_WIDTH - totalWidth) // (totalBlocks + 1)
                    
                    for i in range(len(self.availableBlocks)):
                        # Skip used blocks
                        if self.usedBlocks[i]:
                            continue
                            
                        blockX = spacing + i * (blockWidth + spacing)
                        blockRect = pygame.Rect(blockX - 5, blockY - 5, blockWidth + 10, blockWidth + 10)
                        
                        if blockRect.collidepoint(mousePos):
                            self.selectedBlockIndex = i
                            break
                    
                    # Check if clicked on the grid and a block is selected
                    if self.selectedBlockIndex is not None:
                        gridCell = self.getGridCellFromMouse(mousePos)
                        if gridCell and self.canPlaceBlock:
                            row, col = gridCell
                            self.placeBlock(self.selectedBlockIndex, row, col)
                            self.selectedBlockIndex = None
            
            elif event.type == pygame.MOUSEMOTION and self.gameState == STATE_PLAYING:
                # Update hover cell for block placement preview
                mousePos = pygame.mouse.get_pos()
                self.hoverCell = self.getGridCellFromMouse(mousePos)
                
                # Check if the block can be placed at the hover position
                if self.selectedBlockIndex is not None and self.hoverCell is not None:
                    row, col = self.hoverCell
                    self.canPlaceBlock = self.canPlaceBlockAtPosition(self.selectedBlockIndex, row, col)
        
        return True
    
    def run(self) -> None:
        """
        Main game loop.
        """
        clock = pygame.time.Clock()
        running = True
        
        while running:
            # Handle events
            running = self.handleEvents()
            
            # Draw background
            self.screen.fill(BLUE_BG)
            
            # Draw game elements
            self.drawScore()
            self.drawGrid()
            
            if self.gameState == STATE_PLAYING:
                self.drawAvailableBlocks()
            elif self.gameState == STATE_GAME_OVER:
                self.drawGameOver()
            
            pygame.display.flip()
            clock.tick(60) 