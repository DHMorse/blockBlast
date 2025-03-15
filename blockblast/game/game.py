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
from blockblast.ui.components import createCrownIcon, drawBlock, createButton
from blockblast.utils.audio import MusicPlayer

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
        
        # Initialize music player
        self.musicPlayer = MusicPlayer()
        # Start playing music
        self.musicPlayer.start()
        
        # AI button properties
        self.aiButtonRect: Optional[pygame.Rect] = None
        self.aiPlaying: bool = False
        self.aiMoveDelay: int = 500  # Milliseconds between AI moves
        self.lastAiMoveTime: int = 0
        
        # Transposition table for minimax algorithm
        self.transpositionTable: dict[str, tuple[float, Optional[tuple[int, int, int]]]] = {}
    
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
        Draw the current score and high score.
        """
        # Draw score
        scoreText = self.smallFont.render(f"Score: {self.score}", True, WHITE)
        self.screen.blit(scoreText, (20, 20))
        
        # Draw high score with crown icon
        highScoreText = self.smallFont.render(f"High Score: {self.highScore}", True, GOLD)
        self.screen.blit(highScoreText, (SCREEN_WIDTH - 20 - highScoreText.get_width(), 20))
        self.screen.blit(self.crownIcon, (SCREEN_WIDTH - 20 - highScoreText.get_width() - 35, 15))
    
    def drawAiButton(self) -> None:
        """
        Draw the AI button in the top center of the screen.
        """
        buttonWidth = 120
        buttonHeight = 40
        buttonX = (SCREEN_WIDTH - buttonWidth) // 2
        buttonY = 20
        
        # Create the AI button with a purple color
        buttonText = "Stop AI" if self.aiPlaying else "AI Play"
        buttonColor = (255, 0, 0) if self.aiPlaying else (128, 0, 255)  # Red when active, purple when inactive
        
        self.aiButtonRect = createButton(
            self.screen, 
            buttonText, 
            buttonX, 
            buttonY, 
            buttonWidth, 
            buttonHeight, 
            buttonColor,
            WHITE,
            36
        )
    
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
            # Handle music events
            self.musicPlayer.handleEvent(event)
            
            if event.type == pygame.QUIT:
                # Stop music before quitting
                self.musicPlayer.stop()
                return False
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left mouse button
                    mousePos = pygame.mouse.get_pos()
                    
                    # Check if AI button was clicked
                    if self.aiButtonRect and self.aiButtonRect.collidepoint(mousePos):
                        # Toggle AI playing state
                        self.aiPlaying = not self.aiPlaying
                        if self.aiPlaying:
                            print("AI play started!")
                            # Reset the last move time to make the first move immediately
                            self.lastAiMoveTime = 0
                        else:
                            print("AI play stopped!")
                        return True
                    
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
    
    def evaluateGameState(self) -> float:
        """
        Enhanced evaluation function for the minimax algorithm.
        Provides a comprehensive assessment of the game state.
        
        Returns:
            A score representing how good the current state is
        """
        # Base evaluation is the current score
        evaluation: float = float(self.score) * 3.0  # Increase weight of actual score
        
        # Count empty cells - fewer empty cells is worse for future moves
        emptyCells: int = sum(1 for row in self.grid for cell in row if cell is None)
        emptyPercentage: float = emptyCells / (GRID_SIZE * GRID_SIZE)
        evaluation += emptyCells * 5.0  # Significantly increased weight for empty cells
        
        # Penalize for having too few empty cells (less than 40%)
        if emptyPercentage < 0.4:
            evaluation -= (0.4 - emptyPercentage) * 500  # Increased penalty
        
        # Evaluate rows and columns for completion potential
        rowCompletionPotential: float = 0.0
        colCompletionPotential: float = 0.0
        
        for row in range(GRID_SIZE):
            filledInRow: int = sum(1 for cell in self.grid[row] if cell is not None)
            if 0 < filledInRow < GRID_SIZE:  # Partially filled row
                # More filled cells in a row is better (closer to completing)
                # Cubic reward for rows that are close to completion
                completionRatio: float = filledInRow / GRID_SIZE
                rowCompletionPotential += (completionRatio ** 3) * 200
                
                # Extra bonus for rows that are almost complete (>80%)
                if completionRatio > 0.8:
                    rowCompletionPotential += 300
        
        for col in range(GRID_SIZE):
            filledInCol: int = sum(1 for row in range(GRID_SIZE) if self.grid[row][col] is not None)
            if 0 < filledInCol < GRID_SIZE:  # Partially filled column
                # More filled cells in a column is better (closer to completing)
                # Cubic reward for columns that are close to completion
                completionRatio: float = filledInCol / GRID_SIZE
                colCompletionPotential += (completionRatio ** 3) * 200
                
                # Extra bonus for columns that are almost complete (>80%)
                if completionRatio > 0.8:
                    colCompletionPotential += 300
        
        # Add completion potentials to evaluation
        evaluation += rowCompletionPotential + colCompletionPotential
        
        # Evaluate patterns - prefer placing blocks in clusters
        clusterScore: float = 0.0
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                if self.grid[row][col] is not None:
                    # Check adjacent cells (up, down, left, right)
                    adjacentFilled: int = 0
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        newRow, newCol = row + dr, col + dc
                        if (0 <= newRow < GRID_SIZE and 
                            0 <= newCol < GRID_SIZE and 
                            self.grid[newRow][newCol] is not None):
                            adjacentFilled += 1
                    
                    # Reward for having filled adjacent cells (exponential)
                    clusterScore += adjacentFilled ** 2 * 5.0
        
        evaluation += clusterScore
        
        # Evaluate distribution - prefer even distribution of filled cells
        rowFillCounts: list[int] = [sum(1 for cell in row if cell is not None) for row in self.grid]
        colFillCounts: list[int] = [sum(1 for row in range(GRID_SIZE) if self.grid[row][col] is not None) 
                                  for col in range(GRID_SIZE)]
        
        # Calculate standard deviation of fill counts
        rowStdDev: float = self.calculateStandardDeviation(rowFillCounts)
        colStdDev: float = self.calculateStandardDeviation(colFillCounts)
        
        # Penalize uneven distribution (high standard deviation)
        evaluation -= (rowStdDev + colStdDev) * 20.0
        
        # Evaluate placement flexibility - prefer states where more blocks can be placed
        placementFlexibility: float = 0.0
        
        # Check each available block
        for blockIndex in range(len(self.availableBlocks)):
            # Skip used blocks
            if self.usedBlocks[blockIndex]:
                continue
                
            # Count valid positions for this block
            validPositions: int = 0
            for row in range(GRID_SIZE):
                for col in range(GRID_SIZE):
                    if self.canPlaceBlockAtPosition(blockIndex, row, col):
                        validPositions += 1
            
            # Reward for having more valid positions
            placementFlexibility += validPositions * 0.5
        
        evaluation += placementFlexibility
        
        # Evaluate corner and edge usage - prefer keeping corners and edges empty
        cornerPenalty: float = 0.0
        edgePenalty: float = 0.0
        
        # Check corners
        corners = [(0, 0), (0, GRID_SIZE-1), (GRID_SIZE-1, 0), (GRID_SIZE-1, GRID_SIZE-1)]
        for row, col in corners:
            if self.grid[row][col] is not None:
                cornerPenalty += 50.0
        
        # Check edges (excluding corners)
        for i in range(1, GRID_SIZE-1):
            # Top edge
            if self.grid[0][i] is not None:
                edgePenalty += 10.0
            # Bottom edge
            if self.grid[GRID_SIZE-1][i] is not None:
                edgePenalty += 10.0
            # Left edge
            if self.grid[i][0] is not None:
                edgePenalty += 10.0
            # Right edge
            if self.grid[i][GRID_SIZE-1] is not None:
                edgePenalty += 10.0
        
        evaluation -= (cornerPenalty + edgePenalty)
        
        return evaluation
    
    def calculateStandardDeviation(self, values: list[int]) -> float:
        """
        Calculate the standard deviation of a list of values.
        
        Args:
            values: List of integer values
            
        Returns:
            The standard deviation as a float
        """
        if not values:
            return 0.0
            
        mean: float = sum(values) / len(values)
        variance: float = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5
    
    def getValidMoves(self) -> list[tuple[int, int, int]]:
        """
        Get all valid moves for the current game state.
        
        Returns:
            A list of tuples (blockIndex, row, col) representing valid moves
        """
        validMoves: list[tuple[int, int, int]] = []
        
        # Check each available block
        for blockIndex in range(len(self.availableBlocks)):
            # Skip used blocks
            if self.usedBlocks[blockIndex]:
                continue
            
            # Check every position on the grid
            for row in range(GRID_SIZE):
                for col in range(GRID_SIZE):
                    if self.canPlaceBlockAtPosition(blockIndex, row, col):
                        validMoves.append((blockIndex, row, col))
        
        return validMoves
    
    def simulateMove(self, blockIndex: int, row: int, col: int) -> tuple[list[list[Optional[str]]], int, list[bool]]:
        """
        Simulate placing a block without modifying the actual game state.
        
        Args:
            blockIndex: Index of the block to place
            row: Grid row to place the block
            col: Grid column to place the block
            
        Returns:
            A tuple containing the new grid, score increase, and updated usedBlocks
        """
        # Create a copy of the current grid
        newGrid: list[list[Optional[str]]] = [row.copy() for row in self.grid]
        
        # Get the block to place
        block = self.availableBlocks[blockIndex]
        shape = block["shape"]
        colorName = block["colorName"]
        
        # Place the block on the grid copy
        for r in range(len(shape)):
            for c in range(len(shape[0])):
                if shape[r][c]:
                    newGrid[row + r][col + c] = colorName
        
        # Calculate score increase (base score for placing a block)
        scoreIncrease: int = 10
        
        # Create a copy of usedBlocks and mark this block as used
        newUsedBlocks: list[bool] = self.usedBlocks.copy()
        newUsedBlocks[blockIndex] = True
        
        # Check for completed rows
        rowsToRemove: list[int] = []
        for r in range(GRID_SIZE):
            if all(newGrid[r][c] is not None for c in range(GRID_SIZE)):
                rowsToRemove.append(r)
        
        # Check for completed columns
        colsToRemove: list[int] = []
        for c in range(GRID_SIZE):
            if all(newGrid[r][c] is not None for r in range(GRID_SIZE)):
                colsToRemove.append(c)
        
        # Clear completed rows and add to score
        for r in rowsToRemove:
            for c in range(GRID_SIZE):
                newGrid[r][c] = None
            scoreIncrease += 100
        
        # Clear completed columns and add to score
        for c in colsToRemove:
            for r in range(GRID_SIZE):
                newGrid[r][c] = None
            scoreIncrease += 100
        
        return newGrid, scoreIncrease, newUsedBlocks
    
    def hashGameState(self, grid: list[list[Optional[str]]], usedBlocks: list[bool], depth: int, isMaximizing: bool) -> str:
        """
        Create a hash of the current game state for the transposition table.
        
        Args:
            grid: The current grid state
            usedBlocks: The current state of used blocks
            depth: Current search depth
            isMaximizing: Whether this is a maximizing node
            
        Returns:
            A string hash representing the game state
        """
        # Convert grid to a string representation
        gridStr = ''.join(''.join(str(cell) if cell else '_' for cell in row) for row in grid)
        
        # Convert used blocks to a string
        blocksStr = ''.join('1' if used else '0' for used in usedBlocks)
        
        # Combine with depth and player
        return f"{gridStr}|{blocksStr}|{depth}|{1 if isMaximizing else 0}"
    
    def minimax(self, depth: int, alpha: float, beta: float, isMaximizing: bool, 
                currentGrid: list[list[Optional[str]]], currentScore: int, 
                currentUsedBlocks: list[bool]) -> tuple[float, Optional[tuple[int, int, int]]]:
        """
        Enhanced minimax algorithm with alpha-beta pruning to find the absolute best move.
        This version prioritizes accuracy over performance.
        
        Args:
            depth: Current depth in the search tree
            alpha: Alpha value for pruning
            beta: Beta value for pruning
            isMaximizing: Whether this is a maximizing node
            currentGrid: Current grid state
            currentScore: Current score
            currentUsedBlocks: Current state of used blocks
            
        Returns:
            A tuple containing the evaluation score and the best move (or None if at a leaf node)
        """
        # Check transposition table
        stateHash = self.hashGameState(currentGrid, currentUsedBlocks, depth, isMaximizing)
        if stateHash in self.transpositionTable:
            return self.transpositionTable[stateHash]
        
        # Save original game state
        originalGrid = self.grid
        originalScore = self.score
        originalUsedBlocks = self.usedBlocks
        
        # Set temporary game state for evaluation
        self.grid = currentGrid
        self.score = currentScore
        self.usedBlocks = currentUsedBlocks
        
        # Get valid moves in this state
        validMoves = self.getValidMoves()
        
        # Check if we've reached a terminal state or maximum depth
        if depth == 0 or not validMoves:
            # For terminal states, use a more thorough evaluation
            if not validMoves:
                # Game over state - evaluate based on final score and remaining blocks
                remainingBlocks = sum(1 for used in currentUsedBlocks if not used)
                
                # Penalize having unused blocks at game over
                finalEvaluation = float(currentScore) - remainingBlocks * 50.0
                
                # Restore original game state
                self.grid = originalGrid
                self.score = originalScore
                self.usedBlocks = originalUsedBlocks
                
                # Store in transposition table
                result = (finalEvaluation, None)
                self.transpositionTable[stateHash] = result
                return result
            else:
                # Reached maximum depth - use standard evaluation
                evaluation = self.evaluateGameState()
                
                # Restore original game state
                self.grid = originalGrid
                self.score = originalScore
                self.usedBlocks = originalUsedBlocks
                
                # Store in transposition table
                result = (evaluation, None)
                self.transpositionTable[stateHash] = result
                return result
        
        # Initialize best move
        bestMove: Optional[tuple[int, int, int]] = None
        
        # Pre-evaluate moves for better ordering
        moveScores: list[tuple[float, tuple[int, int, int]]] = []
        
        for move in validMoves:
            blockIndex, row, col = move
            
            # Simulate the move
            newGrid, scoreIncrease, newUsedBlocks = self.simulateMove(blockIndex, row, col)
            
            # Quick evaluation of the resulting position
            self.grid = newGrid
            self.score = currentScore + scoreIncrease
            self.usedBlocks = newUsedBlocks
            quickEval = self.evaluateGameState()
            
            # Add a bonus for moves that complete rows or columns
            if scoreIncrease > 10:  # More than the base score means rows/columns were completed
                quickEval += scoreIncrease * 2.0  # Double the importance of completing rows/columns
            
            moveScores.append((quickEval, move))
            
        # Restore game state after pre-evaluation
        self.grid = currentGrid
        self.score = currentScore
        self.usedBlocks = currentUsedBlocks
        
        # Sort moves by their evaluation score
        if isMaximizing:
            # Best moves first for maximizing player
            moveScores.sort(reverse=True, key=lambda x: x[0])
        else:
            # Worst moves first for minimizing player
            moveScores.sort(key=lambda x: x[0])
        
        # Get ordered moves
        orderedMoves = [move for _, move in moveScores]
        
        if isMaximizing:
            maxEval: float = float('-inf')
            
            for move in orderedMoves:
                blockIndex, row, col = move
                
                # Simulate the move
                newGrid, scoreIncrease, newUsedBlocks = self.simulateMove(blockIndex, row, col)
                
                # Check if all blocks have been used after this move
                allBlocksUsed = all(newUsedBlocks)
                
                if allBlocksUsed:
                    # If all blocks are used, we need to simulate generating new blocks
                    # Since we can't predict the new blocks, we'll evaluate the current state
                    # plus a bonus for using all blocks
                    self.grid = newGrid
                    self.score = currentScore + scoreIncrease
                    self.usedBlocks = newUsedBlocks
                    
                    # Evaluate current state with a bonus for using all blocks
                    currentEval = self.evaluateGameState() + 100.0  # Bonus for using all blocks
                    
                    # Restore game state
                    self.grid = currentGrid
                    self.score = currentScore
                    self.usedBlocks = currentUsedBlocks
                    
                    # Update max evaluation and best move
                    if currentEval > maxEval:
                        maxEval = currentEval
                        bestMove = move
                else:
                    # Recursive minimax call
                    eval, _ = self.minimax(depth - 1, alpha, beta, False, newGrid, 
                                          currentScore + scoreIncrease, newUsedBlocks)
                    
                    # Update max evaluation and best move
                    if eval > maxEval:
                        maxEval = eval
                        bestMove = move
                
                # Alpha-beta pruning
                alpha = max(alpha, maxEval)
                if beta <= alpha:
                    break
            
            # Restore original game state
            self.grid = originalGrid
            self.score = originalScore
            self.usedBlocks = originalUsedBlocks
            
            # Store in transposition table
            result = (maxEval, bestMove)
            self.transpositionTable[stateHash] = result
            return result
        else:
            minEval: float = float('inf')
            
            for move in orderedMoves:
                blockIndex, row, col = move
                
                # Simulate the move
                newGrid, scoreIncrease, newUsedBlocks = self.simulateMove(blockIndex, row, col)
                
                # Check if all blocks have been used after this move
                allBlocksUsed = all(newUsedBlocks)
                
                if allBlocksUsed:
                    # If all blocks are used, we need to simulate generating new blocks
                    # Since we can't predict the new blocks, we'll evaluate the current state
                    # plus a bonus for using all blocks
                    self.grid = newGrid
                    self.score = currentScore + scoreIncrease
                    self.usedBlocks = newUsedBlocks
                    
                    # Evaluate current state with a bonus for using all blocks
                    currentEval = self.evaluateGameState() + 100.0  # Bonus for using all blocks
                    
                    # Restore game state
                    self.grid = currentGrid
                    self.score = currentScore
                    self.usedBlocks = currentUsedBlocks
                    
                    # Update min evaluation and best move
                    if currentEval < minEval:
                        minEval = currentEval
                        bestMove = move
                else:
                    # Recursive minimax call
                    eval, _ = self.minimax(depth - 1, alpha, beta, True, newGrid, 
                                          currentScore + scoreIncrease, newUsedBlocks)
                    
                    # Update min evaluation and best move
                    if eval < minEval:
                        minEval = eval
                        bestMove = move
                
                # Alpha-beta pruning
                beta = min(beta, minEval)
                if beta <= alpha:
                    break
            
            # Restore original game state
            self.grid = originalGrid
            self.score = originalScore
            self.usedBlocks = originalUsedBlocks
            
            # Store in transposition table
            result = (minEval, bestMove)
            self.transpositionTable[stateHash] = result
            return result
    
    def makeAiMove(self) -> None:
        """
        Use an exhaustive minimax algorithm to find the absolute best move regardless of time.
        This version prioritizes finding the optimal move over performance.
        """
        if self.gameState != STATE_PLAYING:
            print("Game is not in playing state")
            return
        
        # Check if all blocks have been used
        if all(self.usedBlocks):
            print("Generating new blocks for AI")
            # Generate new blocks
            self.availableBlocks = generateRandomBlocks(MAX_AVAILABLE_BLOCKS)
            # Reset used blocks tracker
            self.usedBlocks = [False] * MAX_AVAILABLE_BLOCKS
            # Check if there are any valid moves with the new blocks
            self.checkForGameOver()
            return
        
        print("AI is thinking exhaustively to find the absolute best move...")
        
        # Clear the transposition table for a fresh search
        self.transpositionTable.clear()
        
        # Get all valid moves
        validMoves = self.getValidMoves()
        
        if not validMoves:
            print("No valid moves available")
            return
        
        # Record start time for statistics
        startTime = pygame.time.get_ticks()
        
        # Use a very deep search to find the best move
        # We'll use a fixed depth of 10, which should be extremely thorough
        searchDepth = 10
        print(f"Performing deep search at depth {searchDepth}...")
        
        # Evaluate each move at the maximum depth
        moveScores: list[tuple[float, tuple[int, int, int]]] = []
        
        for moveIndex, move in enumerate(validMoves):
            blockIndex, row, col = move
            
            # Print progress
            print(f"Analyzing move {moveIndex+1}/{len(validMoves)}: block {blockIndex} at position ({row}, {col})...")
            
            # Simulate the move
            newGrid, scoreIncrease, newUsedBlocks = self.simulateMove(blockIndex, row, col)
            
            # Use minimax to evaluate this move with maximum depth
            score, _ = self.minimax(searchDepth, float('-inf'), float('inf'), True, 
                                   newGrid, self.score + scoreIncrease, newUsedBlocks)
            
            moveScores.append((score, move))
            print(f"  - Evaluation score: {score}")
        
        # Find the best move
        bestScore, bestMove = max(moveScores, key=lambda x: x[0])
        
        # Print stats about the search
        searchTime = pygame.time.get_ticks() - startTime
        print(f"Exhaustive search completed in {searchTime}ms")
        print(f"Transposition table size: {len(self.transpositionTable)}")
        print(f"Positions evaluated: {len(self.transpositionTable)}")
        
        if bestMove:
            blockIndex, row, col = bestMove
            print(f"AI is placing block {blockIndex} at position ({row}, {col}) with score {bestScore}")
            print(f"This is the optimal move found after exhaustive analysis")
            
            # Make the selected move
            self.placeBlock(blockIndex, row, col)
        else:
            print("AI couldn't find a valid move")
    
    def run(self) -> None:
        """
        Main game loop.
        """
        clock = pygame.time.Clock()
        running = True
        
        while running:
            # Handle events
            running = self.handleEvents()
            
            # Make AI move if AI is playing
            currentTime = pygame.time.get_ticks()
            if self.aiPlaying and self.gameState == STATE_PLAYING:
                if currentTime - self.lastAiMoveTime >= self.aiMoveDelay:
                    self.makeAiMove()
                    self.lastAiMoveTime = currentTime
            
            # Draw background
            self.screen.fill(BLUE_BG)
            
            # Draw game elements
            self.drawScore()
            self.drawAiButton()
            self.drawGrid()
            
            if self.gameState == STATE_PLAYING:
                self.drawAvailableBlocks()
            elif self.gameState == STATE_GAME_OVER:
                self.drawGameOver()
                # Stop AI playing when game is over
                self.aiPlaying = False
            
            pygame.display.flip()
            clock.tick(60)