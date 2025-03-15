import pygame
import sys
import os
import random
from typing import List, Tuple, Dict, Optional, Any

# Initialize pygame
pygame.init()

class BlockBlastGame:
    """
    Main game class for Block Blast puzzle game.
    Handles game logic, rendering, and user interactions.
    """
    
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
    GRID_SIZE: int = 10
    CELL_SIZE: int = 35
    GRID_PADDING: int = 40
    
    # Number of blocks available at a time
    MAX_AVAILABLE_BLOCKS: int = 3
    
    # Game states
    STATE_PLAYING: int = 0
    STATE_GAME_OVER: int = 1
    
    # High score file
    HIGH_SCORE_FILE: str = "highscore.txt"
    
    def __init__(self) -> None:
        """
        Initialize the game, set up the screen, load assets, and set initial game state.
        """
        self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        pygame.display.set_caption("Block Blast")
        
        # Load fonts
        self.largeFont = pygame.font.Font(None, 100)
        self.smallFont = pygame.font.Font(None, 36)
        self.tinyFont = pygame.font.Font(None, 24)
        
        # Load high score
        self.highScore: int = self.loadHighScore()
        
        # Initialize game
        self.initGame()
        
        # Create crown icon
        self.crownIcon = self.createCrownIcon(30, 30)
    
    def loadHighScore(self) -> int:
        """
        Load the high score from file.
        
        Returns:
            The high score as an integer, or 0 if no high score exists
        """
        try:
            if os.path.exists(self.HIGH_SCORE_FILE):
                with open(self.HIGH_SCORE_FILE, 'r') as file:
                    return int(file.read().strip())
        except (IOError, ValueError):
            # If there's an error reading the file or parsing the score, return 0
            pass
        
        return 0
    
    def saveHighScore(self, score: int) -> None:
        """
        Save the high score to file if it's higher than the current high score.
        
        Args:
            score: The score to save
        """
        if score > self.highScore:
            self.highScore = score
            try:
                with open(self.HIGH_SCORE_FILE, 'w') as file:
                    file.write(str(score))
            except IOError:
                # If there's an error writing to the file, just continue
                print(f"Error: Could not save high score to {self.HIGH_SCORE_FILE}")
    
    def initGame(self) -> None:
        """
        Initialize or reset the game state.
        """
        # Game state
        self.score: int = 0
        self.grid: List[List[Optional[str]]] = [[None for _ in range(self.GRID_SIZE)] for _ in range(self.GRID_SIZE)]
        self.gameState: int = self.STATE_PLAYING
        
        # Define block shapes
        self.blockShapes: List[List[List[int]]] = [
            # Lines
            [[1, 1]],

            [[1, 1, 1]],

            [[1, 1, 1, 1]],

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
        
        # Generate available blocks with random colors
        self.availableBlocks = self.generateRandomBlocks()
        
        # Track which blocks have been used
        self.usedBlocks: List[bool] = [False] * self.MAX_AVAILABLE_BLOCKS
        
        # Game interaction state
        self.selectedBlockIndex: Optional[int] = None
        self.hoverCell: Optional[Tuple[int, int]] = None
        self.canPlaceBlock: bool = False
    
    def generateRandomBlocks(self) -> List[Dict[str, Any]]:
        """
        Generate blocks with random colors.
        
        Returns:
            A list of block dictionaries with random colors
        """
        blocks: List[Dict[str, Any]] = []
        
        # Generate MAX_AVAILABLE_BLOCKS random blocks
        for _ in range(self.MAX_AVAILABLE_BLOCKS):
            # Choose a random shape
            shapeIndex = random.randint(0, len(self.blockShapes) - 1)
            shape = self.blockShapes[shapeIndex]
            
            # Choose a random color
            colorIndex = random.randint(0, len(self.BLOCK_COLORS) - 1)
            color = self.BLOCK_COLORS[colorIndex]
            colorName = f"color_{colorIndex}"
            
            blocks.append({
                "shape": shape,
                "color": color,
                "colorName": colorName
            })
        
        return blocks
    
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
            self.availableBlocks = self.generateRandomBlocks()
            # Reset used blocks tracker
            self.usedBlocks = [False] * self.MAX_AVAILABLE_BLOCKS
            
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
        self.gameState = self.STATE_GAME_OVER
        
        # Save high score
        self.saveHighScore(self.score)
    
    def canBlockBePlacedAnywhere(self, blockIndex: int) -> bool:
        """
        Check if the given block can be placed anywhere on the grid.
        
        Args:
            blockIndex: Index of the block to check
            
        Returns:
            True if the block can be placed somewhere, False otherwise
        """
        # Check every position on the grid
        for row in range(self.GRID_SIZE):
            for col in range(self.GRID_SIZE):
                if self.canPlaceBlockAtPosition(blockIndex, row, col):
                    return True
        
        return False
    
    def drawGrid(self) -> None:
        """
        Draw the game grid with placed blocks.
        """
        gridX = (self.SCREEN_WIDTH - self.GRID_SIZE * self.CELL_SIZE) // 2
        gridY = self.GRID_PADDING + 80
        
        # Draw grid background
        pygame.draw.rect(self.screen, self.DARK_BLUE_GRID, 
                        (gridX, gridY, 
                         self.GRID_SIZE * self.CELL_SIZE, 
                         self.GRID_SIZE * self.CELL_SIZE))
        
        # Draw grid lines
        for i in range(self.GRID_SIZE + 1):
            # Vertical lines
            pygame.draw.line(self.screen, self.GRID_LINE, 
                            (gridX + i * self.CELL_SIZE, gridY),
                            (gridX + i * self.CELL_SIZE, gridY + self.GRID_SIZE * self.CELL_SIZE))
            # Horizontal lines
            pygame.draw.line(self.screen, self.GRID_LINE,
                            (gridX, gridY + i * self.CELL_SIZE),
                            (gridX + self.GRID_SIZE * self.CELL_SIZE, gridY + i * self.CELL_SIZE))
        
        # Draw placed blocks
        for row in range(self.GRID_SIZE):
            for col in range(self.GRID_SIZE):
                if self.grid[row][col]:
                    blockX = gridX + col * self.CELL_SIZE
                    blockY = gridY + row * self.CELL_SIZE
                    
                    # Extract color index from colorName (format: "color_X")
                    colorName = self.grid[row][col]
                    if colorName and colorName.startswith("color_"):
                        colorIndex = int(colorName.split('_')[1])
                        color = self.BLOCK_COLORS[colorIndex]
                        self.drawBlock(blockX, blockY, color, self.CELL_SIZE)
        
        # Draw hover preview if a block is selected and mouse is over the grid
        if self.gameState == self.STATE_PLAYING and self.selectedBlockIndex is not None and self.hoverCell is not None:
            hoverRow, hoverCol = self.hoverCell
            selectedBlock = self.availableBlocks[self.selectedBlockIndex]
            shape = selectedBlock["shape"]
            color = selectedBlock["color"]
            
            # Create a semi-transparent surface for the preview
            previewSurface = pygame.Surface(
                (len(shape[0]) * self.CELL_SIZE, len(shape) * self.CELL_SIZE), 
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
                            (col * self.CELL_SIZE, row * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
                        )
            
            # Calculate position to draw the preview
            previewX = gridX + hoverCol * self.CELL_SIZE
            previewY = gridY + hoverRow * self.CELL_SIZE
            
            # Draw the preview
            self.screen.blit(previewSurface, (previewX, previewY))
            
            # Indicate if the block can be placed
            if not self.canPlaceBlock:
                # Draw a red X over the preview
                pygame.draw.line(self.screen, (255, 0, 0), 
                                (previewX, previewY), 
                                (previewX + len(shape[0]) * self.CELL_SIZE, previewY + len(shape) * self.CELL_SIZE), 
                                3)
                pygame.draw.line(self.screen, (255, 0, 0), 
                                (previewX + len(shape[0]) * self.CELL_SIZE, previewY), 
                                (previewX, previewY + len(shape) * self.CELL_SIZE), 
                                3)
    
    def drawGameOver(self) -> None:
        """
        Draw the game over screen with a replay button.
        """
        # Draw semi-transparent overlay
        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))  # Semi-transparent black
        self.screen.blit(overlay, (0, 0))
        
        # Draw "Game Over" text
        gameOverText = self.largeFont.render("GAME OVER", True, self.WHITE)
        gameOverRect = gameOverText.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2 - 80))
        self.screen.blit(gameOverText, gameOverRect)
        
        # Draw final score
        scoreText = self.smallFont.render(f"Final Score: {self.score}", True, self.WHITE)
        scoreRect = scoreText.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2 - 20))
        self.screen.blit(scoreText, scoreRect)
        
        # Draw high score
        highScoreText = self.smallFont.render(f"High Score: {self.highScore}", True, self.GOLD)
        highScoreRect = highScoreText.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2 + 20))
        self.screen.blit(highScoreText, highScoreRect)
        
        # Draw new high score message if applicable
        if self.score >= self.highScore and self.score > 0:
            newHighScoreText = self.tinyFont.render("New High Score!", True, self.GOLD)
            newHighScoreRect = newHighScoreText.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2 + 50))
            self.screen.blit(newHighScoreText, newHighScoreRect)
        
        # Draw replay button
        buttonWidth, buttonHeight = 150, 50
        buttonX = (self.SCREEN_WIDTH - buttonWidth) // 2
        buttonY = self.SCREEN_HEIGHT // 2 + 80
        
        # Button background
        pygame.draw.rect(self.screen, self.GREEN, (buttonX, buttonY, buttonWidth, buttonHeight), border_radius=10)
        
        # Button text
        replayText = self.smallFont.render("Play Again", True, self.WHITE)
        replayRect = replayText.get_rect(center=(buttonX + buttonWidth // 2, buttonY + buttonHeight // 2))
        self.screen.blit(replayText, replayRect)
        
        # Store button position for click detection
        self.replayButtonRect = pygame.Rect(buttonX, buttonY, buttonWidth, buttonHeight)
    
    def drawAvailableBlocks(self) -> None:
        """
        Draw the available blocks at the bottom of the screen.
        """
        blockY = self.SCREEN_HEIGHT - 150
        
        # Calculate spacing based on number of blocks
        totalBlocks = self.MAX_AVAILABLE_BLOCKS
        blockWidth = 80
        totalWidth = totalBlocks * blockWidth
        spacing = (self.SCREEN_WIDTH - totalWidth) // (totalBlocks + 1)
        
        for i, block in enumerate(self.availableBlocks):
            blockX = spacing + i * (blockWidth + spacing)
            
            # Skip drawing used blocks
            if self.usedBlocks[i]:
                continue
                
            # Draw selection highlight if this block is selected
            if self.selectedBlockIndex == i:
                highlightSurface = pygame.Surface((blockWidth + 10, blockWidth + 10), pygame.SRCALPHA)
                highlightSurface.fill((*self.HIGHLIGHT_COLOR, self.HIGHLIGHT_ALPHA))  # Semi-transparent white
                self.screen.blit(highlightSurface, (blockX - 5, blockY - 5))
                pygame.draw.rect(self.screen, self.WHITE, (blockX - 5, blockY - 5, blockWidth + 10, blockWidth + 10), 2)
            
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
                        self.drawBlock(
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
        scoreText = self.smallFont.render(str(self.score), True, self.WHITE)
        self.screen.blit(scoreText, (60, 25))
        
        # Draw large score in the center top
        largeScoreText = self.largeFont.render(str(self.score), True, self.WHITE)
        scoreRect = largeScoreText.get_rect(center=(self.SCREEN_WIDTH // 2, 60))
        self.screen.blit(largeScoreText, scoreRect)
        
        # Draw high score in small text at the top right
        highScoreText = self.tinyFont.render(f"High: {self.highScore}", True, self.GOLD)
        self.screen.blit(highScoreText, (self.SCREEN_WIDTH - 100, 25))
    
    
    def getGridCoordinates(self) -> Tuple[int, int]:
        """
        Get the top-left coordinates of the grid.
        
        Returns:
            A tuple containing the x and y coordinates of the grid
        """
        gridX = (self.SCREEN_WIDTH - self.GRID_SIZE * self.CELL_SIZE) // 2
        gridY = self.GRID_PADDING + 80
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
        if (gridX <= mousePos[0] <= gridX + self.GRID_SIZE * self.CELL_SIZE and
            gridY <= mousePos[1] <= gridY + self.GRID_SIZE * self.CELL_SIZE):
            
            # Calculate grid cell
            col = (mousePos[0] - gridX) // self.CELL_SIZE
            row = (mousePos[1] - gridY) // self.CELL_SIZE
            
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
        if (row + len(shape) > self.GRID_SIZE or 
            col + len(shape[0]) > self.GRID_SIZE):
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
        for row in range(self.GRID_SIZE):
            if all(self.grid[row][col] is not None for col in range(self.GRID_SIZE)):
                rowsToRemove.append(row)
        
        # Check columns
        colsToRemove = []
        for col in range(self.GRID_SIZE):
            if all(self.grid[row][col] is not None for row in range(self.GRID_SIZE)):
                colsToRemove.append(col)
        
        # Clear completed rows
        for row in rowsToRemove:
            for col in range(self.GRID_SIZE):
                self.grid[row][col] = None
            self.score += 100
        
        # Clear completed columns
        for col in colsToRemove:
            for row in range(self.GRID_SIZE):
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
                    if self.gameState == self.STATE_GAME_OVER:
                        # Check if replay button was clicked
                        if hasattr(self, 'replayButtonRect') and self.replayButtonRect.collidepoint(mousePos):
                            self.initGame()  # Reset the game
                        return True
                    
                    # Handle playing state
                    # Check if all blocks have been used and user clicked to get new blocks
                    if all(self.usedBlocks):
                        # Generate new blocks
                        self.availableBlocks = self.generateRandomBlocks()
                        # Reset used blocks tracker
                        self.usedBlocks = [False] * self.MAX_AVAILABLE_BLOCKS
                        # Deselect any selected block
                        self.selectedBlockIndex = None
                        # Check if there are any valid moves with the new blocks
                        self.checkForGameOver()
                        continue
                    
                    # Check if clicked on an available block
                    blockY = self.SCREEN_HEIGHT - 150
                    
                    # Calculate spacing based on number of blocks (same as in drawAvailableBlocks)
                    totalBlocks = self.MAX_AVAILABLE_BLOCKS
                    blockWidth = 80
                    totalWidth = totalBlocks * blockWidth
                    spacing = (self.SCREEN_WIDTH - totalWidth) // (totalBlocks + 1)
                    
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
            
            elif event.type == pygame.MOUSEMOTION and self.gameState == self.STATE_PLAYING:
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
            self.screen.fill(self.BLUE_BG)
            
            # Draw game elements
            self.drawScore()
            self.drawGrid()
            
            if self.gameState == self.STATE_PLAYING:
                self.drawAvailableBlocks()
            elif self.gameState == self.STATE_GAME_OVER:
                self.drawGameOver()
            
            pygame.display.flip()
            clock.tick(60)
        
        pygame.quit()
        sys.exit()

    def createCrownIcon(self, width: int, height: int) -> pygame.Surface:
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
        pygame.draw.rect(icon, self.GOLD, (5, height-10, width-10, 8), border_radius=2)
        
        # Draw crown points
        pygame.draw.polygon(icon, self.GOLD, [
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
    
    def drawBlock(self, x: int, y: int, color: Tuple[int, int, int], size: int) -> None:
        """
        Draw a single block with 3D effect.
        
        Args:
            x: X coordinate
            y: Y coordinate
            color: RGB color tuple
            size: Size of the block
        """
        # Main block face
        pygame.draw.rect(self.screen, color, (x, y, size, size))
        
        # Highlight (top-left edges)
        lighter = tuple(min(c + 30, 255) for c in color)
        pygame.draw.polygon(self.screen, lighter, [
            (x, y),
            (x + size, y),
            (x + size - 5, y + 5),
            (x + 5, y + 5),
            (x + 5, y + size - 5),
            (x, y + size)
        ])
        
        # Shadow (bottom-right edges)
        darker = tuple(max(c - 30, 0) for c in color)
        pygame.draw.polygon(self.screen, darker, [
            (x + size, y),
            (x + size, y + size),
            (x, y + size),
            (x + 5, y + size - 5),
            (x + size - 5, y + size - 5),
            (x + size - 5, y + 5)
        ])

if __name__ == "__main__":
    game = BlockBlastGame()
    game.run() 