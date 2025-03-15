"""
AI implementation for BlockBlast game using minimax algorithm with alpha-beta pruning.
"""
from typing import Dict, List, Tuple, Optional, Any, Callable
import copy
import time
import threading
from dataclasses import dataclass
import hashlib
import json

from blockblast.utils.constants import GRID_SIZE, MAX_AVAILABLE_BLOCKS

@dataclass
class MinimaxResult:
    """
    Class to store the result of a minimax search.
    
    Attributes:
        score: The evaluation score of the position
        blockIndex: The index of the block to place
        position: The position (row, col) to place the block
    """
    score: int
    blockIndex: Optional[int]
    position: Optional[Tuple[int, int]]

class TranspositionTable:
    """
    Transposition table to store previously computed game states.
    """
    
    def __init__(self) -> None:
        """
        Initialize an empty transposition table.
        """
        self.table: Dict[str, Tuple[int, int, Optional[int], Optional[Tuple[int, int]]]] = {}
        self.hits: int = 0
        self.stores: int = 0
    
    def getHash(self, grid: List[List[Optional[str]]], availableBlocks: List[Dict[str, Any]], 
                usedBlocks: List[bool]) -> str:
        """
        Generate a hash key for the current game state.
        
        Args:
            grid: The game grid
            availableBlocks: List of available blocks
            usedBlocks: List indicating which blocks have been used
            
        Returns:
            A string hash representing the game state
        """
        # Convert grid to a string representation
        gridStr = json.dumps(grid)
        
        # Convert available blocks to a string representation
        # We only care about the shape and which blocks are used
        blocksStr = ""
        for i, block in enumerate(availableBlocks):
            if not usedBlocks[i]:
                blocksStr += json.dumps(block["shape"])
        
        # Combine and hash
        stateStr = gridStr + blocksStr
        return hashlib.md5(stateStr.encode()).hexdigest()
    
    def lookup(self, grid: List[List[Optional[str]]], availableBlocks: List[Dict[str, Any]], usedBlocks: List[bool], 
               depth: int, alpha: int, beta: int) -> Optional[MinimaxResult]:
        """
        Look up a position in the transposition table.
        
        Args:
            grid: The game grid
            availableBlocks: List of available blocks
            usedBlocks: List indicating which blocks have been used
            depth: Current search depth
            alpha: Alpha value for alpha-beta pruning
            beta: Beta value for alpha-beta pruning
            
        Returns:
            MinimaxResult if the position is found and valid, None otherwise
        """
        key = self.getHash(grid, availableBlocks, usedBlocks)
        
        if key in self.table:
            storedDepth, storedScore, storedBlockIndex, storedPosition = self.table[key]
            
            # Only use the stored result if it was searched to at least the same depth
            if storedDepth >= depth:
                self.hits += 1
                return MinimaxResult(storedScore, storedBlockIndex, storedPosition)
        
        return None
    
    def store(self, grid: List[List[Optional[str]]], availableBlocks: List[Dict[str, Any]], 
             usedBlocks: List[bool], depth: int, score: int, 
             blockIndex: Optional[int], position: Optional[Tuple[int, int]]) -> None:
        """
        Store a position in the transposition table.
        
        Args:
            grid: The game grid
            availableBlocks: List of available blocks
            usedBlocks: List indicating which blocks have been used
            depth: Current search depth
            score: Evaluation score
            blockIndex: Index of the best block to place
            position: Position (row, col) to place the block
        """
        key = self.getHash(grid, availableBlocks, usedBlocks)
        self.table[key] = (depth, score, blockIndex, position)
        self.stores += 1

class BlockBlastAI:
    """
    AI implementation for BlockBlast game using minimax algorithm with alpha-beta pruning.
    """
    
    def __init__(self) -> None:
        """
        Initialize the AI.
        """
        self.transpositionTable = TranspositionTable()
        self.nodesExplored: int = 0
        self.maxDepth: int = 10  # Maximum search depth
        self.progressUpdateInterval: int = 1000  # Print progress every 1000 nodes
        self.startTime: float = 0
        self.totalMoves: int = 0
        self.currentMoveIndex: int = 0
        self.depthDistribution: Dict[int, int] = {}  # Track nodes explored at each depth
        
        # Thread-related attributes
        self.searchThread: Optional[threading.Thread] = None
        self.isSearching: bool = False
        self.searchResult: Optional[Tuple[Optional[int], Optional[Tuple[int, int]]]] = None
        self.onSearchComplete: Optional[Callable[[Optional[int], Optional[Tuple[int, int]]], None]] = None
        self.shouldCancelSearch: bool = False  # Flag to signal search cancellation
    
    def findBestMoveThreaded(self, game: Any, callback: Callable[[Optional[int], Optional[Tuple[int, int]]], None]) -> None:
        """
        Start a new thread to find the best move for the current game state.
        
        Args:
            game: The BlockBlastGame instance
            callback: Function to call when search is complete with (blockIndex, position) as arguments
        """
        if self.isSearching:
            print("AI is already searching for a move")
            return
        
        self.isSearching = True
        self.onSearchComplete = callback
        
        # Create a deep copy of the game state to avoid thread safety issues
        gridCopy = copy.deepcopy(game.grid)
        availableBlocksCopy = copy.deepcopy(game.availableBlocks)
        usedBlocksCopy = copy.deepcopy(game.usedBlocks)
        
        # Start a new thread for the search
        self.searchThread = threading.Thread(
            target=self._threadedSearch,
            args=(gridCopy, availableBlocksCopy, usedBlocksCopy),
            daemon=True  # Make thread a daemon so it exits when the main program exits
        )
        self.searchThread.start()
    
    def _threadedSearch(self, grid: List[List[Optional[str]]], availableBlocks: List[Dict[str, Any]], 
                       usedBlocks: List[bool]) -> None:
        """
        Run the minimax search in a separate thread.
        
        Args:
            grid: Copy of the game grid
            availableBlocks: Copy of the available blocks
            usedBlocks: Copy of the used blocks
        """
        try:
            # Reset statistics
            self.transpositionTable = TranspositionTable()
            self.nodesExplored = 0
            self.startTime = time.time()
            self.depthDistribution = {d: 0 for d in range(self.maxDepth + 1)}
            self.topMoves = []
            self.shouldCancelSearch = False
            
            # Calculate total possible moves for progress tracking
            self.totalMoves = self.countPossibleMoves(grid, availableBlocks, usedBlocks)
            self.currentMoveIndex = 0
            
            print(f"Starting AI search with {self.totalMoves} possible moves to evaluate")
            print(f"Search depth: {self.maxDepth}")
            
            # Start the minimax search
            result = self.minimax(
                grid,
                availableBlocks,
                usedBlocks,
                0,
                float('-inf'),
                float('inf')
            )
            
            endTime = time.time()
            
            # Print statistics
            print(f"AI search completed in {endTime - self.startTime:.2f} seconds")
            print(f"Nodes explored: {self.nodesExplored}")
            print(f"Transposition table hits: {self.transpositionTable.hits}")
            print(f"Transposition table stores: {self.transpositionTable.stores}")
            
            # Print depth distribution
            self.printDepthDistribution()
            
            # Print top moves
            self.printTopMoves()
            
            if result.blockIndex is not None and result.position is not None:
                print(f"Best move: Block {result.blockIndex} at position {result.position} with score {result.score}")
                self.searchResult = (result.blockIndex, result.position)
            else:
                print("No valid moves found")
                self.searchResult = (None, None)
                
            # Call the callback with the result
            if self.onSearchComplete and not self.shouldCancelSearch:
                self.onSearchComplete(*self.searchResult)
        
        except Exception as e:
            print(f"Error in AI search thread: {e}")
            self.searchResult = (None, None)
            if self.onSearchComplete and not self.shouldCancelSearch:
                self.onSearchComplete(None, None)
        
        finally:
            self.isSearching = False
    
    def findBestMove(self, game: Any) -> Tuple[Optional[int], Optional[Tuple[int, int]]]:
        """
        Find the best move for the current game state using minimax with alpha-beta pruning.
        This is the synchronous version of the search.
        
        Args:
            game: The BlockBlastGame instance
            
        Returns:
            A tuple containing (blockIndex, (row, col)) for the best move, or (None, None) if no move is possible
        """
        # Reset statistics
        self.transpositionTable = TranspositionTable()
        self.nodesExplored = 0
        self.startTime = time.time()
        self.depthDistribution = {d: 0 for d in range(self.maxDepth + 1)}
        
        # For tracking the top moves
        self.topMoves: List[Tuple[int, Tuple[int, int], int]] = []  # [(blockIndex, (row, col), score)]
        
        # Calculate total possible moves for progress tracking
        self.totalMoves = self.countPossibleMoves(game.grid, game.availableBlocks, game.usedBlocks)
        self.currentMoveIndex = 0
        
        print(f"Starting AI search with {self.totalMoves} possible moves to evaluate")
        print(f"Search depth: {self.maxDepth}")
        
        # Start the minimax search
        result = self.minimax(
            copy.deepcopy(game.grid),
            copy.deepcopy(game.availableBlocks),
            copy.deepcopy(game.usedBlocks),
            0,
            float('-inf'),
            float('inf')
        )
        
        endTime = time.time()
        
        # Print statistics
        print(f"AI search completed in {endTime - self.startTime:.2f} seconds")
        print(f"Nodes explored: {self.nodesExplored}")
        print(f"Transposition table hits: {self.transpositionTable.hits}")
        print(f"Transposition table stores: {self.transpositionTable.stores}")
        
        # Print depth distribution
        self.printDepthDistribution()
        
        # Print top moves
        self.printTopMoves()
        
        if result.blockIndex is not None and result.position is not None:
            print(f"Best move: Block {result.blockIndex} at position {result.position} with score {result.score}")
            return result.blockIndex, result.position
        else:
            print("No valid moves found")
            return None, None
    
    def isSearchInProgress(self) -> bool:
        """
        Check if a search is currently in progress.
        
        Returns:
            True if a search is in progress, False otherwise
        """
        return self.isSearching
    
    def printDepthDistribution(self) -> None:
        """
        Print the distribution of nodes explored at each depth.
        """
        print("\nSearch depth distribution:")
        print("Depth | Nodes explored | Percentage")
        print("-" * 40)
        
        for depth in range(self.maxDepth + 1):
            if depth in self.depthDistribution:
                nodes = self.depthDistribution[depth]
                percentage = (nodes / self.nodesExplored) * 100 if self.nodesExplored > 0 else 0
                print(f"{depth:5d} | {nodes:13d} | {percentage:8.2f}%")
        
        print("-" * 40)
    
    def countPossibleMoves(self, grid: List[List[Optional[str]]], availableBlocks: List[Dict[str, Any]], 
                          usedBlocks: List[bool]) -> int:
        """
        Count the total number of possible moves in the current position.
        
        Args:
            grid: The game grid
            availableBlocks: List of available blocks
            usedBlocks: List indicating which blocks have been used
            
        Returns:
            The total number of possible moves
        """
        totalMoves = 0
        
        for blockIndex in range(len(availableBlocks)):
            if usedBlocks[blockIndex]:
                continue
                
            block = availableBlocks[blockIndex]
            shape = block["shape"]
            
            for row in range(GRID_SIZE - len(shape) + 1):
                for col in range(GRID_SIZE - len(shape[0]) + 1):
                    if self.canPlaceBlockAtPosition(grid, shape, row, col):
                        totalMoves += 1
        
        return totalMoves
    
    def minimax(self, grid: List[List[Optional[str]]], availableBlocks: List[Dict[str, Any]], 
               usedBlocks: List[bool], depth: int, alpha: float, beta: float) -> MinimaxResult:
        """
        Minimax algorithm with alpha-beta pruning.
        
        Args:
            grid: The game grid
            availableBlocks: List of available blocks
            usedBlocks: List indicating which blocks have been used
            depth: Current search depth
            alpha: Alpha value for alpha-beta pruning
            beta: Beta value for alpha-beta pruning
            
        Returns:
            MinimaxResult containing the best score, block index, and position
        """
        # Check if search should be cancelled
        if self.shouldCancelSearch:
            return MinimaxResult(0, None, None)
            
        self.nodesExplored += 1
        
        # Track depth distribution
        self.depthDistribution[depth] = self.depthDistribution.get(depth, 0) + 1
        
        # Print progress updates
        if self.nodesExplored % self.progressUpdateInterval == 0:
            elapsedTime = time.time() - self.startTime
            print(f"Search progress: {self.nodesExplored} nodes explored in {elapsedTime:.2f} seconds")
            print(f"Current depth: {depth}, Transposition table hits: {self.transpositionTable.hits}")
            
            # Print current depth distribution every 10,000 nodes
            if self.nodesExplored % (self.progressUpdateInterval * 10) == 0:
                self.printDepthDistribution()
        
        # Check transposition table
        ttResult = self.transpositionTable.lookup(grid, availableBlocks, usedBlocks, depth, alpha, beta)
        if ttResult is not None:
            return ttResult
        
        # Check if we've reached the maximum depth or if the game is over
        if depth >= self.maxDepth or self.isGameOver(availableBlocks, usedBlocks, grid):
            score = self.evaluatePosition(grid, availableBlocks, usedBlocks)
            result = MinimaxResult(score, None, None)
            self.transpositionTable.store(grid, availableBlocks, usedBlocks, depth, score, None, None)
            return result
        
        bestScore = float('-inf')
        bestBlockIndex = None
        bestPosition = None
        
        # Try each available block
        for blockIndex in range(len(availableBlocks)):
            # Skip used blocks
            if usedBlocks[blockIndex]:
                continue
            
            block = availableBlocks[blockIndex]
            shape = block["shape"]
            
            # Try each possible position on the grid
            for row in range(GRID_SIZE - len(shape) + 1):
                for col in range(GRID_SIZE - len(shape[0]) + 1):
                    # Check if the block can be placed at this position
                    if self.canPlaceBlockAtPosition(grid, shape, row, col):
                        # Update progress for top-level search
                        if depth == 0:
                            self.currentMoveIndex += 1
                            if self.currentMoveIndex % 5 == 0 or self.currentMoveIndex == 1:  # Print every 5 moves or the first move
                                progress = (self.currentMoveIndex / self.totalMoves) * 100
                                print(f"Top-level progress: {self.currentMoveIndex}/{self.totalMoves} moves ({progress:.1f}%)")
                                print(f"Evaluating block {blockIndex} at position ({row}, {col})")
                        
                        # Make the move
                        newGrid = copy.deepcopy(grid)
                        newUsedBlocks = copy.deepcopy(usedBlocks)
                        
                        # Place the block
                        self.placeBlock(newGrid, shape, block["colorName"], row, col)
                        newUsedBlocks[blockIndex] = True
                        
                        # Check for completed lines and update the grid
                        linesCleared = self.clearCompletedLines(newGrid)
                        
                        # If all blocks have been used, generate new blocks
                        newAvailableBlocks = copy.deepcopy(availableBlocks)
                        if all(newUsedBlocks):
                            # In a real game, we would generate new random blocks here
                            # For the AI, we'll just reset the used blocks
                            newUsedBlocks = [False] * MAX_AVAILABLE_BLOCKS
                        
                        # Recursive minimax call
                        moveScore = self.minimax(
                            newGrid, 
                            newAvailableBlocks, 
                            newUsedBlocks, 
                            depth + 1, 
                            alpha, 
                            beta
                        ).score
                        
                        # Add immediate score from placing the block and clearing lines
                        immediateScore = 10 + linesCleared * 100  # 10 for placing block, 100 per line cleared
                        moveScore += immediateScore
                        
                        # Update best move
                        if moveScore > bestScore:
                            bestScore = moveScore
                            bestBlockIndex = blockIndex
                            bestPosition = (row, col)
                            
                            # Print when we find a better move at the top level
                            if depth == 0:
                                print(f"Found better move: Block {blockIndex} at ({row}, {col}) with score {moveScore}")
                                # Track top moves at depth 0
                                self.topMoves.append((blockIndex, (row, col), moveScore))
                        
                        # Alpha-beta pruning
                        alpha = max(alpha, bestScore)
                        if beta <= alpha:
                            # Store the result in the transposition table before pruning
                            self.transpositionTable.store(
                                grid, availableBlocks, usedBlocks, depth, bestScore, bestBlockIndex, bestPosition
                            )
                            if depth == 0:
                                print(f"Alpha-beta pruning at depth {depth}")
                            return MinimaxResult(bestScore, bestBlockIndex, bestPosition)
        
        # If no valid moves were found
        if bestBlockIndex is None:
            bestScore = self.evaluatePosition(grid, availableBlocks, usedBlocks)
        
        # Store the result in the transposition table
        self.transpositionTable.store(grid, availableBlocks, usedBlocks, depth, bestScore, bestBlockIndex, bestPosition)
        
        return MinimaxResult(bestScore, bestBlockIndex, bestPosition)
    
    def canPlaceBlockAtPosition(self, grid: List[List[Optional[str]]], shape: List[List[int]], 
                               row: int, col: int) -> bool:
        """
        Check if a block can be placed at the given position.
        
        Args:
            grid: The game grid
            shape: The shape of the block
            row: Grid row
            col: Grid column
            
        Returns:
            True if the block can be placed, False otherwise
        """
        # Check if the block is within grid bounds
        if (row + len(shape) > GRID_SIZE or 
            col + len(shape[0]) > GRID_SIZE):
            return False
        
        # Check if all cells required by the block are empty
        for r in range(len(shape)):
            for c in range(len(shape[0])):
                if shape[r][c] and grid[row + r][col + c] is not None:
                    return False
        
        return True
    
    def placeBlock(self, grid: List[List[Optional[str]]], shape: List[List[int]], 
                  colorName: str, row: int, col: int) -> None:
        """
        Place a block on the grid.
        
        Args:
            grid: The game grid
            shape: The shape of the block
            colorName: The color name of the block
            row: Grid row
            col: Grid column
        """
        for r in range(len(shape)):
            for c in range(len(shape[0])):
                if shape[r][c]:
                    grid[row + r][col + c] = colorName
    
    def clearCompletedLines(self, grid: List[List[Optional[str]]]) -> int:
        """
        Clear completed rows and columns from the grid.
        
        Args:
            grid: The game grid
            
        Returns:
            Number of lines cleared
        """
        linesCleared = 0
        
        # Check rows
        for row in range(GRID_SIZE):
            if all(grid[row][col] is not None for col in range(GRID_SIZE)):
                # Clear the row
                for col in range(GRID_SIZE):
                    grid[row][col] = None
                linesCleared += 1
        
        # Check columns
        for col in range(GRID_SIZE):
            if all(grid[row][col] is not None for row in range(GRID_SIZE)):
                # Clear the column
                for row in range(GRID_SIZE):
                    grid[row][col] = None
                linesCleared += 1
        
        return linesCleared
    
    def isGameOver(self, availableBlocks: List[Dict[str, Any]], usedBlocks: List[bool], 
                  grid: List[List[Optional[str]]]) -> bool:
        """
        Check if the game is over (no valid moves left).
        
        Args:
            availableBlocks: List of available blocks
            usedBlocks: List indicating which blocks have been used
            grid: The game grid
            
        Returns:
            True if the game is over, False otherwise
        """
        # Check each available block
        for blockIndex in range(len(availableBlocks)):
            # Skip used blocks
            if usedBlocks[blockIndex]:
                continue
                
            block = availableBlocks[blockIndex]
            shape = block["shape"]
            
            # Check if this block can be placed anywhere on the grid
            for row in range(GRID_SIZE - len(shape) + 1):
                for col in range(GRID_SIZE - len(shape[0]) + 1):
                    if self.canPlaceBlockAtPosition(grid, shape, row, col):
                        return False  # Found a valid move, game can continue
        
        return True  # No valid moves found, game is over
    
    def evaluatePosition(self, grid: List[List[Optional[str]]], availableBlocks: List[Dict[str, Any]], 
                        usedBlocks: List[bool]) -> int:
        """
        Evaluate the current game position.
        
        Args:
            grid: The game grid
            availableBlocks: List of available blocks
            usedBlocks: List indicating which blocks have been used
            
        Returns:
            A score representing the quality of the position
        """
        # Count empty cells - fewer empty cells is better
        emptyCells = sum(1 for row in grid for cell in row if cell is None)
        
        # Count potential lines that are almost complete
        almostCompleteLines = 0
        
        # Check rows
        for row in range(GRID_SIZE):
            filledCells = sum(1 for col in range(GRID_SIZE) if grid[row][col] is not None)
            if GRID_SIZE - 2 <= filledCells < GRID_SIZE:
                almostCompleteLines += 1
        
        # Check columns
        for col in range(GRID_SIZE):
            filledCells = sum(1 for row in range(GRID_SIZE) if grid[row][col] is not None)
            if GRID_SIZE - 2 <= filledCells < GRID_SIZE:
                almostCompleteLines += 1
        
        # Count available moves
        availableMoves = 0
        for blockIndex in range(len(availableBlocks)):
            if not usedBlocks[blockIndex]:
                block = availableBlocks[blockIndex]
                shape = block["shape"]
                
                for row in range(GRID_SIZE - len(shape) + 1):
                    for col in range(GRID_SIZE - len(shape[0]) + 1):
                        if self.canPlaceBlockAtPosition(grid, shape, row, col):
                            availableMoves += 1
        
        # Calculate score: prefer fewer empty cells, more almost complete lines, and more available moves
        score = (
            -emptyCells * 5 +  # Fewer empty cells is better
            almostCompleteLines * 50 +  # Almost complete lines are valuable
            availableMoves * 10  # Having more available moves is good
        )
        
        return score
    
    def printTopMoves(self) -> None:
        """
        Print the top scoring moves found during the search.
        """
        if not self.topMoves:
            print("\nNo valid moves found during search.")
            return
            
        # Sort moves by score in descending order
        sortedMoves = sorted(self.topMoves, key=lambda x: x[2], reverse=True)
        
        print("\nTop 5 moves considered:")
        print("Rank | Block | Position | Score")
        print("-" * 40)
        
        for i, (blockIndex, position, score) in enumerate(sortedMoves[:5]):
            print(f"{i+1:4d} | {blockIndex:5d} | ({position[0]}, {position[1]}) | {score:5d}")
        
        print("-" * 40)
        
        # Visualize the search tree for the best move
        if sortedMoves:
            bestBlockIndex, bestPosition, _ = sortedMoves[0]
            print(f"\nSearch tree visualization for best move (Block {bestBlockIndex} at {bestPosition}):")
            print("Depth 0: Root")
            print("  |")
            print(f"  └─ Depth 1: Place Block {bestBlockIndex} at {bestPosition}")
            print("      |")
            print("      └─ Depth 2: Evaluate resulting positions")
            print("          |")
            print("          ├─ Check for completed lines")
            print("          |")
            print("          └─ Recursive search for next best move")
            print("              |")
            print("              └─ ... (search continues to depth limit)")

    def getSearchProgress(self) -> Dict[str, Any]:
        """
        Get the current search progress information.
        
        Returns:
            A dictionary containing search progress information
        """
        progress = 0.0
        if hasattr(self, 'totalMoves') and self.totalMoves > 0:
            progress = min(1.0, self.currentMoveIndex / self.totalMoves)
            
        return {
            'nodesExplored': self.nodesExplored,
            'elapsedTime': time.time() - self.startTime if hasattr(self, 'startTime') else 0,
            'progress': progress,
            'totalMoves': self.totalMoves if hasattr(self, 'totalMoves') else 0,
            'currentMoveIndex': self.currentMoveIndex if hasattr(self, 'currentMoveIndex') else 0,
            'transpositionTableHits': self.transpositionTable.hits if hasattr(self, 'transpositionTable') else 0,
            'bestMove': max(self.topMoves, key=lambda x: x[2]) if hasattr(self, 'topMoves') and self.topMoves else None
        } 

    def cancelSearch(self) -> None:
        """
        Cancel an ongoing search.
        """
        if self.isSearching:
            print("Cancelling AI search...")
            self.shouldCancelSearch = True
            
            # Wait for the search thread to finish (with a timeout)
            if self.searchThread and self.searchThread.is_alive():
                self.searchThread.join(timeout=0.5)
                
            self.isSearching = False
            print("AI search cancelled")
        else:
            print("No AI search in progress to cancel") 