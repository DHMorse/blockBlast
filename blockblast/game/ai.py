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
        # Cache for hash keys to avoid recomputing them
        self.hashCache: Dict[str, str] = {}
    
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
        # Create a cache key based on grid state and used blocks
        cacheKey = str(sum(1 for row in grid for cell in row if cell is not None)) + "_" + "".join(str(int(b)) for b in usedBlocks)
        
        if cacheKey in self.hashCache:
            return self.hashCache[cacheKey]
            
        # Convert grid to a string representation - optimize by only including non-empty cells
        gridStr = ""
        for r, row in enumerate(grid):
            for c, cell in enumerate(row):
                if cell is not None:
                    gridStr += f"{r},{c},{cell}|"
        
        # Convert available blocks to a string representation
        # We only care about the shape and which blocks are used
        blocksStr = ""
        for i, block in enumerate(availableBlocks):
            if not usedBlocks[i]:
                blocksStr += str(i)
        
        # Combine and hash
        stateStr = gridStr + blocksStr
        hashValue = hashlib.md5(stateStr.encode()).hexdigest()
        
        # Store in cache
        self.hashCache[cacheKey] = hashValue
        return hashValue
    
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
        self.maxDepth: int = 10  # Reduced maximum search depth for better performance
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
        
        # Precompute valid positions for each block shape to avoid redundant calculations
        self.validPositionsCache: Dict[str, List[Tuple[int, int]]] = {}
        
        # Cache for evaluation scores
        self.evaluationCache: Dict[str, int] = {}
        
        # For iterative deepening
        self.currentSearchDepth: int = 2
        self.bestMoveSoFar: Optional[Tuple[int, Tuple[int, int], int]] = None
    
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
            self.evaluationCache = {}
            self.bestMoveSoFar = None
            
            # Calculate total possible moves for progress tracking
            self.totalMoves = self.countPossibleMoves(grid, availableBlocks, usedBlocks)
            self.currentMoveIndex = 0
            
            # Use iterative deepening to get faster initial results
            for depth in range(2, self.maxDepth + 1, 2):
                if self.shouldCancelSearch:
                    break
                    
                self.currentSearchDepth = depth
                
                # Start the minimax search with current depth limit
                result = self.minimax(
                    grid,
                    availableBlocks,
                    usedBlocks,
                    0,
                    float('-inf'),
                    float('inf'),
                    depth
                )
                
                if result.blockIndex is not None and result.position is not None:
                    self.bestMoveSoFar = (result.blockIndex, result.position, result.score)
                    self.searchResult = (result.blockIndex, result.position)
                    
                    # Add to top moves
                    if (result.blockIndex, result.position, result.score) not in self.topMoves:
                        self.topMoves.append((result.blockIndex, result.position, result.score))
            
            # If no result was found or search was cancelled
            if self.bestMoveSoFar is None:
                self.searchResult = (None, None)
                
            # Call the callback with the result
            if self.onSearchComplete and not self.shouldCancelSearch:
                self.onSearchComplete(*self.searchResult)
        
        except Exception as e:
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
               usedBlocks: List[bool], depth: int, alpha: float, beta: float, 
               depthLimit: Optional[int] = None) -> MinimaxResult:
        """
        Minimax algorithm with alpha-beta pruning.
        
        Args:
            grid: The game grid
            availableBlocks: List of available blocks
            usedBlocks: List indicating which blocks have been used
            depth: Current search depth
            alpha: Alpha value for alpha-beta pruning
            beta: Beta value for alpha-beta pruning
            depthLimit: Optional depth limit for iterative deepening
            
        Returns:
            MinimaxResult containing the best score, block index, and position
        """
        # Use provided depth limit or default max depth
        maxDepthForSearch = depthLimit if depthLimit is not None else self.maxDepth
        
        # Check if search should be cancelled
        if self.shouldCancelSearch:
            return MinimaxResult(0, None, None)
            
        self.nodesExplored += 1
        
        # Track depth distribution
        self.depthDistribution[depth] = self.depthDistribution.get(depth, 0) + 1
        
        # Check transposition table
        ttResult = self.transpositionTable.lookup(grid, availableBlocks, usedBlocks, depth, alpha, beta)
        if ttResult is not None:
            return ttResult
        
        # Check if we've reached the maximum depth or if the game is over
        if depth >= maxDepthForSearch or self.isGameOver(availableBlocks, usedBlocks, grid):
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
            
            # Get valid positions for this shape (using cache if available)
            shapeKey = self._getShapeKey(shape)
            if shapeKey not in self.validPositionsCache:
                self.validPositionsCache[shapeKey] = self._findValidPositions(grid, shape)
            
            validPositions = self.validPositionsCache[shapeKey]
            
            # Try each valid position
            for row, col in validPositions:
                # Check if the block can be placed at this position
                if self.canPlaceBlockAtPosition(grid, shape, row, col):
                    # Update progress for top-level search
                    if depth == 0:
                        self.currentMoveIndex += 1
                    
                    # Make the move (avoid deep copy when possible)
                    newGrid = self._placeBlockAndGetNewGrid(grid, shape, block["colorName"], row, col)
                    newUsedBlocks = usedBlocks.copy()  # Shallow copy is sufficient
                    newUsedBlocks[blockIndex] = True
                    
                    # Check for completed lines and update the grid
                    linesCleared = self.clearCompletedLines(newGrid)
                    
                    # If all blocks have been used, reset used blocks
                    if all(newUsedBlocks):
                        newUsedBlocks = [False] * MAX_AVAILABLE_BLOCKS
                    
                    # Recursive minimax call
                    moveScore = self.minimax(
                        newGrid, 
                        availableBlocks,  # No need to copy, we don't modify this
                        newUsedBlocks, 
                        depth + 1, 
                        alpha, 
                        beta,
                        depthLimit
                    ).score
                    
                    # Add immediate score from placing the block and clearing lines
                    immediateScore = 10 + linesCleared * 100  # 10 for placing block, 100 per line cleared
                    moveScore += immediateScore
                    
                    # Update best move
                    if moveScore > bestScore:
                        bestScore = moveScore
                        bestBlockIndex = blockIndex
                        bestPosition = (row, col)
                        
                        # Track top moves at depth 0
                        if depth == 0:
                            self.topMoves.append((blockIndex, (row, col), moveScore))
                    
                    # Alpha-beta pruning
                    alpha = max(alpha, bestScore)
                    if beta <= alpha:
                        # Store the result in the transposition table before pruning
                        self.transpositionTable.store(
                            grid, availableBlocks, usedBlocks, depth, bestScore, bestBlockIndex, bestPosition
                        )
                        return MinimaxResult(bestScore, bestBlockIndex, bestPosition)
        
        # If no valid moves were found
        if bestBlockIndex is None:
            bestScore = self.evaluatePosition(grid, availableBlocks, usedBlocks)
        
        # Store the result in the transposition table
        self.transpositionTable.store(grid, availableBlocks, usedBlocks, depth, bestScore, bestBlockIndex, bestPosition)
        
        return MinimaxResult(bestScore, bestBlockIndex, bestPosition)
    
    def _getShapeKey(self, shape: List[List[int]]) -> str:
        """
        Generate a string key for a block shape.
        
        Args:
            shape: The block shape
            
        Returns:
            A string key representing the shape
        """
        return "".join("".join(str(cell) for cell in row) for row in shape)
    
    def _findValidPositions(self, grid: List[List[Optional[str]]], shape: List[List[int]]) -> List[Tuple[int, int]]:
        """
        Find all valid positions for a block shape on the grid.
        
        Args:
            grid: The game grid
            shape: The block shape
            
        Returns:
            List of valid (row, col) positions
        """
        validPositions = []
        for row in range(GRID_SIZE - len(shape) + 1):
            for col in range(GRID_SIZE - len(shape[0]) + 1):
                validPositions.append((row, col))
        return validPositions
    
    def _placeBlockAndGetNewGrid(self, grid: List[List[Optional[str]]], shape: List[List[int]], 
                               colorName: str, row: int, col: int) -> List[List[Optional[str]]]:
        """
        Place a block on a copy of the grid and return the new grid.
        
        Args:
            grid: The original game grid
            shape: The shape of the block
            colorName: The color name of the block
            row: Grid row
            col: Grid column
            
        Returns:
            A new grid with the block placed
        """
        # Create a new grid (more efficient than deep copy)
        newGrid = [row.copy() for row in grid]
        
        # Place the block
        for r in range(len(shape)):
            for c in range(len(shape[0])):
                if shape[r][c]:
                    newGrid[row + r][col + c] = colorName
        
        return newGrid
    
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
        # Check cache first
        cacheKey = self.transpositionTable.getHash(grid, availableBlocks, usedBlocks)
        if cacheKey in self.evaluationCache:
            return self.evaluationCache[cacheKey]
        
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
        
        # Count available moves (use cached valid positions)
        availableMoves = 0
        for blockIndex in range(len(availableBlocks)):
            if not usedBlocks[blockIndex]:
                block = availableBlocks[blockIndex]
                shape = block["shape"]
                
                # Get valid positions for this shape
                shapeKey = self._getShapeKey(shape)
                if shapeKey not in self.validPositionsCache:
                    self.validPositionsCache[shapeKey] = self._findValidPositions(grid, shape)
                
                for row, col in self.validPositionsCache[shapeKey]:
                    if self.canPlaceBlockAtPosition(grid, shape, row, col):
                        availableMoves += 1
        
        # Calculate score: prefer fewer empty cells, more almost complete lines, and more available moves
        score = (
            -emptyCells * 5 +  # Fewer empty cells is better
            almostCompleteLines * 50 +  # Almost complete lines are valuable
            availableMoves * 10  # Having more available moves is good
        )
        
        # Cache the result
        self.evaluationCache[cacheKey] = score
        
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
            'bestMove': max(self.topMoves, key=lambda x: x[2]) if hasattr(self, 'topMoves') and self.topMoves else None,
            'currentDepth': self.currentSearchDepth if hasattr(self, 'currentSearchDepth') else 0
        }

    def cancelSearch(self) -> None:
        """
        Cancel an ongoing search.
        """
        if self.isSearching:
            self.shouldCancelSearch = True
            
            # Wait for the search thread to finish (with a timeout)
            if self.searchThread and self.searchThread.is_alive():
                self.searchThread.join(timeout=0.5)
                
            self.isSearching = False 