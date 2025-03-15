# Block Blast

A Tetris-like block puzzle game implemented in Pygame.

## Description

Block Blast is a puzzle game where you place different shaped blocks on a grid. The goal is to create complete rows or columns to clear them and score points. The game continues until no more blocks can be placed on the grid.

## Features

- 10x10 game grid
- 24 different block shapes with various configurations
- 10 different block colors
- High score tracking and persistence
- Score display with crown icon
- Game over screen with replay option
- Block preview with placement validation
- Row and column clearing mechanics
- 3D block rendering with light and shadow effects

## Installation

1. Make sure you have Python 3.6+ installed
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## How to Run

```bash
python blockBlast.py
```

## How to Play

1. Click on one of the three available block shapes at the bottom of the screen to select it
2. Move your mouse over the grid to see a preview of where the block will be placed
   - Green preview: Block can be placed
   - Red X overlay: Block cannot be placed at that position
3. Click on the grid to place the block
4. If a row or column is completely filled, it will be cleared and you'll earn extra points
5. When all three available blocks are used, new blocks will be generated
6. The game ends when no more blocks can be placed on the grid
7. Try to get the highest score possible!

## Scoring

- 10 points for each block placed
- 100 points for each row or column cleared
- High scores are saved between game sessions

## Controls

- **Mouse Click**: Select blocks and place them on the grid
- **Mouse Movement**: Preview block placement
- **Click Play Again**: Restart the game after game over

## Game Elements

- **Top of screen**: Current score and high score with crown icon
- **Center**: 10x10 game grid where blocks are placed
- **Bottom**: Three available blocks to choose from
- **Game Over screen**: Final score, high score, and replay button

## Technical Details

- Written in Python using Pygame
- Object-oriented design with the BlockBlastGame class
- High scores saved to a local file (highscore.txt)
- Semi-transparent block preview with placement validation
- Responsive UI with visual feedback 