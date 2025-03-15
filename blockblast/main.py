"""
Main entry point for BlockBlast game.
"""
import pygame
import sys

from blockblast.game.game import BlockBlastGame

def main() -> None:
    """
    Initialize pygame and start the game.
    """
    # Initialize pygame
    pygame.init()
    
    # Initialize the mixer with good audio quality
    pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=2048)
    
    # Create and run the game
    game = BlockBlastGame()
    game.run()
    
    # Clean up
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main() 