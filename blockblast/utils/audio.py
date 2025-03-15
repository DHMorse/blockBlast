"""
Audio management for BlockBlast.

Handles playing random music from the user's music folder.
"""
import os
import random
import pygame
from typing import List, Optional, Callable
from pathlib import Path

class MusicPlayer:
    """
    Music player that plays random songs from a specified directory.
    Automatically plays the next song when the current one finishes.
    """
    
    def __init__(self, musicDir: Optional[str] = None) -> None:
        """
        Initialize the music player.
        
        Args:
            musicDir: Directory containing music files. If None, will use default locations.
        """
        self.musicDir: str = self._findMusicDirectory(musicDir)
        self.musicFiles: List[str] = []
        self.currentSong: Optional[str] = None
        self.isPlaying: bool = False
        
        # Set up the end event handler
        self.endEventType = pygame.USEREVENT + 1
        pygame.mixer.music.set_endevent(self.endEventType)
        
        # Load available music files
        self._loadMusicFiles()
    
    def _findMusicDirectory(self, userSpecifiedDir: Optional[str] = None) -> str:
        """
        Find the user's music directory.
        
        Args:
            userSpecifiedDir: User-specified music directory
            
        Returns:
            Path to the music directory
        """
        if userSpecifiedDir and os.path.isdir(userSpecifiedDir):
            return userSpecifiedDir
        
        # Common music directory locations
        homeDir = str(Path.home())
        possibleDirs = [
            os.path.join(homeDir, "Music"),       # Linux/macOS
            os.path.join(homeDir, "music"),       # Alternative Linux
            os.path.join(homeDir, "Musik"),       # German
            os.path.join(homeDir, "Musique"),     # French
            os.path.join(homeDir, "Música"),      # Spanish/Portuguese
            os.path.join(homeDir, "Musica"),      # Alternative Spanish/Italian
            os.path.join(homeDir, "Musiken"),     # Swedish
            os.path.join(homeDir, "Muziek"),      # Dutch
            os.path.join(homeDir, "Muzyka"),      # Polish
            os.path.join(homeDir, "Müzik"),       # Turkish
            os.path.join(homeDir, "Музыка"),      # Russian
            os.path.join(homeDir, "音楽"),         # Japanese
            os.path.join(homeDir, "音乐"),         # Chinese
            os.path.join(homeDir, "음악"),         # Korean
        ]
        
        # Windows Music folder
        if os.name == 'nt':
            possibleDirs.append(os.path.join(homeDir, "Music"))
            possibleDirs.append(os.path.join(homeDir, "My Music"))
            
            # Try to get the Windows special folder path
            try:
                import winreg
                with winreg.OpenKey(winreg.HKEY_CURRENT_USER, 
                                   r"Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders") as key:
                    musicDir = winreg.QueryValueEx(key, "My Music")[0]
                    possibleDirs.append(musicDir)
            except (ImportError, OSError):
                pass
        
        # Check if any of the possible directories exist
        for directory in possibleDirs:
            if os.path.isdir(directory):
                return directory
        
        # If no music directory is found, use the current directory
        return os.getcwd()
    
    def _loadMusicFiles(self) -> None:
        """
        Load all music files from the music directory.
        """
        self.musicFiles = []
        supportedExtensions = ['.mp3', '.ogg', '.wav', '.flac', '.m4a']
        
        # Walk through the music directory and its subdirectories
        for root, _, files in os.walk(self.musicDir):
            for file in files:
                if any(file.lower().endswith(ext) for ext in supportedExtensions):
                    self.musicFiles.append(os.path.join(root, file))
        
        print(f"Found {len(self.musicFiles)} music files in {self.musicDir}")
    
    def playRandomSong(self) -> None:
        """
        Play a random song from the music directory.
        """
        if not self.musicFiles:
            print("No music files found.")
            return
        
        # Choose a random song that's different from the current one
        availableSongs = [song for song in self.musicFiles if song != self.currentSong]
        if not availableSongs and self.musicFiles:
            # If there's only one song, use it
            availableSongs = self.musicFiles
        
        if availableSongs:
            self.currentSong = random.choice(availableSongs)
            try:
                pygame.mixer.music.load(self.currentSong)
                pygame.mixer.music.play()
                self.isPlaying = True
                songName = os.path.basename(self.currentSong)
                print(f"Now playing: {songName}")
            except pygame.error as e:
                print(f"Error playing music: {e}")
                self.currentSong = None
                # Try another song
                self.playRandomSong()
    
    def handleEvent(self, event: pygame.event.Event) -> None:
        """
        Handle pygame events related to music playback.
        
        Args:
            event: The pygame event to handle
        """
        if event.type == self.endEventType:
            # Current song has ended, play another one
            self.playRandomSong()
    
    def start(self) -> None:
        """
        Start playing music.
        """
        if not self.isPlaying:
            self.playRandomSong()
    
    def stop(self) -> None:
        """
        Stop playing music.
        """
        if self.isPlaying:
            pygame.mixer.music.stop()
            self.isPlaying = False
            self.currentSong = None
    
    def pause(self) -> None:
        """
        Pause the currently playing music.
        """
        if self.isPlaying:
            pygame.mixer.music.pause()
            self.isPlaying = False
    
    def unpause(self) -> None:
        """
        Unpause the currently paused music.
        """
        if not self.isPlaying and self.currentSong:
            pygame.mixer.music.unpause()
            self.isPlaying = True
    
    def setVolume(self, volume: float) -> None:
        """
        Set the music volume.
        
        Args:
            volume: Volume level (0.0 to 1.0)
        """
        pygame.mixer.music.set_volume(max(0.0, min(1.0, volume))) 