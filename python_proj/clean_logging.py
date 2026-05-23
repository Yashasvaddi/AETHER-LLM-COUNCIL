"""
Clean logging utility for Project AETHER
Logs to file, shows minimal clean TUI in terminal
"""

import logging
import os
from datetime import datetime
import sys

class CleanTUIHandler(logging.Handler):
    """Custom handler that shows only important messages in terminal"""
    
    def __init__(self, show_progress=True):
        super().__init__()
        self.show_progress = show_progress
        self.last_message = ""
    
    def emit(self, record):
        """Only show specific log levels and messages"""
        msg = record.getMessage()
        level = record.levelname
        
        # Show only specific important messages
        show_patterns = [
            "✓",  # Success
            "✅", # Complete
            "⚠️", # Warning
            "❌", # Error
            "🎯", # Important
            "⚔️", # Major events
            "📊", # Summary
            "💾", # Save
            "🔀", # Routing
            "⏳", # Progress
            "Processing factor",
            "Analysis complete",
        ]
        
        # Don't show debug or info unless they have emoji
        if level == "DEBUG":
            return
        
        if level == "INFO":
            if not any(pattern in msg for pattern in show_patterns):
                return
        
        # Avoid duplicate messages
        if msg == self.last_message:
            return
        
        self.last_message = msg
        print(msg)

def setup_logging(name="aether", show_progress=True):
    """Setup logging with file + clean terminal output"""
    
    # Create logs directory
    os.makedirs("logs", exist_ok=True)
    
    # Filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join("logs", f"{name}_{timestamp}.log")
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # File handler (all messages)
    file_handler = logging.FileHandler(log_file, encoding='utf-8')  # UTF-8 for emoji support
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)
    
    # Console handler (clean TUI)
    console_handler = CleanTUIHandler(show_progress=show_progress)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(console_handler)
    
    return logger, log_file

class ProgressBar:
    """Simple ASCII progress bar for terminal"""
    
    def __init__(self, total, label=""):
        self.total = total
        self.current = 0
        self.label = label
        self.bar_length = 40
    
    def update(self, current=None, message=""):
        """Update progress"""
        if current is not None:
            self.current = current
        else:
            self.current += 1
        
        percent = (self.current / self.total) * 100
        filled = int(self.bar_length * self.current / self.total)
        bar = "█" * filled + "░" * (self.bar_length - filled)
        
        if message:
            msg = f"{self.label} [{bar}] {percent:.0f}% - {message}"
        else:
            msg = f"{self.label} [{bar}] {percent:.0f}% ({self.current}/{self.total})"
        
        # Overwrite line in terminal
        sys.stdout.write(f"\r{msg:<80}")
        sys.stdout.flush()
    
    def close(self):
        """Complete the progress bar"""
        self.update(self.total)
        print()  # New line
