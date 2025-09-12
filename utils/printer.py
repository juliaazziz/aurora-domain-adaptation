"""
@file printer.py
@brief Utility module for printing messages to the console with ANSI color codes.
"""

class Ansi:
    """
    ANSI escape codes for colored terminal output.
    """
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[0;33m'
    PURPLE = '\033[0;35m'
    CLEAR = '\033[0m'
    PREV_LINE = '\x1b[F'

class LogLevel:
    """
    Log levels for printing messages to the console.
    """
    INFO = 0
    OK = 1
    WARN = 2
    ERR = 3

def print_msg(level, *args):
    """
    Print a formatted message to the console with a specific log level.
    """
    prefix = {
        LogLevel.INFO: f"[{Ansi.PURPLE}INFO{Ansi.CLEAR}]: ",
        LogLevel.OK: f"[{Ansi.GREEN} OK {Ansi.CLEAR}]: ",
        LogLevel.WARN: f"[{Ansi.YELLOW}WARNING{Ansi.CLEAR}]: ",
        LogLevel.ERR: f"[{Ansi.RED}ERROR{Ansi.CLEAR}]: ",
    }.get(level, "")
    
    print(prefix, *args, flush=True)