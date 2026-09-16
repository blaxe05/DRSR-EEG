"""Small helpers for script-style command-line wrappers."""

import shlex


def normalize_extra_args(args):
    """Return extra parser args as argv tokens.

    PowerShell background jobs can accidentally flatten a string array into one
    whitespace-delimited argument. The training wrappers append these values to
    an argparse argv list, so a flattened string must be expanded back into
    tokens before parsing.
    """
    items = list(args or [])
    if len(items) == 1 and isinstance(items[0], str) and " " in items[0].strip():
        return shlex.split(items[0])
    return items
