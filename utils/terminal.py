from __future__ import annotations

import os
import re


TAG_PATTERN = re.compile(r"^\[([^\]]+)\]")
TAG_COLOR = "\033[1;36m"
RESET_COLOR = "\033[0m"


def colorize_tag(text: str) -> str:
    if os.getenv("NO_COLOR"):
        return text
    return TAG_PATTERN.sub(lambda match: f"{TAG_COLOR}{match.group(0)}{RESET_COLOR}", text, count=1)


def terminal_print(message: str, *, flush: bool = False) -> None:
    print(colorize_tag(message), flush=flush)
