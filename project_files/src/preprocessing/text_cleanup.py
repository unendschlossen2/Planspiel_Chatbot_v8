import re
from typing import Union

def normalize_markdown_whitespace(text: Union[str, None]) -> str:
    if not isinstance(text, str):
        return ""

    processed_text = text.strip()
    processed_text = re.sub(r'\n{2,}', '\n', processed_text)

    return processed_text