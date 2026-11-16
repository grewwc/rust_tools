def first_line(text: str) -> str:
    return text.splitlines()[0] if text else ""


def truncate(text: str, max_len: int = 500) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."
