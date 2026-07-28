def normalize_keyword(s: str) -> str:
    """Lowercase and replace spaces/hyphens with underscores."""
    return s.lower().replace(" ", "_").replace("-", "_")


def split_pipe_separated(s: str | None, *, strip: bool = False) -> list[str]:
    if not s:
        return []
    parts = s.split("||")
    return [p.strip() for p in parts] if strip else parts
