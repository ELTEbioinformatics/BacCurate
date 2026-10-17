def normalize_keyword(s: str) -> str:
    """Lowercase and replace spaces/hyphens with underscores."""
    return s.lower().replace(" ", "_").replace("-", "_")


def split_pipe_separated(s: str | None) -> list[str]:
    if not s:
        return []
    return s.split("||")
