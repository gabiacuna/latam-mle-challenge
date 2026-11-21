"""Query-parameter parsing helpers for the pipeline control API."""

MES_ERROR = "mes must be a comma-separated list of integers"


def parse_csv_list(value: str | None) -> list[str] | None:
    """Split a comma-separated query value, stripping whitespace and dropping empty items.

    Returns ``None`` when the parameter is absent or contains no items.
    """
    if value is None:
        return None
    items = [item.strip() for item in value.split(",")]
    items = [item for item in items if item]
    return items or None


def parse_mes(value: str | None) -> list[int] | None:
    """Parse a comma-separated list of month numbers.

    Raises
    ------
    ValueError
        If any item is not an integer.
    """
    items = parse_csv_list(value)
    if items is None:
        return None
    try:
        return [int(item) for item in items]
    except ValueError as exc:
        raise ValueError(MES_ERROR) from exc
