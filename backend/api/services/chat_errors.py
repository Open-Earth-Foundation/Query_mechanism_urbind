"""Domain exceptions shared across chat services and route translation."""


class ChatBaseContextUnavailableError(ValueError):
    """Raised when the session's base run context can no longer be loaded."""


class ChatNoContextSourcesError(ValueError):
    """Raised when a chat session has no usable selected context sources."""


__all__ = ["ChatBaseContextUnavailableError", "ChatNoContextSourcesError"]
