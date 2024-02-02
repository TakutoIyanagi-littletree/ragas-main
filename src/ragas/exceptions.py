from __future__ import annotations

from ragas.testset.evolutions import Evolution


class RagasException(Exception):
    """
    Base exception class for ragas.
    """

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


class MaxRetriesExceeded(RagasException):
    """
    Exception raised when the maximum number of retries is exceeded.
    """

    def __init__(self, evolution: Evolution):
        msg = f"Max retries exceeded for evolution {evolution.__class__.__name__}."
        super().__init__(msg)
