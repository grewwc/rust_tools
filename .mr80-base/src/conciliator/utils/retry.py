from collections.abc import Awaitable, Callable
from functools import wraps
from typing import Any, ParamSpec

from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

P = ParamSpec("P")


class TransientError(Exception):
    """瞬时错误,可重试。"""


def retry_async(
    name: str,
    attempts: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 10.0,
    exceptions: tuple[type[BaseException], ...] = (TransientError,),
) -> Callable[[Callable[P, Awaitable[Any]]], Callable[P, Awaitable[Any]]]:
    def decorator(
        fn: Callable[P, Awaitable[Any]],
    ) -> Callable[P, Awaitable[Any]]:
        @wraps(fn)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> Any:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(attempts),
                wait=wait_exponential(multiplier=base_delay, max=max_delay),
                retry=retry_if_exception_type(exceptions),
                reraise=True,
            ):
                with attempt:
                    return await fn(*args, **kwargs)
            raise RuntimeError(f"unreachable for {name}")  # pragma: no cover
        return wrapper
    return decorator
