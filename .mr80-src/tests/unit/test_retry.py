import pytest

from conciliator.utils.retry import TransientError, retry_async


class _FlakyService:
    def __init__(self, fail_times: int):
        self.calls = 0
        self.fail_times = fail_times

    @retry_async("flaky", attempts=4, base_delay=0.0)
    async def call(self) -> str:
        self.calls += 1
        if self.calls <= self.fail_times:
            raise TransientError(f"fail #{self.calls}")
        return "ok"


@pytest.mark.asyncio
async def test_retry_succeeds_after_failures():
    svc = _FlakyService(fail_times=2)
    assert await svc.call() == "ok"
    assert svc.calls == 3


@pytest.mark.asyncio
async def test_retry_gives_up_after_max_attempts():
    svc = _FlakyService(fail_times=10)
    with pytest.raises(TransientError):
        await svc.call()
    assert svc.calls == 4


@pytest.mark.asyncio
async def test_retry_does_not_retry_non_transient():
    class _NonFlaky:
        calls = 0

        @retry_async("nonflaky", attempts=3, base_delay=0.0)
        async def call(self) -> None:
            type(self).calls += 1
            raise ValueError("not transient")

    with pytest.raises(ValueError):
        await _NonFlaky().call()
    assert _NonFlaky.calls == 1
