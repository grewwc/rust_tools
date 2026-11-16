import os
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "run" / "prod_run.sh"


@pytest.mark.parametrize("workers, expected", [(None, "4"), ("2", "2"), ("8", "8")])
def test_production_launches_supervised_workers(tmp_path, workers, expected):
    result = _run_script(tmp_path, workers)
    assert result.returncode == 0, result.stderr
    args = (tmp_path / "python-args").read_text().splitlines()
    assert args[:3] == ["-m", "uvicorn", "conciliator.main:app"]
    assert args[args.index("--workers") + 1] == expected
    assert args[args.index("--limit-max-requests") + 1] == "10000"


@pytest.mark.parametrize("workers", ["1", "0", "-1", "invalid"])
def test_production_rejects_single_or_invalid_worker_count(tmp_path, workers):
    result = _run_script(tmp_path, workers)
    assert result.returncode != 0
    assert "WORKER_NUM must be an integer >= 2" in result.stderr
    assert not (tmp_path / "python-args").exists()


def _run_script(tmp_path, workers):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    python = bin_dir / "python"
    python.write_text('#!/bin/bash\nprintf "%s\\n" "$@" > "$APP_DIR/python-args"\n')
    python.chmod(0o755)
    # The launcher must never inspect or stop host processes during this test.
    lsof = bin_dir / "lsof"
    lsof.write_text("#!/bin/bash\nexit 0\n")
    lsof.chmod(0o755)
    env = {**os.environ, "APP_DIR": str(tmp_path), "PATH": f"{bin_dir}:{os.environ['PATH']}"}
    env.pop("WORKER_NUM", None)
    env.pop("MAX_REQUESTS", None)
    if workers is not None:
        env["WORKER_NUM"] = workers
    return subprocess.run(["bash", str(SCRIPT)], env=env, capture_output=True, text=True, timeout=5)
