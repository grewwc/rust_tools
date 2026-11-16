import contextlib
import os
import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path

import git

from conciliator.utils.logging import get_logger

log = get_logger(__name__)


@contextlib.contextmanager
def clone_to_tempdir(
    repo_url: str,
    branch: str,
    token: str | None,
    base_dir: str,
) -> Iterator[Path]:
    """clone 仓库到临时目录,yield 路径,退出时清理。

    token 用于私有仓库:把 token 插入 URL https://oauth2:{token}@host/...
    """
    work_dir = Path(base_dir) / f"clone-{os.getpid()}-{tempfile.mkdtemp(prefix='')}"
    clone_url = _inject_token(repo_url, token) if token else repo_url

    try:
        log.info("cloning_repo", repo_url=repo_url, branch=branch, work_dir=str(work_dir))
        git.Repo.clone_from(clone_url, str(work_dir), branch=branch, depth=100)
        yield work_dir
    finally:
        try:
            shutil.rmtree(work_dir, ignore_errors=True)
        except Exception as e:
            log.warning("cleanup_failed", work_dir=str(work_dir), error=str(e))


def _inject_token(repo_url: str, token: str) -> str:
    """https://host.com/path -> https://oauth2:{token}@host.com/path"""
    if not repo_url.startswith("https://"):
        return repo_url
    return repo_url.replace("https://", f"https://oauth2:{token}@", 1)
