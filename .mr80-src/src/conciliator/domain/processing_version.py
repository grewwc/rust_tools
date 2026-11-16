import hashlib


def build_active_key(repo_id: str, mr_iid: int, source_sha: str, target_sha: str) -> str:
    """Compact version key; terminal tasks release their key as before."""
    if not source_sha or not target_sha:
        raise ValueError("both commit SHAs are required for a processing version")
    return "v1:" + hashlib.sha256(f"{repo_id}:{mr_iid}:{source_sha}:{target_sha}".encode()).hexdigest()
