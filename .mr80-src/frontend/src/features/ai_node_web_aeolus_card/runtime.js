function runtimeScope(task, target, profiles) {
  const round = task?.rounds?.at(-1);
  return JSON.stringify([task?.task_id, round?.round_no, round?.commit_sha, target,
    Object.entries(profiles).map(([id, p]) => [id, p.profile_revision, p.compatible])]);
}

function runtimeSelection(previous, sameScope, profiles, recommended) {
  if (sameScope && profiles[previous]?.compatible !== false && previous in profiles) return previous;
  return recommended && profiles[recommended]?.compatible !== false && recommended in profiles ? recommended : '';
}

module.exports = { runtimeScope, runtimeSelection };
