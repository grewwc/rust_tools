function ppeRequest(task, retry = false, selected) {
  const round = task?.rounds?.at(-1);
  const capability = task?.ppe_capability;
  const deployment = task?.ppe_deployment;
  if (task?.state !== 'review' || (!retry && !capability?.enabled) || round?.status !== 'review'
      || !/^[0-9a-f]{40}$/.test(round?.commit_sha || '')) return null;
  if (retry ? !deployment?.can_retry
      || deployment.round_no !== round.round_no || deployment.commit_sha !== round.commit_sha
    : deployment && !deployment.can_deploy) return null;
  if (!retry && deployment?.status === 'deployed' && !deployment.released
      && deployment.round_no === round.round_no && deployment.commit_sha === round.commit_sha) return null;
  const path = retry ? `/ppe-deployments/${deployment.deployment_id}/retry` : '/ppe-deployments';
  const profileId = selected ?? capability?.recommended_runtime_profile_id;
  const profile = capability?.runtime_profiles?.[profileId];
  if (!retry && capability?.runtime_profiles && (!profile || profile.compatible === false)) return null;
  const body = {
    round_no: round.round_no,
    commit_sha: round.commit_sha,
    profile_revision: retry ? deployment.profile_revision : profile?.profile_revision ?? capability.profile_revision,
    ...(!retry && profile ? { runtime_profile_id: profileId } : {}),
  };
  return { path, body, key: JSON.stringify([task.task_id, path, body]) };
}

function verificationTargets(task) {
  return [
    ...(task?.deployment?.current ? [{ target: 'devbox', deployment_id: task.deployment.deployment_id }] : []),
    ...(task?.ppe_deployment?.can_verify ? [{ target: 'ppe', deployment_id: task.ppe_deployment.deployment_id }] : []),
  ];
}

function completionOptions(task) {
  return [
    ...(verificationTargets(task).length ? [{ value: 'all_verified', label: '验证通过' }] : []),
    { value: 'skip', label: '跳过验证' },
  ];
}

function completionRequest(task, mode) {
  if (task?.state !== 'review') return null;
  if (!completionOptions(task).some(option => option.value === mode)) return null;
  return {
    release_resources: !!task.deployment?.resource_owned || !!task.ppe_deployment && !task.ppe_deployment.released,
    mode: mode === 'all_verified' ? 'verified' : 'code_only',
    ...(mode === 'all_verified' ? { verification_targets: verificationTargets(task) } : {}),
  };
}

function scopedRequest(previous, key, newId) {
  return previous?.key === key ? previous : { key, id: newId() };
}

function ppeEnvironmentNotice(deployment) {
  if (!deployment || deployment.released || deployment.release_requested) return null;
  const current = deployment.current_deployment;
  if (!current || current.deployment_id === deployment.deployment_id && !deployment.superseded) return null;
  const switching = ['deploy', 'ticket_wait', 'observe', 'unknown'].includes(deployment.stage)
    || deployment.resource_state === 'deploying';
  return {
    title: deployment.superseded ? '保留上一轮 PPE 环境，本轮代码尚未部署'
      : '正在更新同一 PPE 环境，本次更新尚未完成',
    detail: switching ? '版本切换中，集群可能处于不同版本；请等待本次部署完成后再验证。'
      : '保留上一版环境与访问方式；上一版已就绪不代表本次更新已生效。',
    imageVersion: current.image_version || '未知',
    commitSha: current.commit_sha,
  };
}

module.exports = { ppeRequest, completionRequest, completionOptions, scopedRequest, ppeEnvironmentNotice };
