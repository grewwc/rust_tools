const test = require('node:test');
const assert = require('node:assert/strict');
const { ppeRequest, completionRequest, completionOptions, scopedRequest, ppeEnvironmentNotice } = require('../src/features/ai_node_web_aeolus_card/ppe');

function task() {
  return {
    task_id: 'task', state: 'review', deployment_workers: [],
    rounds: [{ round_no: 1, status: 'review', commit_sha: 'a'.repeat(40) }],
    ppe_capability: { enabled: true, profile_revision: 'c'.repeat(64) },
  };
}

test('PPE creates from the latest successful round without Devbox and binds retries to that round', () => {
  const snapshot = task();
  assert.deepEqual(ppeRequest(snapshot).body, {
    round_no: 1, commit_sha: 'a'.repeat(40), profile_revision: 'c'.repeat(64),
  });
  snapshot.ppe_deployment = {
    deployment_id: 'one', round_no: 1, commit_sha: 'a'.repeat(40), can_deploy: false, can_retry: true,
  };
  assert.equal(ppeRequest(snapshot), null);
  assert.equal(ppeRequest(snapshot, true).path, '/ppe-deployments/one/retry');
  snapshot.rounds.push({ round_no: 2, status: 'review', commit_sha: 'b'.repeat(40) });
  assert.equal(ppeRequest(snapshot, true), null);
  snapshot.ppe_deployment.can_deploy = true;
  assert.equal(ppeRequest(snapshot).body.commit_sha, 'b'.repeat(40));
  snapshot.evaluation = { code_context: { repository: 'dp/VizQueryService' } };
  assert.equal(ppeRequest(snapshot).body.commit_sha, 'b'.repeat(40));
  snapshot.evaluation = null;
  snapshot.rounds[1].status = 'blocked';
  assert.equal(ppeRequest(snapshot), null);
  snapshot.state = 'completed';
  assert.equal(ppeRequest(snapshot), null);
});

test('VQS PPE uses server capability for deployment and retry eligibility', () => {
  const snapshot = task();
  snapshot.evaluation = { code_context: { repository: 'dp/VizQueryService' } };
  assert.equal(ppeRequest(snapshot).path, '/ppe-deployments');
  snapshot.ppe_deployment = {
    deployment_id: 'vqs-failed', round_no: 1, commit_sha: 'a'.repeat(40), can_deploy: true, can_retry: true,
  };
  assert.equal(ppeRequest(snapshot, true).path, '/ppe-deployments/vqs-failed/retry');
  snapshot.ppe_capability.enabled = false;
  assert.equal(ppeRequest(snapshot), null);
  assert.equal(ppeRequest(snapshot, true).path, '/ppe-deployments/vqs-failed/retry');
});

test('a lost response reuses the request ID only for the same task, operation, round and template', () => {
  let next = 0;
  const newId = () => String(++next);
  const snapshot = task();
  const original = scopedRequest(null, ppeRequest(snapshot).key, newId);
  assert.equal(scopedRequest(original, ppeRequest(snapshot).key, newId), original);
  snapshot.ppe_capability.profile_revision = 'd'.repeat(64);
  assert.notEqual(scopedRequest(original, ppeRequest(snapshot).key, newId).id, original.id);
  snapshot.rounds.push({ round_no: 2, status: 'review', commit_sha: 'b'.repeat(40) });
  assert.notEqual(scopedRequest(original, ppeRequest(snapshot).key, newId).id, original.id);
  snapshot.ppe_deployment = {
    deployment_id: 'failed', round_no: 2, commit_sha: 'b'.repeat(40), can_deploy: true, can_retry: true,
  };
  const retry = scopedRequest(original, ppeRequest(snapshot, true).key, newId);
  assert.notEqual(retry.id, original.id);
  snapshot.ppe_deployment.deployment_id = 'retry-failed';
  assert.notEqual(scopedRequest(retry, ppeRequest(snapshot, true).key, newId).id, retry.id);
  snapshot.task_id = 'another';
  assert.notEqual(scopedRequest(original, ppeRequest(snapshot).key, newId).id, original.id);
});

test('successful current-round PPE hides deployment even with an older permissive snapshot', () => {
  const snapshot = task();
  snapshot.ppe_deployment = {
    deployment_id: 'ready', status: 'deployed', round_no: 1, commit_sha: 'a'.repeat(40),
    can_deploy: true, can_retry: false, released: false,
  };
  assert.equal(ppeRequest(snapshot), null);
  assert.equal(ppeRequest(snapshot, true), null);
  snapshot.rounds.push({ round_no: 2, status: 'review', commit_sha: 'b'.repeat(40) });
  assert.equal(ppeRequest(snapshot).body.commit_sha, 'b'.repeat(40));
  snapshot.rounds.pop();
  snapshot.ppe_deployment.released = true;
  assert.equal(ppeRequest(snapshot).body.commit_sha, 'a'.repeat(40));
  snapshot.ppe_deployment.released = false;
  snapshot.ppe_deployment.status = 'failed';
  snapshot.ppe_deployment.can_retry = true;
  assert.equal(ppeRequest(snapshot, true).path, '/ppe-deployments/ready/retry');
});

test('verification binds both available environments and never falls back to code-only', () => {
  const snapshot = task();
  snapshot.ppe_deployment = { deployment_id: 'ppe', can_verify: true, released: false };
  snapshot.deployment = { deployment_id: 'devbox', current: true, resource_owned: true };
  assert.deepEqual(completionRequest(snapshot, 'all_verified'), {
    mode: 'verified', verification_targets: [
      { target: 'devbox', deployment_id: 'devbox' }, { target: 'ppe', deployment_id: 'ppe' },
    ], release_resources: true,
  });
  assert.equal(completionRequest(snapshot, ''), null);
  snapshot.ppe_deployment.can_verify = false;
  assert.deepEqual(completionRequest(snapshot, 'all_verified').verification_targets,
    [{ target: 'devbox', deployment_id: 'devbox' }]);
  snapshot.deployment.current = false;
  assert.equal(completionRequest(snapshot, 'all_verified'), null);
  assert.equal(completionRequest(snapshot, 'skip').mode, 'code_only');
  snapshot.state = 'coding_blocked';
  assert.equal(completionRequest(snapshot, 'skip'), null);
});

test('completion offers only verification and skip without target-specific choices or a placeholder', () => {
  const snapshot = task();
  assert.deepEqual(completionOptions(snapshot).map(o => o.value), ['skip']);
  for (const mode of ['', undefined, 'ppe', 'devbox', 'verified', 'ppe_verified', 'invalid']) {
    assert.equal(completionRequest(snapshot, mode), null);
  }
  assert.equal(completionRequest(snapshot, 'skip').mode, 'code_only');
  snapshot.ppe_deployment = { deployment_id: 'ppe', can_verify: true };
  snapshot.deployment = { deployment_id: 'box', current: true };
  assert.deepEqual(completionOptions(snapshot).map(o => o.label), ['验证通过', '跳过验证']);
  snapshot.deployment.current = false;
  assert.deepEqual(completionOptions(snapshot).map(o => o.value), ['all_verified', 'skip']);
  snapshot.ppe_deployment.can_verify = false;
  assert.deepEqual(completionOptions(snapshot).map(o => o.value), ['skip']);
});

test('updating retains the previous version notice until deployment succeeds, including failure and coding feedback', () => {
  const deployment = {
    deployment_id: 'new', stage: 'image_wait', image_version: '674', resource_state: 'running',
    current_deployment: { deployment_id: 'old', commit_sha: 'a'.repeat(40), image_version: '672' },
  };
  const notice = ppeEnvironmentNotice(deployment);
  assert.equal(notice.imageVersion, '672');
  assert.equal(notice.commitSha, 'a'.repeat(40));
  assert.match(notice.title, /本次更新尚未完成/);
  assert.match(notice.detail, /上一版已就绪不代表/);
  deployment.stage = 'failed';
  assert.deepEqual(ppeEnvironmentNotice(deployment), notice);
  for (const stage of ['deploy', 'ticket_wait', 'observe', 'unknown']) {
    deployment.stage = stage;
    assert.match(ppeEnvironmentNotice(deployment).detail, /集群可能处于不同版本/);
  }
  deployment.current_deployment.deployment_id = 'new';
  deployment.stage = 'entry_pending';
  assert.equal(ppeEnvironmentNotice(deployment), null);
  deployment.superseded = true;
  assert.match(ppeEnvironmentNotice(deployment).title, /本轮代码尚未部署/);
  deployment.release_requested = true;
  assert.equal(ppeEnvironmentNotice(deployment), null);
  deployment.release_requested = false;
  deployment.released = true;
  assert.equal(ppeEnvironmentNotice(deployment), null);
  assert.equal(ppeEnvironmentNotice({ deployment_id: 'first', stage: 'image_wait' }), null);
  assert.equal(ppeEnvironmentNotice(null), null);
});
