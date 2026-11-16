const test = require('node:test');
const assert = require('node:assert/strict');
const { runtimeScope, runtimeSelection } = require('../src/features/ai_node_web_aeolus_card/runtime');
const { ppeRequest } = require('../src/features/ai_node_web_aeolus_card/ppe');

const profiles = {
  py27: { python_version: '2.7', profile_revision: 'a'.repeat(64), compatible: true },
  py3: { python_version: '3.10', profile_revision: 'b'.repeat(64), compatible: true },
};
const task = () => ({
  task_id: 'task', state: 'review',
  rounds: [{ round_no: 1, status: 'review', commit_sha: 'c'.repeat(40) }],
  ppe_capability: { enabled: true, runtime_profiles: profiles, recommended_runtime_profile_id: null },
});

test('unknown evidence leaves even a single profile unselected; manual choice survives polling', () => {
  assert.equal(runtimeSelection('', false, { py27: profiles.py27 }, null), '');
  assert.equal(runtimeSelection('py27', true, profiles, 'py3'), 'py27');
  assert.equal(runtimeSelection('py27', false, profiles, 'py3'), 'py3');
  assert.equal(runtimeSelection('py27', true, { py27: { compatible: false } }, null), '');
});

test('task, SHA, round, machine and profile revision changes invalidate the selection scope', () => {
  const snapshot = task();
  const original = runtimeScope(snapshot, 'machine', profiles);
  assert.equal(runtimeScope(snapshot, 'machine', profiles), original);
  assert.notEqual(runtimeScope(snapshot, 'other', profiles), original);
  for (const change of [
    s => s.task_id = 'another',
    s => s.rounds[0].commit_sha = 'd'.repeat(40),
    s => s.rounds[0].round_no = 2,
  ]) {
    const next = task();
    change(next);
    assert.notEqual(runtimeScope(next, 'machine', profiles), original);
  }
  assert.notEqual(runtimeScope(snapshot, 'machine', {
    ...profiles, py27: { ...profiles.py27, profile_revision: 'e'.repeat(64) },
  }), original);
});

test('PPE sends the explicit profile and revision; selection changes idempotency key', () => {
  const snapshot = task();
  assert.equal(ppeRequest(snapshot), null);
  assert.equal(ppeRequest(snapshot, false, ''), null);
  const py27 = ppeRequest(snapshot, false, 'py27');
  assert.equal(py27.body.runtime_profile_id, 'py27');
  assert.equal(py27.body.profile_revision, profiles.py27.profile_revision);
  assert.notEqual(py27.key, ppeRequest(snapshot, false, 'py3').key);
  snapshot.ppe_capability.runtime_profiles = { py27: { ...profiles.py27, compatible: false } };
  assert.equal(ppeRequest(snapshot, false, 'py27'), null);
});

test('retry binds the accepted template even when recommendations change', () => {
  const snapshot = task();
  snapshot.ppe_deployment = {
    deployment_id: 'failed', round_no: 1, commit_sha: 'c'.repeat(40),
    can_retry: true, profile_revision: profiles.py27.profile_revision,
  };
  const retry = ppeRequest(snapshot, true, 'py27');
  assert.equal(retry.body.profile_revision, profiles.py27.profile_revision);
  snapshot.ppe_capability = { enabled: false };
  assert.equal(ppeRequest(snapshot, true, 'py3').key, retry.key);
});
