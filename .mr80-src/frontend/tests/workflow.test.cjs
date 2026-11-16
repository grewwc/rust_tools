const test = require('node:test');
const assert = require('node:assert/strict');

const {
  canSubmitFeedback,
  elapsedMs,
  formatDuration,
  historyLayout,
  isEvaluationExecutionFailure,
  isPollingState,
  roundDisplayStatus,
  roundDuration,
  roundTiming,
  roundPreview,
  roundTitle,
  safeCodeUrl,
  stageForState,
  taskLookupUrl,
  taskTiming,
  shouldShowLarkAuthorization,
  larkAuthorizationState,
  shouldShowInlineError,
  safeLarkDocumentUrl,
  documentPrecheckFailures,
} = require('../src/features/ai_node_web_aeolus_card/workflow');

test('adds the active phase and keeps retry waiting a subset of human waiting', () => {
  const timing = { automatic_ms: 60_000, human_wait_ms: 120_000, retry_wait_ms: 30_000,
    current_phase: 'automatic', phase_started_at: '2026-10-12T00:00:00Z' };
  const now = Date.parse('2026-10-12T00:00:30Z');
  assert.deepEqual(taskTiming(timing, now), { automatic: '1分30秒', humanWait: '2分0秒', retryWait: '30秒' });
  assert.deepEqual(taskTiming({ ...timing, current_phase: 'human_wait' }, now),
    { automatic: '1分0秒', humanWait: '2分30秒', retryWait: '30秒' });
  assert.deepEqual(taskTiming({ ...timing, current_phase: 'retry_wait' }, now),
    { automatic: '1分0秒', humanWait: '2分30秒', retryWait: '1分0秒' });
  assert.deepEqual(taskTiming({ ...timing, current_phase: 'completed' }, now),
    { automatic: '1分0秒', humanWait: '2分0秒', retryWait: '30秒' });
  for (const missing of [null, {}, { ...timing, phase_started_at: 'invalid' }]) {
    assert.deepEqual(taskTiming(missing, now), { automatic: '—', humanWait: '—', retryWait: '—' });
  }
});

test('formats elapsed time and rejects missing or invalid timestamps', () => {
  const start = '2026-10-10T08:00:00+08:00';
  const end = '2026-10-10T00:01:30Z';
  assert.equal(elapsedMs(start, end), 90_000);
  assert.equal(elapsedMs(start, null, Date.parse(end)), 90_000);
  for (const [value, expected] of [
    [0, '0秒'], [59_000, '59秒'], [90_000, '1分30秒'],
    [3_660_000, '1小时1分'], [90_060_000, '1天1小时1分'], [null, '—'],
  ]) assert.equal(formatDuration(value), expected);
  for (const [startValue, endValue] of [[null, end], ['bad', end], [start, 'bad'], [end, start]]) {
    assert.equal(elapsedMs(startValue, endValue), null);
  }
});

test('round timing includes human waiting and freezes independently of execution finish', () => {
  const round = { status: 'review', started_at: '2026-10-12T00:00:00Z', finished_at: '2026-10-12T00:01:00Z' };
  const timing = { automatic_ms: 60_000, human_wait_ms: 0, retry_wait_ms: 0,
    current_phase: 'human_wait', phase_started_at: round.finished_at,
    started_at: round.started_at, completed_at: null };
  const now = Date.parse('2026-10-12T00:02:00Z');
  assert.deepEqual(roundTiming(round, timing, now),
    { total: '2分0秒', automatic: '1分0秒', humanWait: '1分0秒' });
  assert.deepEqual(roundTiming(round, { ...timing, current_phase: 'automatic' }, now),
    { total: '2分0秒', automatic: '2分0秒', humanWait: '0秒' });
  const frozen = { ...timing, human_wait_ms: 60_000, current_phase: 'completed',
    completed_at: '2026-10-12T00:02:00Z' };
  assert.deepEqual(roundTiming(round, frozen, now + 3600_000),
    { total: '2分0秒', automatic: '1分0秒', humanWait: '1分0秒' });
  assert.deepEqual(roundTiming(round, undefined, now),
    { total: '1分0秒', automatic: '—', humanWait: '—' });
});

test('round durations advance only while running and freeze when finished', () => {
  const start = '2026-10-10T00:00:00Z';
  const end = '2026-10-10T00:01:00Z';
  const now = Date.parse('2026-10-10T00:02:00Z');
  for (const status of ['evaluating', 'coding', 'retrying']) {
    assert.equal(roundDuration({ status, started_at: start }, now), '2分0秒');
  }
  for (const status of ['review', 'confirmed', 'completed', 'error', 'revision_requested']) {
    assert.equal(roundDuration({ status, started_at: start, finished_at: end }, now), '1分0秒');
    assert.equal(roundDuration({ status, started_at: start }, now), '—');
  }
  assert.equal(roundDuration({ status: 'coding' }, now), '—');
});

test('distinguishes execution errors from a blocked requirement evaluation', () => {
  assert.equal(isEvaluationExecutionFailure(null), false);
  assert.equal(isEvaluationExecutionFailure({
    state: 'validation_failed', evaluation: null, last_error: 'invalid JSON',
  }), true);
  assert.equal(isEvaluationExecutionFailure({
    state: 'validation_failed', evaluation: { valid: false, summary: '无法读取需求文档' },
    evaluation_rounds: [{ status: 'error' }],
  }), true);
  assert.equal(isEvaluationExecutionFailure({
    state: 'evaluation_review', evaluation: { valid: false },
    evaluation_rounds: [{ status: 'error' }, { status: 'review' }],
  }), false);
  assert.equal(isEvaluationExecutionFailure({
    state: 'validation_failed', evaluation: { valid: false },
    evaluation_rounds: [{ status: 'review' }],
  }), false);
});

test('maps backend states to card stages', () => {
  assert.equal(stageForState('reading_document'), 'validating');
  assert.equal(stageForState('validation_failed'), 'validation_failed');
  assert.equal(stageForState('evaluation_review'), 'evaluation_review');
  assert.equal(stageForState('recoding'), 'coding');
  assert.equal(stageForState('review'), 'review');
  assert.equal(stageForState('completing'), 'completing');
  assert.equal(stageForState('completed'), 'completed');
  assert.equal(stageForState('failed'), 'error');
});

test('prioritizes and opens only the current phase history', () => {
  for (const state of ['received', 'reading_document', 'evaluating', 'validation_failed', 'evaluation_review']) {
    assert.deepEqual(historyLayout(state), { order: ['evaluation', 'coding'], expanded: 'evaluation' });
  }
  for (const state of ['coding', 'recoding', 'review', 'completing', 'failed']) {
    assert.deepEqual(historyLayout(state), { order: ['coding', 'evaluation'], expanded: 'coding' });
  }
  assert.deepEqual(historyLayout('completed'), { order: ['coding', 'evaluation'], expanded: null });
});

test('polls processing and review states to pick up actions from Feishu', () => {
  assert.equal(isPollingState('evaluating'), true);
  assert.equal(isPollingState('coding'), true);
  assert.equal(isPollingState('completing'), true);
  assert.equal(isPollingState('review'), true);
  assert.equal(isPollingState('validation_failed'), true);
  assert.equal(isPollingState('evaluation_review'), true);
  assert.equal(isPollingState('completed'), false);
  assert.equal(isPollingState('failed'), true);
  assert.equal(isPollingState('unknown'), false);
});

test('builds the task lookup from AI node context', () => {
  const url = new URL(taskLookupUrl('http://10.37.248.120:8000', {
    spaceId: 'space id',
    workItemId: 42,
    nodeId: 'node/1',
    workObjectId: 'story',
  }));

  assert.equal(url.pathname, '/api/aeolus/tasks/by-node');
  assert.equal(url.searchParams.get('space_id'), 'space id');
  assert.equal(url.searchParams.get('work_item_id'), '42');
  assert.equal(url.searchParams.get('node_id'), 'node/1');
  assert.equal(url.searchParams.get('work_object_id'), 'story');
});

test('requires non-blank feedback before recoding', () => {
  assert.equal(canSubmitFeedback('   '), false);
  assert.equal(canSubmitFeedback('fix tests'), true);
  assert.equal(canSubmitFeedback('x'.repeat(4_000)), true);
  assert.equal(canSubmitFeedback('x'.repeat(4_001)), false);
});

test('only opens HTTPS links from the Aeolus code host', () => {
  assert.equal(safeCodeUrl('https://code.byted.org/dp/aeolus/pulls/1'), 'https://code.byted.org/dp/aeolus/pulls/1');
  assert.equal(safeCodeUrl('http://code.byted.org/dp/aeolus'), '');
  assert.equal(safeCodeUrl('javascript:alert(1)'), '');
  assert.equal(safeCodeUrl('https://example.com/phishing'), '');
});

test('keeps session retries inside the same business round', () => {
  const round = {
    round_no: 2,
    status: 'coding',
    trigger_type: 'review_feedback',
    feedback: '补充边界测试',
    attempts: [
      { attempt_no: 1, status: 'failed' },
      { attempt_no: 2, status: 'running' },
    ],
  };

  assert.equal(roundDisplayStatus(round, false), 'retrying');
  assert.equal(roundTitle(round), '第 2 轮 · 反馈修改');
  assert.equal(roundPreview(round), '补充边界测试');
});

test('describes a reviewed historical round as requiring revision', () => {
  assert.equal(roundDisplayStatus({ status: 'review' }, true), 'revision_requested');
});


const { safeMaatUrl, shouldPollTask, canDeploy } = require('../src/features/ai_node_web_aeolus_card/workflow');

test('optional deployment and the per-instance MAAT entry', () => {
  const task = { state: 'review', rounds: [{ commit_sha: 'a'.repeat(40) }] };
  assert.equal(canDeploy(task, { online: true, deployment_counts: { total: 3 } }), true);
  assert.equal(canDeploy(task, { online: false }), false);
  assert.equal(canDeploy({ ...task, deployment: { status: 'running' } }, { online: true }), false);
  assert.equal(canDeploy({ ...task, deployment: { status: 'failed' } }, { online: true }), true);
  assert.equal(safeMaatUrl('https://maat.bytedance.net/project/poc'), 'https://maat.bytedance.net/project/poc');
  assert.equal(safeMaatUrl('https://xiaorui-aeolus-hua.maat.bytedance.net'), 'https://xiaorui-aeolus-hua.maat.bytedance.net/');
  for (const url of ['http://maat.bytedance.net/poc', 'https://other.example/poc',
    'https://user:password@maat.bytedance.net/poc', 'https://maat.bytedance.net/poc?token=x',
    'https://maat.bytedance.net.evil.example/poc']) {
    assert.equal(safeMaatUrl(url), '');
  }
  assert.equal(shouldPollTask({ state: 'completed', deployment: { resource_releasing: true } }), true);
  assert.equal(shouldPollTask({ state: 'completed', deployment: { released: true } }), false);
});


test('deployment choices respect task execution and per-instance release states', () => {
  const { canDeploy, shouldPollTask } = require('../src/features/ai_node_web_aeolus_card/workflow');
  const worker = { worker_id: 'box', online: true };
  const task = { task_id: 'task', state: 'review', rounds: [{ commit_sha: 'sha' }], deployment_workers: [] };
  assert.equal(canDeploy(task, worker), true);
  for (const state of ['recoding', 'coding', 'failed', 'completed']) assert.equal(canDeploy({ ...task, state }, worker), false);
  for (const status of ['running', 'unknown', 'cancel_requested']) {
    assert.equal(canDeploy({ ...task, deployment: { status } }, worker), false);
  }
  assert.equal(canDeploy({ ...task, deployment: { status: 'succeeded', resource_releasing: true } }, worker), false);
  assert.equal(canDeploy(task, { ...worker, online: false }), false);
  assert.equal(canDeploy(task, { ...worker, supports_repository: false }), false);
  assert.equal(canDeploy(task, { ...worker, supports_repository: true }), true);
  assert.equal(shouldPollTask({ state: 'completed', deployment: { resource_releasing: true } }), true);
});

test('blocked coding waits for human feedback and never enables deployment', () => {
  const { canDeploy, shouldPollTask } = require('../src/features/ai_node_web_aeolus_card/workflow');
  assert.equal(stageForState('coding_blocked'), 'blocked');
  assert.equal(isPollingState('coding_blocked'), true);
  assert.equal(shouldPollTask({ state: 'coding_blocked' }), true);
  assert.deepEqual(historyLayout('coding_blocked'), { order: ['coding', 'evaluation'], expanded: 'coding' });
  assert.equal(roundDisplayStatus({ status: 'blocked', attempts: [{ status: 'blocked' }] }, true), 'blocked');
  assert.equal(canDeploy({ state: 'coding_blocked', rounds: [{ commit_sha: 'a'.repeat(40) }] },
    { online: true, worker_id: 'box' }), false);
});

test('machine counts keep cleanup separate and review polls pressure updates', () => {
  const { deploymentCountLabel, shouldPollTask } = require('../src/features/ai_node_web_aeolus_card/workflow');
  const worker = { deployment_counts: { total: 7, running: 4, deploying: 2, stopping: 1, unknown: 0, cleanup_pending: 2 } };
  assert.match(deploymentCountLabel(worker), /部署 7/);
  assert.match(deploymentCountLabel(worker), /清理待处理 2/);
  assert.equal(shouldPollTask({ state: 'review', deployment_workers: [worker] }), true);
  assert.equal(deploymentCountLabel({}), '');
});

test('PPE progress polls without any Devbox workers, including cleanup after completion', () => {
  assert.equal(shouldPollTask({state: 'review', deployment_workers: [], ppe_capability: {enabled: true}}), true);
  assert.equal(shouldPollTask({state: 'completed', ppe_deployment: {status: 'releasing'}}), true);
  assert.equal(shouldPollTask({state: 'completed', ppe_deployment: {status: 'released'}}), false);
});


test('shows lark authorization only for actionable stages with an unconsented user key', () => {
  const task = { user_key: 'ou_1', authorization_required: true, lark_authorized: false };
  for (const stage of ['validating', 'validation_failed', 'evaluation_review', 'coding', 'review', 'blocked', 'error']) {
    assert.equal(shouldShowLarkAuthorization(task, stage), true, stage);
  }
  for (const stage of ['completing', 'completed']) {
    assert.equal(shouldShowLarkAuthorization(task, stage), false, stage);
  }
  assert.equal(shouldShowLarkAuthorization({ ...task, lark_authorized: true }, 'coding'), false);
  assert.equal(shouldShowLarkAuthorization({ ...task, user_key: '  ' }, 'coding'), false);
  assert.equal(shouldShowLarkAuthorization(task, 'coding'), true);
  task.lark_authorized = true;
  assert.equal(shouldShowLarkAuthorization(task, 'coding'), false);
});

test('falls back to legacy evaluation authorization flags', () => {
  const base = { user_key: 'ou_1', lark_authorized: false };
  assert.equal(shouldShowLarkAuthorization({ ...base, evaluation: { document_authorization_required: true } }, 'coding'), true);
  assert.equal(shouldShowLarkAuthorization({ ...base, evaluation: { coding_authorization_required: true } }, 'error'), true);
  assert.equal(shouldShowLarkAuthorization({ ...base, evaluation: { document_authorization_required: false } }, 'coding'), false);
  assert.equal(shouldShowLarkAuthorization({ ...base, authorization_required: false,
    evaluation: { document_authorization_required: true } }, 'coding'), false);
});

test('accepts only trusted lark document urls and maps precheck failures', () => {
  assert.equal(safeLarkDocumentUrl('https://example.feishu.cn/docx/Abc_123'), 'https://example.feishu.cn/docx/Abc_123');
  assert.equal(safeLarkDocumentUrl('https://example.larkoffice.com/wiki/Abc-123/'), 'https://example.larkoffice.com/wiki/Abc-123/');
  assert.equal(safeLarkDocumentUrl('http://example.feishu.cn/docx/Abc'), '');
  assert.equal(safeLarkDocumentUrl('https://example.feishu.cn.evil.test/docx/Abc'), '');
  assert.equal(safeLarkDocumentUrl('https://example.feishu.cn/sheets/Abc'), '');
  assert.equal(safeLarkDocumentUrl('not a url'), '');

  const failures = documentPrecheckFailures({
    evaluation: {
      document_precheck: {
        failed_documents: [
          { url: 'https://example.feishu.cn/docx/Allowed', reason: 'permission_denied', message: ' 无权限 ' },
          { url: 'https://evil.test/docx/Denied', message: '' },
        ],
      },
    },
  });
  assert.equal(failures.length, 2);
  assert.equal(failures[0].url, 'https://example.feishu.cn/docx/Allowed');
  assert.equal(failures[0].sourceUrl, 'https://example.feishu.cn/docx/Allowed');
  assert.equal(failures[0].reason, 'permission_denied');
  assert.equal(failures[0].message, '无权限');
  assert.equal(failures[1].url, '');
  assert.equal(failures[1].sourceUrl, 'https://evil.test/docx/Denied');
  assert.equal(failures[1].reason, 'read_failed');
  assert.equal(failures[1].message, '该引用文档无法读取');
  assert.deepEqual(documentPrecheckFailures({}), []);
});

test('distinguishes unknown lark authorization from a required one', () => {
  assert.equal(larkAuthorizationState({ lark_authorized: true }), 'authorized');
  assert.equal(larkAuthorizationState({ lark_authorized: false }), 'required');
  assert.equal(larkAuthorizationState({ lark_authorized: null }), 'unknown');
  assert.equal(larkAuthorizationState({}), 'unknown');
});

test('keeps action errors visible when the task already has a last_error', () => {
  assert.equal(shouldShowInlineError('授权链接打开失败', 'error', 'coding failed'), true);
  assert.equal(shouldShowInlineError('授权链接打开失败', 'error', '授权链接打开失败'), false);
  assert.equal(shouldShowInlineError('授权链接打开失败', 'error', ''), false);
  assert.equal(shouldShowInlineError('授权链接打开失败', 'review', 'coding failed'), true);
  assert.equal(shouldShowInlineError('', 'error', 'coding failed'), false);
});
