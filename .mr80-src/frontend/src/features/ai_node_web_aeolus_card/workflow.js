const stageByState = {
  received: 'validating',
  reading_document: 'validating',
  evaluating: 'validating',
  evaluation_review: 'evaluation_review',
  validation_failed: 'validation_failed',
  coding: 'coding',
  recoding: 'coding',
  review: 'review',
  coding_blocked: 'blocked',
  completing: 'completing',
  completed: 'completed',
  failed: 'error',
};

const pollingStates = new Set([
  'received',
  'reading_document',
  'evaluating',
  'evaluation_review',
  'validation_failed',
  'coding',
  'recoding',
  'review',
  'coding_blocked',
  'completing',
  'failed',
]);

const MAX_FEEDBACK_LENGTH = 4_000;

const larkAuthorizationStages = new Set([
  'validating',
  'validation_failed',
  'evaluation_review',
  'coding',
  'review',
  'blocked',
  'error',
]);

function elapsedMs(start, end, now = Date.now()) {
  if (typeof start !== 'string' || !start.trim()) return null;
  const started = Date.parse(start);
  const finished = end == null ? now : Date.parse(end);
  return Number.isFinite(started) && Number.isFinite(finished) && finished >= started
    ? finished - started : null;
}

function formatDuration(milliseconds) {
  if (milliseconds == null || !Number.isFinite(milliseconds) || milliseconds < 0) return '—';
  const seconds = Math.floor(milliseconds / 1000);
  if (seconds < 60) return `${seconds}秒`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}分${seconds % 60}秒`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}小时${minutes % 60}分`;
  return `${Math.floor(hours / 24)}天${hours % 24}小时${minutes % 60}分`;
}

function roundDuration(round, now = Date.now()) {
  const running = ['evaluating', 'coding', 'retrying'].includes(round.status);
  if (!round.finished_at && !running) return '—';
  return formatDuration(elapsedMs(round.started_at, round.finished_at, now));
}

function taskTiming(timing, now = Date.now()) {
  const unknown = { automatic: '—', humanWait: '—', retryWait: '—' };
  if (!timing || !['automatic', 'human_wait', 'retry_wait', 'completed'].includes(timing.current_phase)) return unknown;
  let { automatic_ms: automatic, human_wait_ms: humanWait, retry_wait_ms: retryWait } = timing;
  if (![automatic, humanWait, retryWait].every((value) => typeof value === 'number' && Number.isFinite(value) && value >= 0)) return unknown;
  if (timing.current_phase !== 'completed') {
    const elapsed = elapsedMs(timing.phase_started_at, null, now);
    if (elapsed == null) return unknown;
    if (timing.current_phase === 'automatic') automatic += elapsed;
    else humanWait += elapsed;
    if (timing.current_phase === 'retry_wait') retryWait += elapsed;
  }
  return {
    automatic: formatDuration(automatic),
    humanWait: formatDuration(humanWait),
    retryWait: retryWait > 0 || timing.current_phase === 'retry_wait' ? formatDuration(retryWait) : null,
  };
}

function roundTiming(round, timing, now = Date.now()) {
  const phases = taskTiming(timing, now);
  return {
    total: timing ? formatDuration(elapsedMs(timing.started_at, timing.completed_at, now)) : roundDuration(round, now),
    automatic: phases.automatic,
    humanWait: phases.humanWait,
  };
}

function stageForState(state) {
  return stageByState[state] || 'error';
}

function isPollingState(state) {
  return pollingStates.has(state);
}

function isEvaluationExecutionFailure(task) {
  if (task?.state !== 'validation_failed') return false;
  const rounds = task.evaluation_rounds || [];
  const latest = rounds[rounds.length - 1];
  return latest?.status === 'error' || (!task.evaluation && !!task.last_error);
}

function shouldShowLarkAuthorization(task, stage) {
  if (!larkAuthorizationStages.has(stage)
    || typeof task?.user_key !== 'string'
    || task.user_key.trim().length === 0) return false;
  // Documents are read with the application identity first and only then with
  // the operator's token, so a lapsed user token alone does not block a read.
  // Only a round that both identities failed sets this verdict; older backends
  // report the same condition through the evaluation flags.
  const evaluation = task.evaluation || {};
  const required = typeof task.authorization_required === 'boolean'
    ? task.authorization_required
    : evaluation.document_authorization_required === true
      || evaluation.coding_authorization_required === true;
  if (!required) return false;
  // Consent recorded after the blocked round clears the prompt immediately
  // instead of waiting for the next round to re-read the document.
  return task.lark_authorized !== true;
}

function larkAuthorizationState(task) {
  if (task?.lark_authorized === true) return 'authorized';
  if (task?.lark_authorized === false) return 'required';
  return 'unknown';
}

function shouldShowInlineError(errorMessage, stage, lastError) {
  if (!errorMessage) return false;
  if (stage !== 'error') return true;
  return Boolean(lastError) && errorMessage !== lastError;
}

function historyLayout(state) {
  const stage = stageForState(state);
  const codingFirst = ['coding', 'review', 'blocked', 'completing', 'completed', 'error'].includes(stage);
  return {
    order: codingFirst ? ['coding', 'evaluation'] : ['evaluation', 'coding'],
    expanded: stage === 'completed' ? null : codingFirst ? 'coding' : 'evaluation',
  };
}

function taskLookupUrl(apiBase, context) {
  const url = new URL('/api/aeolus/tasks/by-node', apiBase);
  url.searchParams.set('space_id', context.spaceId);
  url.searchParams.set('work_item_id', String(context.workItemId));
  url.searchParams.set('node_id', context.nodeId);
  url.searchParams.set('work_object_id', context.workObjectId);
  return url.toString();
}

function safeCodeUrl(value) {
  try {
    const url = new URL(value);
    return url.protocol === 'https:' && url.hostname === 'code.byted.org'
      ? url.toString()
      : '';
  } catch {
    return '';
  }
}

function safeLarkDocumentUrl(value) {
  try {
    const url = new URL(value);
    const host = url.hostname.toLowerCase();
    const trustedHost = host === 'larkoffice.com' || host.endsWith('.larkoffice.com')
      || host === 'feishu.cn' || host.endsWith('.feishu.cn');
    return url.protocol === 'https:' && trustedHost
      && /^\/(?:docx|wiki)\/[A-Za-z0-9_-]+\/?$/.test(url.pathname)
      ? url.toString()
      : '';
  } catch {
    return '';
  }
}

function documentPrecheckFailures(task) {
  const failures = task?.evaluation?.document_precheck?.failed_documents;
  if (!Array.isArray(failures)) return [];
  return failures.map((item) => {
    const url = safeLarkDocumentUrl(item?.url);
    return {
      url,
      sourceUrl: typeof item?.url === 'string' ? item.url.trim().slice(0, 500) : '',
      reason: typeof item.reason === 'string' ? item.reason : 'read_failed',
      message: typeof item.message === 'string' && item.message.trim()
        ? item.message.trim()
        : '该引用文档无法读取',
    };
  });
}

function safeMaatUrl(value) {
  try {
    const url = new URL(value);
    return url.protocol === 'https:' && (url.hostname === 'maat.bytedance.net' || url.hostname.endsWith('.maat.bytedance.net'))
      && !url.username && !url.password && !url.search && !url.hash ? url.toString() : '';
  } catch {
    return '';
  }
}

function shouldPollTask(task) {
  return isPollingState(task.state) || !!task.deployment?.resource_releasing
    || task.state === 'review' && (!!task.deployment_workers?.length || !!task.ppe_capability?.enabled)
    || ['pending', 'running', 'releasing', 'unknown', 'deployed'].includes(task.ppe_deployment?.status);
}

function deploymentCountLabel(worker) {
  const counts = worker?.deployment_counts;
  if (!counts) return '';
  return `部署 ${counts.total} · 运行 ${counts.running} · 部署中 ${counts.deploying}`
    + ` · 停止待确认 ${counts.stopping} · 未知 ${counts.unknown} · 清理待处理 ${counts.cleanup_pending}`;
}

function canDeploy(task, worker) {
  return task?.state === 'review' && !!worker?.online
    && worker?.supports_repository !== false
    && !task.deployment?.resource_releasing
    && !!task.rounds?.at(-1)?.commit_sha
    && !['running', 'unknown', 'cancel_requested'].includes(task.deployment?.status);
}

function canSubmitFeedback(value) {
  return typeof value === 'string'
    && value.trim().length > 0
    && value.length <= MAX_FEEDBACK_LENGTH;
}

function roundDisplayStatus(round, hasLaterRound) {
  const attempts = Array.isArray(round.attempts) ? round.attempts : [];
  if (round.status === 'coding' && attempts.some((attempt) => attempt.status === 'failed')) {
    return 'retrying';
  }
  if (round.status === 'review' && hasLaterRound) return 'revision_requested';
  return round.status;
}

function roundTitle(round) {
  return round.trigger_type === 'review_feedback'
    ? `第 ${round.round_no} 轮 · 反馈修改`
    : `第 ${round.round_no} 轮 · 初次编码`;
}

function roundPreview(round) {
  if (round.feedback) return round.feedback;
  if (round.summary) return round.summary;
  if (round.status === 'coding') return '正在根据需求文档生成代码';
  if (round.error) return round.error;
  return '根据需求文档进行首次编码';
}

module.exports = {
  MAX_FEEDBACK_LENGTH,
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
  safeLarkDocumentUrl,
  documentPrecheckFailures,
  shouldShowLarkAuthorization,
  larkAuthorizationState,
  shouldShowInlineError,
  safeMaatUrl,
  shouldPollTask,
  canDeploy,
  deploymentCountLabel,
  stageForState,
  taskLookupUrl,
  taskTiming,
};
