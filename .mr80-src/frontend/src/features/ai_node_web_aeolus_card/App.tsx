import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import buildConfig from '../../build-config.json';
import './index.css';
import RequirementContract from './RequirementContract';
import Select from './Select';
import TaskIdFooter from './TaskIdFooter';
import { completionRequest, completionOptions, ppeRequest, scopedRequest, ppeEnvironmentNotice } from './ppe';
import { runtimeScope, runtimeSelection } from './runtime';

import {
  MAX_FEEDBACK_LENGTH,
  canSubmitFeedback,
  elapsedMs,
  formatDuration,
  historyLayout,
  isEvaluationExecutionFailure,
  shouldPollTask,
  canDeploy,
  deploymentCountLabel,
  documentPrecheckFailures,
  safeLarkDocumentUrl,
  shouldShowLarkAuthorization,
  larkAuthorizationState,
  shouldShowInlineError,
  safeMaatUrl,
  roundDisplayStatus,
  roundTiming,
  roundPreview,
  roundTitle,
  safeCodeUrl,
  stageForState,
  taskLookupUrl,
  taskTiming,
} from './workflow';

const API_BASE = buildConfig.apiBase;
const POLL_INTERVAL_MS = 5_000;
const MAX_POLL_FAILURES = 3;

type Stage =
  | 'validating'
  | 'validation_failed'
  | 'evaluation_review'
  | 'coding'
  | 'review'
  | 'blocked'
  | 'completing'
  | 'completed'
  | 'error';

type NodeContext = {
  spaceId: string;
  workObjectId: string;
  workItemId: number;
  nodeId: string;
};

type Evaluation = {
  code_context?: { repository?: string };
  valid?: boolean;
  summary?: string;
  missing_items?: string[];
  risks?: string[];
  requirement_contract?: string;
  document_read_error_code?: number | null;
  document_authorization_required?: boolean;
  coding_authorization_required?: boolean;
  document_precheck?: {
    status?: 'authorization_required' | 'failed';
    total_documents?: number;
    failed_documents?: Array<{
      url?: string;
      reason?: string;
      message?: string;
    }>;
  };
};

type EvaluationRound = {
  round_no: number;
  status: 'evaluating' | 'review' | 'revision_requested' | 'confirmed' | 'error';
  feedback: string;
  reviewed_round_no: number | null;
  result: Evaluation | null;
  error?: string;
  started_at?: string;
  finished_at?: string | null;
};

type CodingRound = {
  round_no: number;
  status: 'coding' | 'review' | 'blocked' | 'revision_requested' | 'completed' | 'error';
  feedback: string;
  trigger_type?: 'initial' | 'review_feedback';
  reviewed_round_no?: number | null;
  base_commit_sha?: string;
  summary: string;
  commit_sha?: string;
  code_url: string;
  tests: string[];
  unresolved_items?: string[];
  failure_code?: string;
  error: string;
  attempts?: CodingAttempt[];
  started_at?: string;
  finished_at?: string | null;
};

type CodingAttempt = {
  attempt_no: number;
  status: 'running' | 'succeeded' | 'blocked' | 'failed';
  session_id: string;
  run_id: string;
  error: string;
};

type Timing = {
  automatic_ms: number;
  human_wait_ms: number;
  retry_wait_ms: number;
  current_phase: string;
  phase_started_at: string;
  completed_at: string | null;
  started_at?: string;
};

function RoundTiming({ round, timing, now }: {
  round: EvaluationRound | CodingRound;
  timing?: Timing;
  now: number;
}) {
  const duration = roundTiming(round, timing, now);
  return <small className="round-duration">
    <span>历时 <b>{duration.total}</b></span>
    <span>自动 {duration.automatic}</span>
    <span>等待 {duration.humanWait}</span>
  </small>;
}

type RuntimeProfile = {
  python_version: string; profile_revision: string; runtime_revision?: string;
  compatible?: boolean; incompatibility?: string;
};
type RuntimeDetection = {
  commit_sha: string; evidence: { path: string; kind: string; detail: string }[];
  warnings: string[]; branch_hints?: string[];
};
const RuntimeEvidence: React.FC<{ detection?: RuntimeDetection }> = ({ detection }) => detection ? <>
  <small>识别提交：{detection.commit_sha.slice(0, 8) || '待生成'}</small>
  {detection.evidence.map((item, index) => <small key={index}>{item.path}：{item.detail}</small>)}
  {detection.warnings.map((warning, index) => <small key={`warning-${index}`}>{warning}</small>)}
  {!!detection.branch_hints?.length && <small>基础分支提示：{detection.branch_hints.join('、')}（仅供参考）</small>}
</> : null;

type WorkerSnapshot = {
  worker_id: string; display_name: string; vmid?: string | null; online: boolean;
  supports_repository?: boolean;
  protocol_version?: number;
  deployment_counts?: { total: number; running: number; deploying: number; stopping: number;
    unknown: number; cleanup_pending: number; by_repository: Record<string, number> };
  counts_observed_at?: string;
  runtime_profiles?: Record<string, RuntimeProfile>;
  recommended_runtime_profile_id?: string | null;
  runtime_detection?: RuntimeDetection;

};

type DeploymentSnapshot = {
  deployment_id: string; worker_id: string; status: string; stage: string;
  display_name: string; vmid?: string | null; commit_sha: string; maat_url: string;
  error?: string; log_path?: string; current?: boolean; resource_releasing?: boolean;
  released?: boolean; verified?: boolean; resource_owned?: boolean;
  generation?: number | null; can_release?: boolean; superseded?: boolean; status_label?: string; status_note?: string;
  protocol_version?: number; runtime_profile_id?: string; runtime_revision?: string; python_version?: string;
};

type TaskSnapshot = {
  task_id: string;
  state: string;
  deployment?: DeploymentSnapshot | null;
  deployment_workers?: WorkerSnapshot[] | null;
  ppe_capability?: { enabled: boolean; profile_revision: string; python_version?: string | null;
    runtime_profiles?: Record<string, RuntimeProfile>; recommended_runtime_profile_id?: string | null;
    runtime_detection?: RuntimeDetection } | null;
  ppe_deployment?: {
    deployment_id: string; environment_id: string; generation: number; environment_name: string;
    round_no: number; commit_sha: string; status: string; stage: string; status_label: string; error?: string;
    python_version?: string | null;
    scm_version?: string; image_version?: string; image_url?: string; ticket_id?: string;
    bits_run_id?: string; bits_run_seq?: string; bits_run_url?: string;
    resource_state: string; release_requested: boolean; released: boolean; verified: boolean; superseded: boolean;
    can_verify: boolean; can_retry: boolean; can_release: boolean; can_deploy: boolean;
    can_view_environment: boolean; can_access_environment: boolean;
    environment_url: string; scm_url: string; access_hint: string; due_time?: string | number;
    clusters: { id: string; name: string; psm?: string; image_version?: string; ready?: boolean; replicas?: number }[];
    current_deployment?: {
      deployment_id: string; round_no: number; commit_sha: string; image_version?: string;
    } | null;
    approvals?: { cluster_name: string; tce_ticket_id: string }[];
  } | null;
  evaluation: Evaluation | null;
  evaluation_rounds?: EvaluationRound[];
  mr_url: string;
  rounds: CodingRound[];
  last_error: string;
  user_key?: string;
  authorization_required?: boolean;
  action_token?: string;
  can_retry_coding?: boolean;
  can_retry_completion?: boolean;
  created_at?: string;
  updated_at?: string;
  timing?: Timing & {
    current_round?: string;
    rounds?: Record<string, Timing>;
  } | null;
};

const stageLabels: Record<Stage, string> = {
  validating: '输入校验',
  validation_failed: '待补充',
  evaluation_review: '评估待确认',
  coding: 'Coding',
  review: '待确认',
  blocked: '编码阻塞，待处理',
  completing: '完成中',
  completed: '已完成',
  error: '处理失败',
};

const roundStatusLabels = {
  coding: '处理中',
  retrying: '重试处理中',
  review: '待确认',
  blocked: '编码阻塞，待处理',
  revision_requested: '需修改',
  completed: '已接受',
  error: '失败',
};

const attemptStatusLabels = {
  running: '执行中',
  succeeded: '执行成功',
  blocked: '等待人工处理',
  failed: '执行失败',
};

function sleep(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function requestId() {
  return typeof crypto?.randomUUID === 'function'
    ? crypto.randomUUID()
    : `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

async function readError(response: Response) {
  try {
    const body = await response.json();
    return typeof body.detail === 'string' ? body.detail : `请求失败 (${response.status})`;
  } catch {
    return `请求失败 (${response.status})`;
  }
}

const App: React.FC = () => {
  const [context, setContext] = useState<NodeContext | null>(null);
  const [task, setTask] = useState<TaskSnapshot | null>(null);
  const [feedback, setFeedback] = useState('');
  const [selectedWorker, setSelectedWorker] = useState('');
  const [selectedRuntime, setSelectedRuntime] = useState('');
  const [selectedPpeRuntime, setSelectedPpeRuntime] = useState('');
  const deploymentRequestRef = useRef<{ key: string; id: string } | null>(null);
  const deploymentWorker = task?.deployment_workers?.find((worker) => worker.worker_id === selectedWorker);
  const runtimeProfiles = deploymentWorker?.runtime_profiles || {};
  const runtimeKey = runtimeScope(task, selectedWorker, runtimeProfiles);
  const runtimeWorkerRef = useRef('');
  useEffect(() => {
    const recommended = deploymentWorker?.recommended_runtime_profile_id;
    const sameWorker = runtimeWorkerRef.current === runtimeKey;
    runtimeWorkerRef.current = runtimeKey;
    setSelectedRuntime(previous => runtimeSelection(previous, sameWorker, runtimeProfiles, recommended));
    deploymentRequestRef.current = null;
  }, [runtimeKey, deploymentWorker?.recommended_runtime_profile_id]);
  const ppeProfiles = task?.ppe_capability?.runtime_profiles || {};
  const ppeRuntimeKey = runtimeScope(task, 'ppe', ppeProfiles);
  const ppeRuntimeRef = useRef('');
  useEffect(() => {
    const sameScope = ppeRuntimeRef.current === ppeRuntimeKey;
    ppeRuntimeRef.current = ppeRuntimeKey;
    setSelectedPpeRuntime(previous => runtimeSelection(
      previous, sameScope, ppeProfiles, task?.ppe_capability?.recommended_runtime_profile_id,
    ));
  }, [ppeRuntimeKey, task?.ppe_capability?.recommended_runtime_profile_id]);
  const [verificationMode, setVerificationMode] = useState('');
  useEffect(() => setVerificationMode(''), [task?.task_id, task?.deployment?.deployment_id,
    task?.ppe_deployment?.deployment_id, task?.rounds?.at(-1)?.round_no,
    task?.deployment?.current, task?.ppe_deployment?.can_verify]);
  const [evaluationFeedback, setEvaluationFeedback] = useState('');
  const [errorMessage, setErrorMessage] = useState('');
  const [waitingForTask, setWaitingForTask] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [pollVersion, setPollVersion] = useState(0);
  const actionTokenRef = useRef('');
  const feedbackRequestRef = useRef('');
  const confirmRequestRef = useRef<{ key: string; id: string } | null>(null);
  const retryCodingRequestRef = useRef('');
  const evaluationFeedbackRequestRef = useRef('');
  const evaluationConfirmRequestRef = useRef('');
  const ppeRequestRef = useRef<{ key: string; id: string } | null>(null);
  useEffect(() => {
    setFeedback('');
    feedbackRequestRef.current = '';
  }, [task?.task_id, task?.rounds?.at(-1)?.round_no]);
  useEffect(() => {
    setEvaluationFeedback('');
    evaluationFeedbackRequestRef.current = '';
  }, [task?.task_id, task?.evaluation_rounds?.at(-1)?.round_no]);
  const completion = completionRequest(task, verificationMode);
  const ppeNotice = ppeEnvironmentNotice(task?.ppe_deployment);

  const stage = (task ? stageForState(task.state) : 'validating') as Stage;
  const evaluationFailed = isEvaluationExecutionFailure(task);
  const failedDocuments = useMemo(() => documentPrecheckFailures(task), [task]);
  const larkAuthorization = shouldShowLarkAuthorization(task, stage) ? larkAuthorizationState(task) : 'authorized';
  const rounds = task?.rounds || [];
  const history = historyLayout(task?.state || 'received');
  const [now, setNow] = useState(Date.now);
  useEffect(() => {
    setNow(Date.now());
    if (!task || task.state === 'completed') return;
    const timer = setInterval(() => setNow(Date.now()), POLL_INTERVAL_MS);
    return () => clearInterval(timer);
  }, [task?.task_id, task?.state]);
  const taskDuration = task ? formatDuration(
    task.state === 'completed' && !task.updated_at ? null
      : elapsedMs(task.created_at, task.state === 'completed' ? task.timing?.completed_at || task.updated_at : null, now),
  ) : '—';
  const timing = taskTiming(task?.timing, now);
  const codeUrl = safeCodeUrl(task?.mr_url || rounds[rounds.length - 1]?.code_url || '');

  const applySnapshot = useCallback((snapshot: TaskSnapshot) => {
    if (snapshot.action_token) actionTokenRef.current = snapshot.action_token;
    setTask(snapshot);
    setWaitingForTask(false);
    setErrorMessage('');
  }, []);

  useEffect(() => {
    let active = true;

    const loadContext = async () => {
      try {
        const next = await window.JSSDK.aiNode.getContext();
        console.log('[getContext]', next);
        if (active) setContext(next);
      } catch (error) {
        console.error('[getContext]', error);
        if (active) setErrorMessage('无法读取 AI 节点上下文，请刷新后重试。');
      }
    };

    void loadContext();
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    if (!context) return undefined;
    let active = true;

    const poll = async () => {
      let failureCount = 0;
      while (active) {
        try {
          const response = await fetch(taskLookupUrl(API_BASE, context));
          if (response.status === 404) {
            actionTokenRef.current = '';
            setTask(null);
            setWaitingForTask(true);
            setErrorMessage('');
            await sleep(POLL_INTERVAL_MS);
            continue;
          }
          if (!response.ok) throw new Error(await readError(response));

          const snapshot = await response.json() as TaskSnapshot;
          if (!active) return;
          applySnapshot(snapshot);
          failureCount = 0;
          if (!shouldPollTask(snapshot)) return;
        } catch (error) {
          console.error('[loadAeolusTask]', error);
          failureCount += 1;
          if (active) {
            setErrorMessage(error instanceof Error ? error.message : '加载任务失败，请重试。');
          }
          if (failureCount >= MAX_POLL_FAILURES) return;
        }
        await sleep(POLL_INTERVAL_MS);
      }
    };

    void poll();
    return () => {
      active = false;
    };
  }, [applySnapshot, context, pollVersion]);

  const postAction = useCallback(async (
    path: string,
    body?: Record<string, unknown>,
  ) => {
    if (!task || !actionTokenRef.current) return false;
    setSubmitting(true);
    setErrorMessage('');
    try {
      const response = await fetch(`${API_BASE}/api/aeolus/tasks/${encodeURIComponent(task.task_id)}${path}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-Aeolus-Task-Token': actionTokenRef.current,
        },
        body: body ? JSON.stringify(body) : undefined,
      });
      if (!response.ok) {
        setErrorMessage(await readError(response));
        return false;
      }
      applySnapshot(await response.json() as TaskSnapshot);
      setPollVersion((current) => current + 1);
      return true;
    } catch (error) {
      console.error('[aeolusAction]', error);
      setErrorMessage(error instanceof Error ? error.message : '操作失败，请重试。');
      setPollVersion((current) => current + 1);
      return false;
    } finally {
      setSubmitting(false);
    }
  }, [applySnapshot, task]);

  const retryValidation = () => {
    void postAction('/retry-validation');
  };

  const retryCoding = async () => {
    if (!retryCodingRequestRef.current) retryCodingRequestRef.current = requestId();
    if (await postAction('/retry-coding', { request_id: retryCodingRequestRef.current })) {
      retryCodingRequestRef.current = '';
    }
  };

  const submitEvaluationFeedback = async () => {
    if (!canSubmitFeedback(evaluationFeedback)) return;
    if (!evaluationFeedbackRequestRef.current) evaluationFeedbackRequestRef.current = requestId();
    const accepted = await postAction('/evaluation-feedback', {
      feedback: evaluationFeedback.trim(),
      request_id: evaluationFeedbackRequestRef.current,
    });
    if (accepted) {
      evaluationFeedbackRequestRef.current = '';
      setEvaluationFeedback('');
    }
  };

  const confirmEvaluation = async () => {
    if (!evaluationConfirmRequestRef.current) evaluationConfirmRequestRef.current = requestId();
    if (await postAction('/confirm-evaluation', { request_id: evaluationConfirmRequestRef.current })) {
      evaluationConfirmRequestRef.current = '';
    }
  };

  const recode = async () => {
    if (!canSubmitFeedback(feedback)) return;
    if (!feedbackRequestRef.current) feedbackRequestRef.current = requestId();
    const accepted = await postAction('/feedback', {
      feedback: feedback.trim(),
      request_id: feedbackRequestRef.current,
    });
    if (accepted) {
      feedbackRequestRef.current = '';
      setFeedback('');
    }
  };

  const confirm = async () => {
    if (!completion) return;
    confirmRequestRef.current = scopedRequest(confirmRequestRef.current,
      JSON.stringify([task?.task_id, rounds.at(-1)?.round_no, completion]), requestId);
    const accepted = await postAction('/confirm', {
      ...completion, request_id: confirmRequestRef.current!.id,
    });
    if (accepted) confirmRequestRef.current = null;
  };

  const deployPpe = async (retry = false) => {
    const request = ppeRequest(task, retry, selectedPpeRuntime);
    if (!request) return;
    ppeRequestRef.current = scopedRequest(ppeRequestRef.current, request.key, requestId);
    if (await postAction(request.path, {
      ...request.body, request_id: ppeRequestRef.current!.id,
    })) ppeRequestRef.current = null;
  };

  const releasePpe = async () => {
    const ppe = task?.ppe_deployment;
    if (!ppe?.can_release) return;
    await postAction(`/ppe-deployments/${ppe.deployment_id}/release`, {
      environment_id: ppe.environment_id, generation: ppe.generation,
    });
  };

  const openPpeLink = (value: string) => {
    try {
      const url = new URL(value);
      if (url.protocol === 'https:' && ['bits.bytedance.net', 'cloud.bytedance.net'].includes(url.hostname)
          && !url.username && !url.password) window.JSSDK.navigation.open(url.toString());
    } catch { setErrorMessage('PPE 链接无效，请刷新重试。'); }
  };

  const deploy = async () => {
    const round = rounds[rounds.length - 1];
    if (!round?.commit_sha || !selectedWorker || !canDeploy(task, deploymentWorker)
      || !selectedRuntime || !(selectedRuntime in runtimeProfiles)) return;
    const key = JSON.stringify([runtimeKey, selectedRuntime]);
    const scoped = scopedRequest(deploymentRequestRef.current, key, requestId);
    deploymentRequestRef.current = scoped;
    if (await postAction('/deploy', {
      request_id: scoped.id, worker_id: selectedWorker,
      round_no: String(round.round_no), commit_sha: round.commit_sha,
      ...(selectedRuntime ? {
        runtime_profile_id: selectedRuntime,
        expected_profile_revision: deploymentWorker?.runtime_profiles?.[selectedRuntime]?.profile_revision,
      } : {}),
    })) deploymentRequestRef.current = null;
  };

  const releaseEnvironment = async () => {
    if (!task?.deployment?.can_release) return;
    await postAction('/release-environment', {
      deployment_id: task.deployment.deployment_id, generation: task.deployment.generation ?? null,
    });
  };

  const openEnvironment = () => {
    const url = safeMaatUrl(task?.deployment?.maat_url);
    const deployment = task?.deployment;
    const pending = deployment?.stage === 'maat_entry_pending' && deployment.resource_owned
      && !deployment.resource_releasing && !deployment.superseded;
    if (url && (deployment?.current || pending)) window.JSSDK.navigation.open(url);
  };

  const reload = () => {
    setErrorMessage('');
    setPollVersion((current) => current + 1);
  };

  const openCode = (targetUrl: string) => {
    const safeUrl = safeCodeUrl(targetUrl);
    if (!safeUrl) {
      setErrorMessage('代码链接无效或不受信任。');
      return;
    }
    try {
      window.JSSDK.navigation.open(safeUrl);
    } catch (error) {
      console.error('[open]', error);
      setErrorMessage('代码链接打开失败，请稍后重试。');
    }
  };

  const openLarkAuthorize = () => {
    if (!task?.user_key) return;
    const url = `${API_BASE}/oauth/lark/start?user_key=${encodeURIComponent(task.user_key)}`;
    try {
      window.JSSDK.navigation.open(url);
    } catch (error) {
      console.error('[openLarkAuthorize]', error);
      setErrorMessage('授权链接打开失败，请稍后重试。');
    }
  };

  const openLarkDocument = (targetUrl: string) => {
    const safeUrl = safeLarkDocumentUrl(targetUrl);
    if (!safeUrl) {
      setErrorMessage('飞书文档链接无效或不受信任。');
      return;
    }
    try {
      window.JSSDK.navigation.open(safeUrl);
    } catch (error) {
      console.error('[openLarkDocument]', error);
      setErrorMessage('飞书文档链接打开失败，请稍后重试。');
    }
  };

  const renderedRounds = useMemo(() => rounds.map((round, index) => {
    const nextRound = rounds[index + 1];
    const displayStatus = roundDisplayStatus(
      round,
      index < rounds.length - 1,
    ) as keyof typeof roundStatusLabels;
    return {
      ...round,
      displayStatus,
      title: roundTitle(round),
      preview: roundPreview(round),
      safeCodeUrl: safeCodeUrl(round.code_url),
      nextRevision: nextRound?.reviewed_round_no === round.round_no
        ? nextRound
        : undefined,
    };
  }).reverse(), [rounds]);

  return (
    <main className="card">
      <header className="card-header">
        <div>
          <p className="eyebrow">AI 节点</p>
          <h1>aeolus 编码助手</h1>
        </div>
        <span className={`status status-${evaluationFailed ? 'error' : stage}`}>
          {evaluationFailed ? '评估执行失败' : stageLabels[stage]}
        </span>
        {task && <div className="task-timing">
          <div className="timing-metrics">
            <div><span>{task.state === 'completed' ? '整体历时' : '已历时'}</span><strong>{taskDuration}</strong></div>
            <div><span>自动处理</span><strong>{timing.automatic}</strong></div>
            <div><span>等待人工</span><strong>{timing.humanWait}</strong></div>
          </div>
          <details className="timing-explanation">
            <summary>耗时说明</summary>
            {timing.retryWait && timing.retryWait !== '—' && <small>其中失败后等待重试 · {timing.retryWait}</small>}
            <small>{task.timing ? '自动处理包含排队及恢复时间；等待人工包含失败后等待重试。历史轮次未记录的分项显示为 —。' : '历史任务未记录阶段耗时，仅展示整体及轮次历时。'}</small>
          </details>
        </div>}
      </header>

      <section className="card-body" aria-live="polite">
        <div className="workspace">
        <div className="workspace-primary" key={`${task?.task_id}:${rounds.at(-1)?.round_no}:${task?.evaluation_rounds?.at(-1)?.round_no}`}>
        {stage === 'validating' && (
          <div className="processing" role="status">
            <span className="spinner" aria-hidden="true" />
            <div>
              <h2>{waitingForTask ? '正在等待任务启动' : '正在检查需求文档'}</h2>
              <p>{waitingForTask ? '节点触发后将自动展示校验进度。' : '正在读取并评估节点配置的需求文档。'}</p>
            </div>
          </div>
        )}

        {larkAuthorization !== 'authorized' && (
          <div className="notice notice-warning">
            <strong>{larkAuthorization === 'unknown' ? '暂时无法确认飞书授权状态' : '需要飞书授权'}</strong>
            <span>
              {larkAuthorization === 'unknown'
                ? '授权状态暂时无法确认，需求文档或其中引用的文档可能无法读取。请稍后重新检查。'
                : '当前没有可用的飞书用户授权，需求文档或其中引用的文档可能无法读取。完成授权后返回卡片；若 Coding 已停止，请在授权后重试编码。'}
            </span>
            {larkAuthorization === 'required' ? (
              <button className="button button-primary" onClick={openLarkAuthorize}>
                去飞书授权
              </button>
            ) : (
              <button className="button button-secondary" onClick={reload}>
                重新检查
              </button>
            )}
          </div>
        )}

        {(stage === 'validation_failed' || stage === 'evaluation_review') && (
          <div className="stack evaluation-panel">
            <div className={`notice result-panel ${evaluationFailed ? 'notice-error' : 'notice-warning'}`}>
              <div className="result-notice">
                <div className="result-copy">
                  <strong>{evaluationFailed ? '评估执行失败' : task?.evaluation?.valid ? '评估通过，请确认' : '需求文档需要补充'}</strong>
                  <span>{task?.evaluation?.summary || task?.last_error || '请补充需求信息后重新评估。'}</span>
                </div>
                {stage === 'evaluation_review' && task?.evaluation?.valid && (
                  <button className="button button-primary result-action" disabled={submitting} onClick={confirmEvaluation}>
                    {submitting ? '提交中…' : '确认需求，开始 Coding'}
                  </button>
                )}
              </div>
              {evaluationFailed && <span>本轮未完成评估，不代表需求不合理。可直接重试，无需修改文档；若持续失败，请联系维护人员检查。</span>}
              {!!failedDocuments.length && (
                <div className="evaluation-issues">
                  <strong>无法读取的引用文档</strong>
                  <ul className="missing-items evaluation-list">
                    {failedDocuments.map((item, index) => (
                      <li key={`${item.sourceUrl}-${index}`}>
                        <span>{item.message}</span>
                        <span>{item.url || item.sourceUrl}</span>
                        {!!item.url && (
                          <button className="button" onClick={() => openLarkDocument(item.url)}>
                            {item.reason === 'permission_denied' ? '打开并申请权限' : '打开文档'}
                          </button>
                        )}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
              {!failedDocuments.length && !!task?.evaluation?.missing_items?.length && (
                <div className="evaluation-issues"><strong>待补充信息</strong>
                  <ol className="missing-items evaluation-list">{task.evaluation.missing_items.map((item) => <li key={item}>{item}</li>)}</ol>
                </div>
              )}
              {!!task?.evaluation?.risks?.length && (
                <details className="evaluation-risks">
                  <summary>风险提示 · {task.evaluation.risks.length} 项</summary>
                  <ol className="missing-items evaluation-list">{task.evaluation.risks.map((risk, index) => <li key={index}>{risk}</li>)}</ol>
                </details>
              )}
              {task?.evaluation?.requirement_contract && (
                <RequirementContract title="查看需求共识 · 待确认" content={task.evaluation.requirement_contract} />
              )}
            </div>
            <div className="stack feedback-panel">
              <h2>补充或调整需求</h2>
              <label className="feedback-field">
                <span>评估反馈</span>
                <textarea value={evaluationFeedback} onChange={(event) => setEvaluationFeedback(event.target.value)}
                  placeholder={evaluationFailed ? '可选：补充处理说明；也可直接点击重试评估' : '补充需求信息，或指出评估结论需要调整的地方'} rows={3} maxLength={MAX_FEEDBACK_LENGTH} />
                <span className="feedback-count">{evaluationFeedback.length}/{MAX_FEEDBACK_LENGTH}</span>
              </label>
              <div className="actions">
                <button className="button button-secondary"
                  disabled={submitting || !canSubmitFeedback(evaluationFeedback)} onClick={submitEvaluationFeedback}>
                  反馈并重新评估
                </button>
                <button className={`button ${evaluationFailed || !task?.evaluation?.valid ? 'button-primary' : 'button-secondary'}`}
                  disabled={submitting} onClick={retryValidation}>
                  {submitting ? '提交中…' : evaluationFailed ? '重试评估' : '已更新文档，重新评估'}
                </button>
              </div>
            </div>
          </div>
        )}

        {stage === 'coding' && (
          <div className="processing" role="status">
            <span className="spinner" aria-hidden="true" />
            <div>
              <h2>正在 Coding</h2>
              <p>正在根据已通过校验的需求文档生成代码。</p>
            </div>
          </div>
        )}

        {stage === 'blocked' && <div className="current-result">
          <div className="notice notice-warning result-panel">
            <strong>编码阻塞，待人工处理</strong>
            <span>{rounds.at(-1)?.summary || '本轮需要补充信息或处理阻塞后继续。'}</span>
            {!!rounds.at(-1)?.unresolved_items?.length && <>
              <strong>阻塞项 · {rounds.at(-1)?.unresolved_items?.length} 项</strong>
              <ol>{rounds.at(-1)?.unresolved_items?.map((item, index) => <li key={index}>{item}</li>)}</ol>
            </>}
            {rounds.at(-1)?.failure_code && <small>错误码：{rounds.at(-1)?.failure_code}</small>}
            {codeUrl && <button className="code-link" onClick={() => openCode(codeUrl)}>
              Review 已有代码<span aria-hidden="true">↗</span>
            </button>}
            <small>本轮尚未完成验收；请提供处理意见后继续编码。</small>
          </div>
        </div>}

        {stage === 'review' && <div className="current-result">
          <div className="notice notice-success result-panel">
            <div className="result-notice">
              <div className="result-copy">
                <strong>代码已生成</strong>
                <span>{rounds.at(-1)?.summary?.slice(0, 400) || '先查看代码，再选择完成、部署验证或提出修改建议。'}</span>
              </div>
              <button className="code-link result-action" disabled={!codeUrl} onClick={() => openCode(codeUrl)}>
                查看代码<span aria-hidden="true">↗</span>
              </button>
            </div>
          </div>
        </div>}

        {stage === 'completing' && (
          <div className="processing" role="status">
            <span className="spinner" aria-hidden="true" />
            <div><h2>正在完成任务</h2><p>正在同步 AI 节点状态，请稍候。</p></div>
          </div>
        )}

        {stage === 'completed' && (
          <div className="current-result">
            <div className="notice notice-success result-panel">
              <div className="result-notice">
                <div className="result-copy">
                  <strong>任务已完成</strong><span>代码结果已由用户确认。</span>
                </div>
                <button className="code-link result-action" disabled={!codeUrl} onClick={() => openCode(codeUrl)}>
                  查看代码<span aria-hidden="true">↗</span>
                </button>
              </div>
              {task?.evaluation?.requirement_contract && <RequirementContract
                title="查看编码依据 · 已确认需求共识" content={task.evaluation.requirement_contract} />}
            </div>
          </div>
        )}

        {stage === 'error' && (
          <div className="stack">
            <div className="notice notice-error">
              <strong>处理失败</strong>
              <span>{task?.last_error || errorMessage || '发生未知错误，请稍后重试。'}</span>
            </div>
            {task?.can_retry_completion && (
              <button className="button button-primary" disabled={submitting}
                onClick={() => void postAction('/retry-completion', { request_id: requestId() })}>
                重试完成节点
              </button>
            )}
            {task?.can_retry_coding && (
              <button className="button button-primary" disabled={submitting} onClick={retryCoding}>
                {submitting ? '正在提交…' : '重试编码'}
              </button>
            )}
            <button className="button button-secondary" onClick={reload}>重新加载</button>
          </div>
        )}

        {(stage === 'review' || stage === 'blocked') && (
          <div className="stack review-panel">
            <h2>{stage === 'blocked' ? '提供处理意见' : '完成确认'}</h2>
            {stage === 'review' && <>
              <div className="actions">
                <Select label="验证结果" placeholder="请选择验证结果" value={verificationMode}
                  options={completionOptions(task)} disabled={submitting}
                  onChange={value => { setVerificationMode(value); confirmRequestRef.current = null; }} />
                <button className="button button-primary" disabled={submitting || !completion} onClick={confirm}>
                  {submitting ? '提交中…' : task?.deployment?.resource_owned && !task.deployment.resource_releasing
                    || task?.ppe_deployment && !task.ppe_deployment.released && !task.ppe_deployment.release_requested
                    ? '完成节点并释放环境' : '确认代码，完成节点'}
                </button>
              </div>
              {verificationMode && !completion && <small role="status">所选环境暂不可验证，请刷新状态或重新选择验证结果。</small>}
              {verificationMode === 'all_verified' && <small>
                将确认当前{[task?.deployment?.current && 'Devbox', task?.ppe_deployment?.can_verify && 'PPE']
                  .filter(Boolean).join(' 和 ')}环境验证通过。
              </small>}
              {(task?.deployment?.resource_owned || task?.ppe_deployment && !task.ppe_deployment.released) &&
                <small>完成节点将释放当前任务所属环境；释放结果以停服或平台回查确认为准。</small>}
            </>}
            <details className="feedback-disclosure" open={stage === 'blocked' || !!feedback || undefined}>
              <summary>{stage === 'blocked' ? '处理意见 · 继续 Coding' : '提出修改建议'}</summary>
              <div className="stack feedback-panel">
                <label className="feedback-field">
                  <span>修改建议</span>
                  <textarea value={feedback} onChange={(event) => setFeedback(event.target.value)}
                    placeholder="描述需要调整的问题" rows={3} maxLength={MAX_FEEDBACK_LENGTH} />
                  <span className="feedback-count">{feedback.length}/{MAX_FEEDBACK_LENGTH}</span>
                </label>
                <button className="button button-secondary" disabled={submitting || !canSubmitFeedback(feedback)} onClick={recode}>
                  {stage === 'blocked' ? '提交意见并继续 Coding' : '反馈并重新 Coding'}
                </button>
              </div>
            </details>
          </div>
        )}
        {shouldShowInlineError(errorMessage, stage, task?.last_error) && (
          <div className="notice notice-error inline-error" role="alert">
            <span>{errorMessage}</span>
            <button className="button button-secondary" onClick={reload}>重新连接</button>
          </div>
        )}
        {stage === 'review' && task?.evaluation?.requirement_contract && <RequirementContract
          key={`${task.task_id}:${rounds.at(-1)?.round_no}`}
          title="查看编码依据 · 已确认需求共识" content={task.evaluation.requirement_contract} />}
        {stage === 'review' && rounds.at(-1)?.summary && rounds.at(-1)!.summary.length > 400 &&
          <details className="round-section"><summary>完整编码结果</summary><p>{rounds.at(-1)?.summary}</p></details>}
        {stage === 'review' && !!rounds.at(-1)?.tests?.length &&
          <details className="round-section"><summary>Agent 测试记录 · {rounds.at(-1)?.tests.length} 项（非 CI 状态）</summary>
            <ul>{rounds.at(-1)?.tests.map((test, index) => <li key={index}>{test}</li>)}</ul>
          </details>}
        </div>
        <aside className="workspace-environments" aria-label="部署与验证">
        {(stage === 'review' || task?.deployment) && task?.deployment_workers && (
          <details className="deployment-panel deployment-devbox" open={stage !== 'completed' || !!task.deployment?.resource_owned
            || !!task.deployment?.resource_releasing}>
            <summary className="deployment-heading">
              <span><strong>Devbox 部署与验证</strong><small>可选 · {task?.evaluation?.code_context?.repository || 'dp/aeolus'}</small></span>
              {task.deployment && <span className={`round-status round-status-${
                task.deployment.resource_releasing ? 'coding' : task.deployment.released ? 'idle'
                  : task.deployment.status === 'failed' ? 'error' : task.deployment.current ? 'completed'
                    : task.deployment.superseded || task.deployment.status === 'unknown' ? 'revision_requested' : 'coding'
              }`}>{task.deployment.status_label || task.deployment.status}</span>}
            </summary>
            <div className="stack deployment-content">
            {task.deployment && <>
              {task.deployment.error && <div className="notice notice-error deployment-error">
                <strong>部署错误</strong><span>{task.deployment.error}</span>
              </div>}
              {task.deployment.status_note && <div className={`notice ${
                task.deployment.superseded || task.deployment.status === 'unknown' ? 'notice-warning' : 'deployment-note'
              }`}><span>{task.deployment.status_note}</span></div>}
              <div className="deployment-version">
              <div className="deployment-meta"><span>Devbox</span><strong>{task.deployment.display_name}</strong></div>
              <div className="deployment-meta"><span>部署提交</span>
                <code title={task.deployment.commit_sha}>{task.deployment.commit_sha.slice(0, 12)}</code>
              </div>
              </div>
              {(task.deployment.current || (task.deployment.stage === 'maat_entry_pending'
                && task.deployment.resource_owned && !task.deployment.resource_releasing && !task.deployment.superseded))
                && safeMaatUrl(task.deployment.maat_url) && <>
                <div className="deployment-access">
                <button className="button button-primary" onClick={openEnvironment}>
                  {task.deployment.current ? '访问部署环境（MAAT）' : '打开 MAAT（入口待验证）'}
                </button>
                <small>{!task.deployment.current ? '环境已创建，入口尚未验证；不代表部署成功。'
                  : '此入口属于本次部署，环境释放后失效。'}</small>
                </div>
              </>}
              <details className="deployment-diagnostics">
                <summary>运行配置、完整提交与部署日志</summary>
                {task.deployment.runtime_profile_id && <small>运行配置：{task.deployment.runtime_profile_id}
                  {' · Python '}{task.deployment.python_version}{' · '}{task.deployment.runtime_revision}</small>}
                <small>完整提交：<code>{task.deployment.commit_sha}</code></small>
                {task.deployment.log_path && <small>日志位置：<code>{task.deployment.log_path}</code></small>}
              </details>
              {(task.deployment.can_release || task.deployment.resource_releasing) &&
                <button className="button button-release" disabled={submitting || !!task.deployment.resource_releasing}
                  onClick={releaseEnvironment}>
                  {task.deployment.resource_releasing ? '释放中…' : '释放环境（保留 Coding 任务）'}
                </button>}
            </>}
            {stage === 'review' && <details className="deployment-settings"
              key={task.deployment?.deployment_id || 'new'} open={!task.deployment || undefined}>
              <summary>{task.deployment ? '重新部署设置' : '部署设置 · 可选'}</summary>
              <div className="stack">
              <Select label="选择 Devbox" placeholder="请选择 Devbox"
                disabled={submitting || !!task.deployment?.resource_releasing
                  || ['running', 'unknown', 'cancel_requested'].includes(task.deployment?.status || '')}
                value={selectedWorker} onChange={value => {
                  setSelectedWorker(value); deploymentRequestRef.current = null;
                }} options={task.deployment_workers.map(worker => ({
                  value: worker.worker_id, disabled: !worker.online || worker.supports_repository === false,
                  label: `${worker.display_name} · ${!worker.online ? '离线/对账中'
                    : worker.supports_repository === false ? '不支持此仓库' : '在线'}`
                    + (worker.deployment_counts ? ` · 部署 ${worker.deployment_counts.total}` : ''),
                }))} />
              {deploymentWorker && <small>
                {deploymentCountLabel(deploymentWorker)}
                {deploymentWorker.deployment_counts && ` · Aeolus ${deploymentWorker.deployment_counts.by_repository['dp/aeolus'] || 0}`
                  + ` / VQS ${deploymentWorker.deployment_counts.by_repository['dp/VizQueryService'] || 0}`}
                {!deploymentWorker.online && ` · 非实时，上次统计 ${deploymentWorker.counts_observed_at || '未知'}`}
              </small>}
              {deploymentWorker && <Select label="选择运行配置"
                placeholder={Object.keys(runtimeProfiles).length ? '请选择运行配置' : '暂无可用运行配置'}
                value={selectedRuntime} disabled={submitting} onChange={value => {
                  setSelectedRuntime(value); deploymentRequestRef.current = null;
                }} options={Object.entries(runtimeProfiles).map(([id, profile]) => ({
                  value: id, disabled: profile.compatible === false,
                  label: `${id} · Python ${profile.python_version} · ${profile.runtime_revision}`
                    + (id === deploymentWorker.recommended_runtime_profile_id ? ' · 推荐' : '')
                    + (profile.compatible === false ? ' · 不兼容' : ''),
                }))} />}
              {deploymentWorker && <RuntimeEvidence detection={deploymentWorker.runtime_detection} />}
              {!task.deployment_workers.length && <span>暂无支持此仓库的 Devbox；仍可反馈或直接完成代码。</span>}
              <button className="button button-primary" onClick={deploy}
                disabled={submitting || !canDeploy(task, deploymentWorker)
                  || !selectedRuntime || !(selectedRuntime in runtimeProfiles)}>
                部署当前代码
              </button>
              </div>
            </details>}
            </div>
          </details>
        )}

        {(task?.ppe_capability?.enabled || task?.ppe_deployment) && (stage === 'review' || task?.ppe_deployment) && (
          <details className="deployment-panel deployment-ppe" open={!task.ppe_deployment?.released}>
            <summary className="deployment-heading">
              <span><strong>PPE 部署与验证</strong><small>可选 · LF / IPv6
                {(task.ppe_deployment?.python_version || task.ppe_capability?.python_version) === '2.7'
                  && ' · Python 2.7'}</small></span>
              {task.ppe_deployment && <span className={`round-status round-status-${
                task.ppe_deployment.release_requested && !task.ppe_deployment.released ? 'coding'
                  : task.ppe_deployment.released ? 'idle' : task.ppe_deployment.status === 'failed' ? 'error'
                    : task.ppe_deployment.superseded || task.ppe_deployment.status === 'unknown' ? 'revision_requested'
                      : task.ppe_deployment.status === 'deployed' ? 'completed' : 'coding'
              }`}>{task.ppe_deployment.status_label}</span>}
            </summary>
            <div className="stack deployment-content">
              {ppeNotice && <div className="notice notice-warning ppe-environment-notice" role="status">
                <strong>{ppeNotice.title}</strong>
                <span>{ppeNotice.detail}</span>
                <span>上一版就绪镜像：<b>{ppeNotice.imageVersion}</b></span>
                <span>上一版提交：<code title={ppeNotice.commitSha}>{ppeNotice.commitSha.slice(0, 12)}</code></span>
              </div>}
              {task.ppe_deployment && <>
                {task.ppe_deployment.error && <div className="notice notice-error deployment-error">
                  <strong>部署错误</strong><span>{task.ppe_deployment.error}</span>
                </div>}
                {task.ppe_deployment.superseded && !ppeNotice && <div className="notice notice-warning">
                  <span>此部署属于旧轮次，不能用于确认本轮验证通过。</span>
                </div>}
                {task.ppe_deployment.approvals?.map((approval) => <div className="notice notice-warning" key={approval.tce_ticket_id}>
                  <span>{approval.cluster_name} 等待 TCE 资源审批，工单 {approval.tce_ticket_id}；请在平台确认。</span>
                </div>)}
                <div className="deployment-version">
                  <div className="deployment-meta"><span>环境</span><code>{task.ppe_deployment.environment_name}</code></div>
                  <div className="deployment-meta"><span>{task.ppe_deployment.status === 'deployed' && !ppeNotice ? '部署提交' : '目标提交'}</span>
                    <code title={task.ppe_deployment.commit_sha}>{task.ppe_deployment.commit_sha.slice(0, 12)}</code>
                  </div>
                  <div className="deployment-meta"><span>{task.ppe_deployment.status === 'deployed' && !ppeNotice ? '部署镜像' : '目标镜像'}</span>
                    <strong>{task.ppe_deployment.image_version || '待构建'}</strong>
                  </div>
                </div>
              </>}
              {task.ppe_deployment?.can_view_environment
                && <div className="deployment-access">
                  <button className="button button-primary" onClick={() => openPpeLink(task.ppe_deployment!.environment_url)}>查看 PPE 环境</button>
                  {task.ppe_deployment.can_access_environment ? <>
                    <small>{task.ppe_deployment.access_hint}</small>
                    <small>请求头 X-TT-ENV：<code>{task.ppe_deployment.environment_name}</code></small>
                    {ppeNotice && <small>更新期间访问方式保持不变，请等待目标版本生效。</small>}
                  </> : <small>环境已创建，可查看部署进度；实例核验就绪后再进行业务验证。</small>}
                </div>}
              {stage === 'review' && task.ppe_capability?.enabled && <>
                <Select label="选择 PPE 运行配置" placeholder="请选择运行配置"
                  value={selectedPpeRuntime} disabled={submitting} onChange={setSelectedPpeRuntime}
                  options={Object.entries(ppeProfiles).map(([id, profile]) => ({
                    value: id, disabled: profile.compatible === false,
                    label: `${id} · Python ${profile.python_version}`
                      + (id === task.ppe_capability?.recommended_runtime_profile_id ? ' · 推荐' : '')
                      + (profile.compatible === false ? ' · 不兼容' : ''),
                  }))} />
                <RuntimeEvidence detection={task.ppe_capability.runtime_detection} />
                <button className="button button-primary"
                  disabled={submitting || !ppeRequest(task, false, selectedPpeRuntime)}
                  onClick={() => void deployPpe()}>一键部署当前代码到 PPE</button>
              </>}
              {task.ppe_deployment && <>
                {task.ppe_deployment.verified && <small>用户已确认业务验证通过。</small>}
                <details className="deployment-diagnostics">
                <summary>部署详情 · 构建与集群</summary>
                <div className="stack">
                <div className="deployment-actions" role="group" aria-label="PPE 构建记录">
                  {task.ppe_deployment.bits_run_url && <button className="button button-secondary"
                    onClick={() => openPpeLink(task.ppe_deployment!.bits_run_url!)}>查看 BITS 构建运行</button>}
                  <button className="button button-secondary" onClick={() => openPpeLink(task.ppe_deployment!.scm_url)}>查看 SCM 制品</button>
                  {task.ppe_deployment.image_url && <button className="button button-secondary"
                    onClick={() => openPpeLink(task.ppe_deployment!.image_url!)}>查看 ICM 镜像构建</button>}
                </div>
                {ppeNotice && <div className="deployment-meta"><span>上一版提交</span><code>{ppeNotice.commitSha}</code></div>}
                <div className="deployment-meta"><span>环境</span><code>{task.ppe_deployment.environment_name}</code></div>
                <div className="deployment-meta"><span>本次目标提交</span><code>{task.ppe_deployment.commit_sha}</code></div>
                <div className="deployment-meta"><span>本次 SCM 版本</span><strong>{task.ppe_deployment.scm_version || '待构建'}</strong></div>
                <div className="deployment-meta"><span>本次目标镜像</span><strong>{task.ppe_deployment.image_version || '待构建'}</strong></div>
                {task.ppe_deployment.ticket_id && <small>ENV 工单：{task.ppe_deployment.ticket_id}</small>}
                {task.ppe_deployment.due_time && <small>平台租期：{task.ppe_deployment.due_time}</small>}
                {task.ppe_deployment.clusters.map((cluster) => <div className="deployment-meta" key={cluster.id}>
                  <span>{cluster.psm === 'data.vqs.api' ? 'VQS · ' : cluster.psm === 'data.aeolus.api_v3' ? 'Aeolus · ' : ''}{cluster.name}</span><strong>{cluster.image_version || '版本待查询'} · {cluster.ready ? '实例已就绪' : '实例未就绪'}（最近观测）</strong>
                </div>)}
                </div>
                </details>
                <div className="deployment-actions deployment-maintenance">
                  {ppeRequest(task, true) && <button className="button button-primary"
                    disabled={submitting} onClick={() => void deployPpe(true)}>重试 PPE 部署</button>}
                  {task.ppe_deployment.can_release && <button className="button button-release"
                    disabled={submitting} onClick={releasePpe}>释放 PPE（保留 Coding 任务）</button>}
                </div>
              </>}
            </div>
          </details>
        )}

        </aside>
        </div>

        {history.order[0] === 'coding' && stage !== 'review' && stage !== 'completed'
          && task?.evaluation?.requirement_contract && (
          <RequirementContract title="查看编码依据 · 已确认需求共识" content={task.evaluation.requirement_contract} />
        )}

        {!!(rounds.length || task?.evaluation_rounds?.length) && <details className="history-disclosure" key={task?.task_id}>
          <summary>Coding 记录 · {rounds.length} 轮 / 评估记录 · {task?.evaluation_rounds?.length || 0} 轮</summary>
        {history.order.map((kind) => kind === 'evaluation' ? (
          !!task?.evaluation_rounds?.length && (
          <section className="history" aria-label="输入评估记录" key="evaluation">
            <h2 className="history-heading">评估记录 <span>{task.evaluation_rounds.length} 轮</span></h2>
            <div className="timeline">
              {[...task.evaluation_rounds].reverse().map((round) => (
                <details className="timeline-item" key={round.round_no}>
                  <summary>
                    <span className="round-heading">
                      <span>第 {round.round_no} 轮评估</span>
                      <RoundTiming round={round} timing={task.timing?.rounds?.[`evaluation:${round.round_no}`]} now={now} />
                      <span className="round-preview">{round.result?.summary || round.error || round.feedback || '正在评估需求文档'}</span>
                    </span>
                    <span className="summary-meta">
                      <span className={`round-status round-status-${round.status === 'confirmed' ? 'completed' : round.status === 'evaluating' ? 'coding' : round.status}`}>
                        {{ evaluating: '评估中', review: round.result?.valid ? '待确认' : '待补充', revision_requested: '已重新评估', confirmed: '已确认', error: '失败' }[round.status]}
                      </span>
                      <span className="disclosure" aria-hidden="true">›</span>
                    </span>
                  </summary>
                  <div className="round-content">
                    {round.feedback && <div className="round-section">
                      <strong>评估反馈</strong><span>{round.feedback}</span>
                      {round.reviewed_round_no && <small>针对第 {round.reviewed_round_no} 轮评估结果</small>}
                    </div>}
                    {!round.feedback && <p className="round-empty">{round.round_no === 1 ? '根据需求文档进行首次评估。' : '重新读取更新后的需求文档进行评估。'}</p>}
                    {round.result && <div className="round-section">
                      <strong>{round.result.valid ? '评估通过' : '评估未通过'}</strong>
                      <span>{round.result.summary}</span>
                      {round.result.requirement_contract && (
                        <RequirementContract title="查看本轮需求共识" content={round.result.requirement_contract} />
                      )}
                      {!!round.result.missing_items?.length && <div><strong>待补充信息</strong>
                        <ol className="evaluation-list">{round.result.missing_items.map((item, i) => <li key={i}>{item}</li>)}</ol>
                      </div>}
                      {!!round.result.risks?.length && <div><strong>风险提示</strong>
                        <ol className="evaluation-list">{round.result.risks.map((risk, i) => <li key={i}>{risk}</li>)}</ol>
                      </div>}
                    </div>}
                    {round.error && <p className="round-empty round-error-copy">{round.error}</p>}
                  </div>
                </details>
              ))}
            </div>
          </section>
          )
        ) : (
          renderedRounds.length > 0 && (
          <section className="history" aria-label="Coding 交互记录" key="coding">
            <h2 className="history-heading">Coding 记录 <span>{renderedRounds.length} 轮</span></h2>
            <div className="timeline">
              {renderedRounds.map((round) => (
                <details className="timeline-item" key={round.round_no}>
                  <summary>
                    <span className="round-heading">
                      <span>{round.title}</span>
                      <RoundTiming round={round} timing={task?.timing?.rounds?.[`coding:${round.round_no}`]} now={now} />
                      <span className="round-preview">{round.preview}</span>
                    </span>
                    <span className="summary-meta">
                      <span className={`round-status round-status-${round.displayStatus}`}>
                        {roundStatusLabels[round.displayStatus]}
                      </span>
                      <span className="disclosure" aria-hidden="true">›</span>
                    </span>
                  </summary>
                  <div className="round-content">
                    <div className="round-section">
                      <strong>{round.trigger_type === 'review_feedback' ? '修改要求' : '需求目标'}</strong>
                      <span>{round.feedback || '根据已通过校验的需求文档进行首次编码。'}</span>
                      {round.reviewed_round_no && (
                        <small>针对第 {round.reviewed_round_no} 轮代码结果</small>
                      )}
                    </div>
                    {round.base_commit_sha && (
                      <div className="round-section round-version">
                        <strong>基于版本</strong>
                        <code>{round.base_commit_sha.slice(0, 8)}</code>
                      </div>
                    )}
                    {round.summary && (
                      <div className="round-section">
                        <strong>处理结果</strong><span>{round.summary}</span>
                      </div>
                    )}
                    {!!round.unresolved_items?.length && <div className="round-section">
                      <strong>阻塞 / 未完成项 · {round.unresolved_items.length} 项</strong>
                      <ol>{round.unresolved_items.map((item, index) => <li key={index}>{item}</li>)}</ol>
                    </div>}
                    {round.failure_code && <div className="round-section">
                      <strong>错误码</strong><span>{round.failure_code}</span>
                    </div>}
                    {!!round.tests?.length && (
                      <details className="round-tests">
                        <summary>验证结果 · {round.tests.length} 项</summary>
                        <ul>{round.tests.map((test) => <li key={test}>{test}</li>)}</ul>
                      </details>
                    )}
                    {round.safeCodeUrl && (
                      <button className="history-code-link" onClick={() => openCode(round.safeCodeUrl)}>
                        查看本轮代码<span aria-hidden="true">↗</span>
                      </button>
                    )}
                    {round.nextRevision && (
                      <p className="revision-link">
                        用户提出修改意见，已进入第 {round.nextRevision.round_no} 轮
                      </p>
                    )}
                    {!!round.attempts?.length && (
                      round.attempts.length > 1
                      || round.attempts.some((attempt) => attempt.status === 'failed')
                    ) && (
                      <details
                        className="attempts"
                        open={round.attempts.some((attempt) => attempt.status === 'failed') || undefined}
                      >
                        <summary>执行详情 · {round.attempts.length} 次尝试</summary>
                        <ol>
                          {round.attempts.map((attempt) => (
                            <li key={attempt.attempt_no}>
                              <span>尝试 {attempt.attempt_no} · {attemptStatusLabels[attempt.status]}</span>
                              {(attempt.session_id || attempt.run_id) && (
                                <code title={`Session ${attempt.session_id} / Run ${attempt.run_id}`}>
                                  {attempt.session_id.slice(0, 8) || attempt.run_id.slice(0, 8)}
                                </code>
                              )}
                              {attempt.error && <small>{attempt.error}</small>}
                            </li>
                          ))}
                        </ol>
                      </details>
                    )}
                    {round.status === 'coding' && <p className="round-empty">本轮正在处理中。</p>}
                    {round.status === 'error' && <p className="round-empty round-error-copy">{round.error || '本轮处理失败。'}</p>}
                  </div>
                </details>
              ))}
            </div>
          </section>
          )
        ))}
        </details>}
      </section>
      {task?.task_id && <TaskIdFooter key={task.task_id} taskId={task.task_id} />}
    </main>
  );
};

export default App;
