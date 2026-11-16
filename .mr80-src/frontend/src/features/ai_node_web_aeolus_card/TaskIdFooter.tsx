import { useState } from 'react';

async function copyText(text: string) {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    // HTTP embeds or iframe permissions may make the Clipboard API unavailable.
  }
  const focused = document.activeElement;
  const input = document.createElement('textarea');
  input.value = text;
  input.readOnly = true;
  input.style.cssText = 'position:fixed;left:-9999px;top:0';
  document.body.appendChild(input);
  try {
    input.select();
    return document.execCommand('copy');
  } catch {
    return false;
  } finally {
    input.remove();
    if (focused instanceof HTMLElement) focused.focus({ preventScroll: true });
  }
}

export default function TaskIdFooter({ taskId }: { taskId: string }) {
  const [status, setStatus] = useState<'idle' | 'copying' | 'copied' | 'failed'>('idle');

  const copy = async () => {
    setStatus('copying');
    setStatus(await copyText(taskId) ? 'copied' : 'failed');
  };

  return (
    <footer className="task-id-footer" aria-label="任务信息">
      <span>任务 ID</span>
      <code>{taskId}</code>
      <button type="button" className="task-id-copy" aria-label="复制任务 ID"
        disabled={status === 'copying'} onClick={copy}>
        {status === 'copied' ? '已复制' : status === 'copying' ? '复制中…' : '复制'}
      </button>
      <span className="task-id-copy-status" role="status">
        {status === 'copied' ? '任务 ID 已复制' : status === 'failed' ? '复制失败，请选中任务 ID 手动复制。' : ''}
      </span>
    </footer>
  );
}
