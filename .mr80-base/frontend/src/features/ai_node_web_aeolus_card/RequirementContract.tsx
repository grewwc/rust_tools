import { lazy, Suspense, useState } from 'react';

const Markdown = lazy(() => import('./Markdown.mjs'));

export default function RequirementContract({ title, content }: { title: string; content: string }) {
  const [open, setOpen] = useState(false);

  return (
    <details className="round-section" onToggle={(event) => setOpen(event.currentTarget.open)}>
      <summary>{title}</summary>
      {open && (
        <Suspense fallback={<p>正在加载需求共识…</p>}>
          <Markdown content={content} />
        </Suspense>
      )}
    </details>
  );
}
