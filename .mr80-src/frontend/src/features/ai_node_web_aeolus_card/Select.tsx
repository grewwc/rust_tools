import React, { useEffect, useId, useLayoutEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';

type Option = { value: string; label: string; disabled?: boolean };

export default function Select({ label, placeholder, value, options, disabled, onChange }: {
  label: string; placeholder: string; value: string; options: Option[];
  disabled?: boolean; onChange: (value: string) => void;
}) {
  const id = useId();
  const trigger = useRef<HTMLButtonElement>(null);
  const menu = useRef<HTMLDivElement>(null);
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState('');
  const [position, setPosition] = useState<React.CSSProperties>({});
  const selected = options.find(option => option.value === value && !option.disabled);
  const enabled = options.filter(option => !option.disabled);
  const optionKey = JSON.stringify(options);

  useLayoutEffect(() => {
    if (!open) return;
    const place = () => {
      const rect = trigger.current?.getBoundingClientRect();
      if (!rect) return;
      const below = window.innerHeight - rect.bottom - 12;
      const above = rect.top - 12;
      const up = below < 160 && above > below;
      const width = Math.min(Math.max(rect.width, 220), window.innerWidth - 16);
      setPosition({
        left: Math.max(8, Math.min(rect.left, window.innerWidth - width - 8)),
        width,
        maxHeight: Math.max(40, Math.min(280, up ? above : below)),
        ...(up ? { bottom: window.innerHeight - rect.top + 4 } : { top: rect.bottom + 4 }),
      });
    };
    place();
    window.addEventListener('resize', place);
    window.addEventListener('scroll', place, true);
    return () => {
      window.removeEventListener('resize', place);
      window.removeEventListener('scroll', place, true);
    };
  }, [open, optionKey]);

  useEffect(() => {
    if (disabled) setOpen(false);
    if (!enabled.some(option => option.value === active)) setActive(enabled[0]?.value || '');
  }, [disabled, optionKey, active]);

  useEffect(() => {
    if (!open) return;
    const close = (event: PointerEvent) => {
      if (!trigger.current?.contains(event.target as Node) && !menu.current?.contains(event.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener('pointerdown', close);
    return () => document.removeEventListener('pointerdown', close);
  }, [open]);

  useEffect(() => {
    if (open) document.getElementById(`${id}-${active}`)?.scrollIntoView({ block: 'nearest' });
  }, [open, active, id]);

  const choose = (next: string) => {
    if (!enabled.some(option => option.value === next)) return;
    onChange(next);
    setOpen(false);
    trigger.current?.focus();
  };

  return <div className="select-field">
    <button ref={trigger} type="button" className="select-trigger" role="combobox"
      aria-label={label} aria-expanded={open} aria-controls={open ? id : undefined}
      aria-haspopup="listbox" aria-activedescendant={open && active ? `${id}-${active}` : undefined}
      disabled={disabled || !enabled.length}
      onBlur={event => { if (!menu.current?.contains(event.relatedTarget as Node)) setOpen(false); }}
      onClick={() => { setActive(selected?.value || enabled[0]?.value || ''); setOpen(!open); }}
      onKeyDown={event => {
        if (event.key === 'Escape' || event.key === 'Tab') { setOpen(false); return; }
        if (!['ArrowDown', 'ArrowUp', 'Home', 'End', 'Enter', ' '].includes(event.key)) return;
        event.preventDefault();
        if (event.key === 'Enter' || event.key === ' ') {
          if (open) choose(active);
          else { setActive(selected?.value || enabled[0]?.value || ''); setOpen(true); }
          return;
        }
        const index = enabled.findIndex(option => option.value === active);
        const next = event.key === 'Home' ? 0 : event.key === 'End' ? enabled.length - 1
          : !open ? Math.max(0, enabled.findIndex(option => option.value === value))
            : (index + (event.key === 'ArrowDown' ? 1 : -1) + enabled.length) % enabled.length;
        setActive(enabled[next]?.value || '');
        setOpen(true);
      }}>
      <span className={selected ? '' : 'select-placeholder'}>{selected?.label || placeholder}</span>
      <span aria-hidden="true" className="select-arrow">⌄</span>
    </button>
    {open && createPortal(<div ref={menu} id={id} className="select-menu"
      role="listbox" aria-label={label} style={position}>
      {options.map(option => <div key={option.value} id={`${id}-${option.value}`} role="option"
        aria-selected={selected?.value === option.value} aria-disabled={!!option.disabled}
        className={`select-option${active === option.value ? ' is-active' : ''}`}
        onPointerDown={event => event.preventDefault()}
        onPointerMove={() => { if (!option.disabled) setActive(option.value); }}
        onClick={() => choose(option.value)}>
        <span>{option.label}</span><span aria-hidden="true">{selected?.value === option.value ? '✓' : ''}</span>
      </div>)}
    </div>, document.body)}
  </div>;
}
