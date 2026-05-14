/**
 * HistoryDrawer — slide-in right drawer listing named snapshots.
 *
 * Click to restore. Magenta highlight on hover. New snapshot via input
 * at top; defaults to "Snapshot @ <time>" when empty.
 */
import { AnimatePresence, motion } from 'framer-motion';
import { useState } from 'react';
import { Clock, X, Camera, RotateCcw, Trash2, GitBranch } from 'lucide-react';
import type { Snapshot } from '../../hooks/useSnapshots';

interface Props {
  open: boolean;
  onClose: () => void;
  snapshots: Snapshot[];
  onCreate: (label: string) => void;
  onRestore: (id: string) => void;
  onDelete: (id: string) => void;
}

export function HistoryDrawer({ open, onClose, snapshots, onCreate, onRestore, onDelete }: Props) {
  const [label, setLabel] = useState('');

  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    onCreate(label);
    setLabel('');
  };

  return (
    <AnimatePresence>
      {open && (
        <>
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            className="fixed inset-0 z-40 bg-ink-0/20 backdrop-blur-[2px]"
          />
          <motion.aside
            initial={{ x: '100%' }}
            animate={{ x: 0 }}
            exit={{ x: '100%' }}
            transition={{ duration: 0.28, ease: [0.22, 1, 0.36, 1] }}
            className="fixed top-0 right-0 bottom-0 z-50 w-[360px] bg-paper border-l border-[var(--ink-line-bold)] shadow-lift flex flex-col"
          >
            <div className="flex items-center justify-between px-4 py-3 border-b border-[var(--ink-line)]">
              <div className="flex items-center gap-2">
                <Clock size={14} className="text-ink-2" />
                <h2
                  className="font-display text-ink-0 text-[16px] tracking-tight"
                  style={{ fontVariationSettings: '"wght" 520, "opsz" 24' }}
                >
                  Version history
                </h2>
              </div>
              <button onClick={onClose} className="text-ink-2 hover:text-ink-0 p-1 rounded hover:bg-paper-2">
                <X size={15} />
              </button>
            </div>

            {/* Create new snapshot */}
            <form onSubmit={submit} className="p-4 border-b border-[var(--ink-line)] flex gap-2">
              <input
                value={label}
                onChange={(e) => setLabel(e.target.value)}
                placeholder="Name this snapshot…"
                className="flex-1 px-3 py-1.5 rounded-md bg-paper-1 border border-[var(--ink-line)] text-[12.5px] text-ink-0 placeholder:text-ink-3 outline-none focus:border-brand focus:ring-2 focus:ring-brand/15"
              />
              <button
                type="submit"
                className="inline-flex items-center gap-1 bg-brand hover:bg-brand-deep text-white text-[12px] font-medium px-3 py-1.5 rounded-md transition-colors"
              >
                <Camera size={11} /> Save
              </button>
            </form>

            {/* Snapshots list */}
            <div className="flex-1 overflow-y-auto p-2">
              {snapshots.length === 0 ? (
                <div className="px-4 py-12 text-center text-[12px] text-ink-3">
                  No snapshots yet. Save one with <kbd className="text-ink-1">⌘ S</kbd>.
                </div>
              ) : (
                snapshots.map((s, i) => (
                  <div
                    key={s.id}
                    className="group p-3 rounded-md hover:bg-paper-2 transition-colors border-l-2"
                    style={{ borderLeftColor: i === 0 ? 'var(--brand)' : 'transparent' }}
                  >
                    <div className="flex items-start justify-between gap-2">
                      <div className="min-w-0 flex-1">
                        <div className="text-[13px] text-ink-0 truncate">{s.label}</div>
                        <div className="text-[11px] text-ink-2 mt-0.5">
                          {new Date(s.createdAt).toLocaleString('en-US', { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' })}
                          <span className="text-ink-3 mx-1.5">·</span>
                          <span className="inline-flex items-center gap-0.5"><GitBranch size={10} /> {s.nodeCount}</span>
                        </div>
                      </div>
                      <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity shrink-0">
                        <button
                          onClick={() => onRestore(s.id)}
                          title="Restore this snapshot"
                          className="p-1.5 rounded hover:bg-paper-3 text-ink-1 hover:text-ink-0"
                        >
                          <RotateCcw size={12} />
                        </button>
                        <button
                          onClick={() => { if (window.confirm(`Delete snapshot "${s.label}"?`)) onDelete(s.id); }}
                          title="Delete"
                          className="p-1.5 rounded hover:bg-paper-3 text-ink-1 hover:text-[var(--accent-red)]"
                        >
                          <Trash2 size={12} />
                        </button>
                      </div>
                    </div>
                  </div>
                ))
              )}
            </div>

            <div className="px-4 py-2 border-t border-[var(--ink-line)] text-[10.5px] text-ink-3">
              Capped at 20 · compressed in localStorage
            </div>
          </motion.aside>
        </>
      )}
    </AnimatePresence>
  );
}
