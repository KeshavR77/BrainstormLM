/**
 * CommentsPanel — slide-in left drawer listing comment threads for the project.
 *
 * Click thread → selects the node it's pinned to.
 */
import { AnimatePresence, motion } from 'framer-motion';
import { useState } from 'react';
import { MessageSquare, X, Check, Trash2, Send } from 'lucide-react';
import type { Thread } from '../../hooks/useComments';
import type { BrainstormTree } from '../../types';
import { useWorkspaceStore } from '../../stores/workspaceStore';

interface Props {
  open: boolean;
  onClose: () => void;
  tree: BrainstormTree;
  threads: Thread[];
  onAdd: (nodeId: string, body: string) => void;
  onResolve: (nodeId: string) => void;
  onDelete: (nodeId: string) => void;
  onFocusNode: (id: string) => void;
}

export function CommentsPanel({ open, onClose, tree, threads, onAdd, onResolve, onDelete, onFocusNode }: Props) {
  const selectedIds = useWorkspaceStore((s) => s.selectedNodeIds);
  const setSelection = useWorkspaceStore((s) => s.setSelection);
  const activeNodeId = selectedIds[0];
  const [draft, setDraft] = useState('');

  const activeThread = threads.find((t) => t.nodeId === activeNodeId);
  const activeNode = activeNodeId ? tree.nodes[activeNodeId] : null;

  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!activeNodeId || !draft.trim()) return;
    onAdd(activeNodeId, draft);
    setDraft('');
  };

  return (
    <AnimatePresence>
      {open && (
        <>
          <motion.div
            initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
            onClick={onClose}
            className="fixed inset-0 z-40 bg-ink-0/20 backdrop-blur-[2px]"
          />
          <motion.aside
            initial={{ x: '-100%' }} animate={{ x: 0 }} exit={{ x: '-100%' }}
            transition={{ duration: 0.28, ease: [0.22, 1, 0.36, 1] }}
            className="fixed top-0 left-0 bottom-0 z-50 w-[360px] bg-paper border-r border-[var(--ink-line-bold)] shadow-lift flex flex-col"
          >
            <div className="flex items-center justify-between px-4 py-3 border-b border-[var(--ink-line)]">
              <div className="flex items-center gap-2">
                <MessageSquare size={14} className="text-ink-2" />
                <h2 className="font-display text-ink-0 text-[16px] tracking-tight" style={{ fontVariationSettings: '"wght" 520, "opsz" 24' }}>
                  Comments
                </h2>
              </div>
              <button onClick={onClose} className="text-ink-2 hover:text-ink-0 p-1 rounded hover:bg-paper-2"><X size={15} /></button>
            </div>

            {/* Active thread editor */}
            {activeNode ? (
              <div className="border-b border-[var(--ink-line)] p-4 space-y-3">
                <div className="text-[10px] uppercase tracking-[0.16em] text-ink-3">Pinned to</div>
                <div className="text-[12.5px] text-ink-0 truncate font-medium">
                  {activeNode.summary || activeNode.content.slice(0, 60)}
                </div>
                {activeThread && activeThread.comments.length > 0 && (
                  <div className="space-y-2 max-h-[260px] overflow-y-auto -mx-2 px-2">
                    {activeThread.comments.map((c) => (
                      <div key={c.id} className="rounded-md bg-paper-1 border border-[var(--ink-line)] p-2.5">
                        <div className="flex items-center gap-2 mb-1">
                          <span className="h-4 w-4 rounded-full bg-brand text-white text-[8px] flex items-center justify-center font-medium">
                            {c.author[0]}
                          </span>
                          <span className="text-[11px] text-ink-1">{c.author}</span>
                          <span className="ml-auto text-[10px] text-ink-3 tabular-nums">
                            {new Date(c.createdAt).toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit' })}
                          </span>
                        </div>
                        <p className="text-[12.5px] text-ink-0 leading-relaxed whitespace-pre-wrap">{c.body}</p>
                      </div>
                    ))}
                  </div>
                )}
                <form onSubmit={submit} className="flex gap-2">
                  <input
                    value={draft}
                    onChange={(e) => setDraft(e.target.value)}
                    placeholder="Add a comment…"
                    className="flex-1 px-3 py-2 rounded-md bg-paper-1 border border-[var(--ink-line)] text-[12.5px] text-ink-0 placeholder:text-ink-3 outline-none focus:border-brand focus:ring-2 focus:ring-brand/15"
                  />
                  <button
                    type="submit"
                    disabled={!draft.trim()}
                    className="bg-brand hover:bg-brand-deep disabled:opacity-40 text-white text-[12px] font-medium px-3 py-2 rounded-md transition-colors inline-flex items-center gap-1"
                  >
                    <Send size={11} />
                  </button>
                </form>
                {activeThread && (
                  <div className="flex items-center gap-2 pt-1">
                    <button
                      onClick={() => onResolve(activeNodeId!)}
                      className={`inline-flex items-center gap-1 text-[11px] px-2 py-1 rounded-full transition-colors ${
                        activeThread.resolved
                          ? 'bg-brand/10 text-brand border border-brand/30'
                          : 'border border-[var(--ink-line)] text-ink-1 hover:text-ink-0 hover:bg-paper-2'
                      }`}
                    >
                      <Check size={10} /> {activeThread.resolved ? 'Resolved' : 'Mark resolved'}
                    </button>
                    <button
                      onClick={() => { if (window.confirm('Delete thread?')) onDelete(activeNodeId!); }}
                      className="inline-flex items-center gap-1 text-[11px] px-2 py-1 rounded-full border border-[var(--ink-line)] text-ink-1 hover:text-[var(--accent-red)] hover:bg-paper-2 transition-colors"
                    >
                      <Trash2 size={10} />
                    </button>
                  </div>
                )}
              </div>
            ) : (
              <div className="p-4 text-[12px] text-ink-3">Select a node on the canvas to add a comment.</div>
            )}

            {/* All threads */}
            <div className="flex-1 overflow-y-auto">
              <div className="px-4 py-2 text-[10px] uppercase tracking-[0.18em] text-ink-3">All threads</div>
              {threads.length === 0 ? (
                <div className="px-4 py-6 text-[12px] text-ink-3">No threads yet.</div>
              ) : (
                threads.map((t) => {
                  const node = tree.nodes[t.nodeId];
                  if (!node) return null;
                  return (
                    <button
                      key={t.nodeId}
                      onClick={() => { setSelection([t.nodeId]); onFocusNode(t.nodeId); }}
                      className="w-full text-left p-3 hover:bg-paper-2 transition-colors border-l-2"
                      style={{ borderLeftColor: t.resolved ? 'var(--paper-3)' : 'var(--brand)' }}
                    >
                      <div className="flex items-center justify-between gap-2 mb-1">
                        <div className="text-[12.5px] text-ink-0 truncate font-medium">
                          {node.summary || node.content.slice(0, 50)}
                        </div>
                        {t.resolved && (
                          <span className="text-[9px] uppercase tracking-[0.12em] text-ink-3">Resolved</span>
                        )}
                      </div>
                      <div className="text-[11px] text-ink-2 truncate">
                        {t.comments.length} {t.comments.length === 1 ? 'reply' : 'replies'}
                        {t.comments.length > 0 && ` · "${t.comments[t.comments.length - 1].body.slice(0, 60)}"`}
                      </div>
                    </button>
                  );
                })
              )}
            </div>
          </motion.aside>
        </>
      )}
    </AnimatePresence>
  );
}
