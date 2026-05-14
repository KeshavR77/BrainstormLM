/**
 * PropertiesPanel — three-tab inspector for the selected node.
 *
 * Tabs: Content (editable), Metrics (depth, diversity placeholder), History.
 * Content edits debounce-save back through onEditNode after 500ms.
 */
import { useEffect, useRef, useState } from 'react';
import { motion } from 'framer-motion';
import { Info, FileText, BarChart3, Clock, RefreshCw, ThumbsUp, ThumbsDown } from 'lucide-react';
import type { BrainstormTree } from '../../types';
import { useWorkspaceStore } from '../../stores/workspaceStore';
import { depthHue } from '../../styles/motion';

type Tab = 'content' | 'metrics' | 'history';

interface Props {
  tree: BrainstormTree;
  onEditNode?: (id: string, content: string, summary?: string) => void;
  onRegenerate?: (id: string) => void;
  onFeedback?: (id: string, thumbs: 'up' | 'down') => void;
}

export function PropertiesPanel({ tree, onEditNode, onRegenerate, onFeedback }: Props) {
  const selectedIds = useWorkspaceStore((s) => s.selectedNodeIds);
  const [tab, setTab] = useState<Tab>('content');
  const node = selectedIds.length === 1 ? tree.nodes[selectedIds[0]] : null;

  /* Local editable buffer with debounce */
  const [summary, setSummary] = useState(node?.summary ?? '');
  const [content, setContent] = useState(node?.content ?? '');
  const saveT = useRef<number | null>(null);

  useEffect(() => {
    setSummary(node?.summary ?? '');
    setContent(node?.content ?? '');
  }, [node?.id, node?.summary, node?.content]);

  const queueSave = (newContent: string, newSummary?: string) => {
    if (!node) return;
    if (saveT.current) window.clearTimeout(saveT.current);
    saveT.current = window.setTimeout(() => {
      onEditNode?.(node.id, newContent, newSummary);
    }, 500);
  };

  /* Sibling diversity placeholder (proper Vendi calc lands with the eval pipeline) */
  const siblings = node?.parentId ? tree.nodes[node.parentId]?.children ?? [] : [];
  const descendants = node ? countDescendants(tree, node.id) : 0;

  return (
    <div className="h-full flex flex-col bg-paper">
      {/* Header */}
      <div className="h-9 shrink-0 flex items-center justify-between px-3 border-b border-[var(--ink-line)]">
        <div className="flex items-center gap-1.5">
          <Info size={12} className="text-ink-2" />
          <span className="text-[11px] uppercase tracking-[0.16em] text-ink-2 font-medium">Inspector</span>
        </div>
        {node && (
          <span className="text-[10px] text-ink-3 font-mono">{node.id.slice(0, 8)}</span>
        )}
      </div>

      {!node ? (
        <div className="flex-1 p-4">
          <p className="text-[12px] text-ink-3 leading-relaxed">
            {selectedIds.length === 0
              ? 'Select a node on the canvas or layers panel to inspect its content, metrics, and history.'
              : `${selectedIds.length} nodes selected · bulk actions coming soon.`}
          </p>
        </div>
      ) : (
        <div className="flex-1 flex flex-col min-h-0" style={{ '--gen-hue': depthHue(node.depth) } as React.CSSProperties}>
          {/* Tabs */}
          <div className="flex border-b border-[var(--ink-line)]">
            {([
              { id: 'content', label: 'Content', Icon: FileText },
              { id: 'metrics', label: 'Metrics', Icon: BarChart3 },
              { id: 'history', label: 'History', Icon: Clock },
            ] as const).map(({ id, label, Icon }) => {
              const active = tab === id;
              return (
                <button
                  key={id}
                  onClick={() => setTab(id)}
                  className={`flex-1 inline-flex items-center justify-center gap-1.5 py-2 text-[11.5px] transition-colors relative ${
                    active ? 'text-ink-0' : 'text-ink-2 hover:text-ink-1'
                  }`}
                >
                  <Icon size={11} /> {label}
                  {active && (
                    <motion.span
                      layoutId="inspector-tab-underline"
                      className="absolute bottom-0 left-2 right-2 h-px bg-ink-0"
                    />
                  )}
                </button>
              );
            })}
          </div>

          {/* Body */}
          <div className="flex-1 overflow-y-auto p-4">
            {tab === 'content' && (
              <div className="space-y-4">
                <div>
                  <label className="text-[10px] uppercase tracking-[0.16em] text-ink-3 mb-1.5 block">Summary</label>
                  <input
                    value={summary}
                    onChange={(e) => { setSummary(e.target.value); queueSave(content, e.target.value); }}
                    className="w-full px-3 py-2 rounded-md bg-paper-1 border border-[var(--ink-line)] text-[13px] text-ink-0 outline-none focus:border-brand focus:ring-2 focus:ring-brand/15 transition-all"
                  />
                </div>
                <div>
                  <label className="text-[10px] uppercase tracking-[0.16em] text-ink-3 mb-1.5 block">Content</label>
                  <textarea
                    rows={8}
                    value={content}
                    onChange={(e) => { setContent(e.target.value); queueSave(e.target.value); }}
                    className="w-full px-3 py-2 rounded-md bg-paper-1 border border-[var(--ink-line)] text-[13px] text-ink-0 outline-none focus:border-brand focus:ring-2 focus:ring-brand/15 transition-all leading-relaxed resize-none"
                  />
                  <div className="mt-1.5 text-[10.5px] text-ink-3">Autosaves 500ms after you stop typing.</div>
                </div>
                <div className="flex items-center gap-2 pt-3 border-t border-[var(--ink-line)]">
                  <button
                    onClick={() => onRegenerate?.(node.id)}
                    className="inline-flex items-center gap-1.5 text-[12px] px-3 py-1.5 rounded-full bg-paper-1 border border-[var(--ink-line)] text-ink-1 hover:text-ink-0 hover:bg-paper-2 transition-colors"
                  >
                    <RefreshCw size={11} /> Regenerate
                  </button>
                  <button
                    onClick={() => onFeedback?.(node.id, 'up')}
                    disabled={node.feedbackGiven === 'up'}
                    className={`inline-flex items-center gap-1 text-[12px] px-2.5 py-1.5 rounded-full border transition-colors ${
                      node.feedbackGiven === 'up'
                        ? 'bg-brand/10 border-brand/30 text-brand cursor-not-allowed'
                        : 'bg-paper-1 border-[var(--ink-line)] text-ink-1 hover:text-ink-0 hover:bg-paper-2'
                    }`}
                  >
                    <ThumbsUp size={11} />
                  </button>
                  <button
                    onClick={() => onFeedback?.(node.id, 'down')}
                    disabled={node.feedbackGiven === 'down'}
                    className={`inline-flex items-center gap-1 text-[12px] px-2.5 py-1.5 rounded-full border transition-colors ${
                      node.feedbackGiven === 'down'
                        ? 'bg-[var(--accent-red)]/10 border-[var(--accent-red)]/30 text-[var(--accent-red)] cursor-not-allowed'
                        : 'bg-paper-1 border-[var(--ink-line)] text-ink-1 hover:text-ink-0 hover:bg-paper-2'
                    }`}
                  >
                    <ThumbsDown size={11} />
                  </button>
                </div>
              </div>
            )}

            {tab === 'metrics' && (
              <div className="space-y-4">
                <MetricRow label="Depth"          value={`d${node.depth}`} hue={depthHue(node.depth)} />
                <MetricRow label="Children"       value={String(node.children.length)} />
                <MetricRow label="Descendants"    value={String(descendants)} />
                <MetricRow label="Siblings"       value={String(Math.max(0, siblings.length - 1))} />
                <MetricRow label="Expanded"       value={node.isExpanded ? 'Yes' : 'No'} />
                <MetricRow label="Type"           value={node.type} capitalize />
                <div className="pt-3 border-t border-[var(--ink-line)]">
                  <div className="text-[10px] uppercase tracking-[0.16em] text-ink-3 mb-2">Diversity vs siblings</div>
                  <div className="rounded-md bg-paper-1 border border-[var(--ink-line)] p-3">
                    <div className="text-[12px] text-ink-2 leading-relaxed">
                      Vendi score & near-duplicate signal land with the evaluator wiring in the next round.
                    </div>
                  </div>
                </div>
              </div>
            )}

            {tab === 'history' && (
              <div className="space-y-3">
                <div className="text-[10px] uppercase tracking-[0.16em] text-ink-3">Lifecycle</div>
                <HistoryRow when={node.createdAt} label="Created" />
                {node.feedbackGiven && (
                  <HistoryRow
                    when={node.createdAt}
                    label={`Feedback · ${node.feedbackGiven}`}
                    tint={node.feedbackGiven === 'up' ? 'var(--brand)' : 'var(--accent-red)'}
                  />
                )}
                {node.trajectoryId && (
                  <div className="pt-3 mt-3 border-t border-[var(--ink-line)]">
                    <div className="text-[10px] uppercase tracking-[0.16em] text-ink-3 mb-1.5">Trajectory ID</div>
                    <div className="text-[11px] font-mono text-ink-1 break-all bg-paper-1 border border-[var(--ink-line)] rounded-md p-2.5">
                      {node.trajectoryId}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

function MetricRow({ label, value, hue, capitalize }: { label: string; value: string; hue?: number; capitalize?: boolean }) {
  return (
    <div className="flex items-center justify-between py-1.5 border-b border-[var(--ink-line-soft)]">
      <span className="text-[10.5px] uppercase tracking-[0.16em] text-ink-3">{label}</span>
      <div className="flex items-center gap-2">
        {hue !== undefined && (
          <span className="h-1.5 w-1.5 rounded-full" style={{ background: `oklch(0.62 0.20 ${hue})` }} />
        )}
        <span className={`text-[13px] tabular-nums text-ink-0 ${capitalize ? 'capitalize' : ''}`}>{value}</span>
      </div>
    </div>
  );
}

function HistoryRow({ when, label, tint }: { when: string; label: string; tint?: string }) {
  const d = new Date(when);
  return (
    <div className="flex items-center gap-2.5 text-[12px]">
      <span className="h-1.5 w-1.5 rounded-full shrink-0" style={{ background: tint ?? 'var(--ink-3)' }} />
      <span className="text-ink-1">{label}</span>
      <span className="ml-auto text-ink-3 tabular-nums text-[11px]">
        {d.toLocaleString('en-US', { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' })}
      </span>
    </div>
  );
}

function countDescendants(tree: BrainstormTree, id: string): number {
  let n = 0;
  const stack = [...(tree.nodes[id]?.children ?? [])];
  while (stack.length) {
    const cid = stack.pop()!;
    n++;
    const c = tree.nodes[cid];
    if (c) stack.push(...c.children);
  }
  return n;
}
