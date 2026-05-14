/**
 * InspectorStub — Phase 2 placeholder for the right-hand properties panel.
 * Phase 3 replaces this with the full PropertiesPanel (3 tabs).
 */
import { Info } from 'lucide-react';
import type { BrainstormTree } from '../../types';
import { useWorkspaceStore } from '../../stores/workspaceStore';
import { depthHue } from '../../styles/motion';

interface Props { tree: BrainstormTree }

export function InspectorStub({ tree }: Props) {
  const selectedIds = useWorkspaceStore((s) => s.selectedNodeIds);
  const node = selectedIds.length === 1 ? tree.nodes[selectedIds[0]] : null;

  return (
    <div className="h-full flex flex-col bg-paper">
      <div className="h-9 shrink-0 flex items-center justify-between px-3 border-b border-[var(--ink-line)]">
        <div className="flex items-center gap-1.5">
          <Info size={12} className="text-ink-2" />
          <span className="text-[11px] uppercase tracking-[0.16em] text-ink-2 font-medium">Inspector</span>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-4">
        {!node && selectedIds.length === 0 && (
          <p className="text-[12px] text-ink-3 leading-relaxed">
            Select a node on the canvas or layers panel to inspect its content, metrics, and history.
          </p>
        )}
        {!node && selectedIds.length > 1 && (
          <p className="text-[12px] text-ink-2 leading-relaxed">
            {selectedIds.length} nodes selected · bulk actions coming soon.
          </p>
        )}
        {node && (
          <div className="space-y-4" style={{ '--gen-hue': depthHue(node.depth) } as React.CSSProperties}>
            <div>
              <div className="text-[10px] uppercase tracking-[0.16em] text-ink-3 mb-1.5">Type · depth</div>
              <div className="flex items-center gap-2">
                <span className="h-1.5 w-1.5 rounded-full" style={{ background: 'var(--gen)' }} />
                <span className="text-[12px] text-ink-1 capitalize">{node.type}</span>
                <span className="text-ink-3">·</span>
                <span className="text-[12px] text-ink-1 tabular-nums">d{node.depth}</span>
              </div>
            </div>

            <div>
              <div className="text-[10px] uppercase tracking-[0.16em] text-ink-3 mb-1.5">Summary</div>
              <div className="text-[13.5px] text-ink-0 font-display leading-snug" style={{ fontVariationSettings: '"wght" 500, "opsz" 24' }}>
                {node.summary || '—'}
              </div>
            </div>

            {node.content && node.content !== node.summary && (
              <div>
                <div className="text-[10px] uppercase tracking-[0.16em] text-ink-3 mb-1.5">Content</div>
                <p className="text-[12.5px] text-ink-1 leading-relaxed whitespace-pre-wrap">{node.content}</p>
              </div>
            )}

            <div className="grid grid-cols-2 gap-3 pt-2 border-t border-[var(--ink-line)]">
              <div>
                <div className="text-[10px] uppercase tracking-[0.16em] text-ink-3 mb-1">Children</div>
                <div className="text-[14px] tabular-nums text-ink-0">{node.children.length}</div>
              </div>
              <div>
                <div className="text-[10px] uppercase tracking-[0.16em] text-ink-3 mb-1">Expanded</div>
                <div className="text-[14px] text-ink-0">{node.isExpanded ? 'Yes' : 'No'}</div>
              </div>
              {node.feedbackGiven && (
                <div>
                  <div className="text-[10px] uppercase tracking-[0.16em] text-ink-3 mb-1">Feedback</div>
                  <div className="text-[14px] text-ink-0 capitalize">{node.feedbackGiven}</div>
                </div>
              )}
              {node.trajectoryId && (
                <div className="col-span-2">
                  <div className="text-[10px] uppercase tracking-[0.16em] text-ink-3 mb-1">Trajectory ID</div>
                  <div className="text-[11px] font-mono text-ink-1 break-all">{node.trajectoryId}</div>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
