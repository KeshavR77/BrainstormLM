/**
 * LayersPanel — recursive tree mirror of the brainstorm canvas.
 *
 * Click a row → focus that node on the canvas (via fitView in Phase 3).
 * Hover → highlight on canvas. Right-click → context menu (Phase 5).
 * Left border on each row is colored by depth via OKLCH ramp.
 */
import { ChevronRight, FileText, Layers, Sparkles, Star } from 'lucide-react';
import { useMemo, useState } from 'react';
import type { BrainstormTree, BrainstormNode } from '../../types';
import { useWorkspaceStore } from '../../stores/workspaceStore';
import { depthHue } from '../../styles/motion';

interface Props {
  tree: BrainstormTree;
  onFocusNode?: (id: string) => void;
}

interface RowProps {
  node: BrainstormNode;
  tree: BrainstormTree;
  level: number;
  collapsedIds: Set<string>;
  setCollapsed: (id: string, c: boolean) => void;
  onFocusNode?: (id: string) => void;
}

function Row({ node, tree, level, collapsedIds, setCollapsed, onFocusNode }: RowProps) {
  const selected = useWorkspaceStore((s) => s.selectedNodeIds.includes(node.id));
  const setSelection = useWorkspaceStore((s) => s.setSelection);
  const setHovered   = useWorkspaceStore((s) => s.setHovered);

  const hasChildren = node.children.length > 0;
  const collapsed   = collapsedIds.has(node.id);

  const Icon =
    node.type === 'seed' ? FileText :
    node.type === 'selected' ? Star :
    hasChildren ? Sparkles : FileText;

  return (
    <>
      <button
        onClick={(e) => {
          if (e.shiftKey) {
            // toggle add — handled by parent later
          } else {
            setSelection([node.id]);
          }
          onFocusNode?.(node.id);
        }}
        onMouseEnter={() => setHovered(node.id)}
        onMouseLeave={() => setHovered(null)}
        className={`group relative flex items-start gap-1.5 w-full px-2 py-1.5 text-left text-[12px] rounded-md transition-colors ${
          selected ? 'bg-brand/10 text-ink-0' : 'text-ink-1 hover:bg-paper-2'
        }`}
        style={{ paddingLeft: 8 + level * 14 }}
      >
        {/* Depth accent stripe */}
        <span
          aria-hidden
          className="absolute left-0 top-1.5 bottom-1.5 w-[2px] rounded-full"
          style={{ background: `oklch(0.62 0.20 ${depthHue(node.depth)})`, opacity: selected ? 1 : 0.55 }}
        />
        {/* Chevron — span with role=button to avoid nested <button> */}
        {hasChildren ? (
          <span
            role="button"
            tabIndex={-1}
            onClick={(e) => { e.stopPropagation(); setCollapsed(node.id, !collapsed); }}
            className="mt-0.5 shrink-0 text-ink-2 hover:text-ink-0 cursor-pointer"
          >
            <ChevronRight size={11} className={`transition-transform ${collapsed ? '' : 'rotate-90'}`} />
          </span>
        ) : (
          <span className="w-[11px] shrink-0" />
        )}
        <Icon size={11} className="mt-0.5 shrink-0 text-ink-2" />
        <span className="truncate flex-1">{node.summary || node.content.slice(0, 50)}</span>
        {node.feedbackGiven === 'up' && (
          <span className="text-[8px] text-brand shrink-0 mt-0.5">●</span>
        )}
      </button>

      {hasChildren && !collapsed && node.children.map((cid) => {
        const child = tree.nodes[cid];
        if (!child) return null;
        return (
          <Row
            key={cid}
            node={child}
            tree={tree}
            level={level + 1}
            collapsedIds={collapsedIds}
            setCollapsed={setCollapsed}
            onFocusNode={onFocusNode}
          />
        );
      })}
    </>
  );
}

export function LayersPanel({ tree, onFocusNode }: Props) {
  const [collapsedIds, setCollapsedIds] = useState<Set<string>>(new Set());
  const setCollapsed = (id: string, c: boolean) => {
    setCollapsedIds((prev) => {
      const next = new Set(prev);
      if (c) next.add(id); else next.delete(id);
      return next;
    });
  };

  const rootNode = useMemo(() => (tree.rootId ? tree.nodes[tree.rootId] : null), [tree]);
  const nodeCount = Object.keys(tree.nodes).length;

  return (
    <div className="h-full flex flex-col bg-paper-1">
      {/* Header */}
      <div className="h-9 shrink-0 flex items-center justify-between px-3 border-b border-[var(--ink-line)]">
        <div className="flex items-center gap-1.5">
          <Layers size={12} className="text-ink-2" />
          <span className="text-[11px] uppercase tracking-[0.16em] text-ink-2 font-medium">Layers</span>
        </div>
        <span className="text-[11px] text-ink-3 tabular-nums">{nodeCount}</span>
      </div>

      {/* Body */}
      <div className="flex-1 overflow-y-auto px-1 py-1.5">
        {rootNode ? (
          <Row
            node={rootNode}
            tree={tree}
            level={0}
            collapsedIds={collapsedIds}
            setCollapsed={setCollapsed}
            onFocusNode={onFocusNode}
          />
        ) : (
          <div className="px-3 py-6 text-[12px] text-ink-3 text-center">
            No nodes yet. Type a seed to start.
          </div>
        )}
      </div>

      {/* Footer hint */}
      <div className="h-7 shrink-0 flex items-center px-3 border-t border-[var(--ink-line)] text-[10.5px] text-ink-3">
        Click to focus · ⌘/ to toggle
      </div>
    </div>
  );
}
