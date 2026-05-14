/**
 * StatusBar — workspace footer with metrics and shortcuts hint.
 */
import { Keyboard, PanelLeft, PanelRight, PanelLeftClose, PanelRightClose } from 'lucide-react';
import { useWorkspaceStore } from '../../stores/workspaceStore';
import type { BrainstormPhase } from '../../types';

interface Props {
  phase: BrainstormPhase;
  nodeCount: number;
  maxDepth: number;
}

export function StatusBar({ phase, nodeCount, maxDepth }: Props) {
  const tool         = useWorkspaceStore((s) => s.activeTool);
  const zoom         = useWorkspaceStore((s) => s.zoomPercent);
  const selectedN    = useWorkspaceStore((s) => s.selectedNodeIds.length);
  const showLayers   = useWorkspaceStore((s) => s.showLayers);
  const showInspector= useWorkspaceStore((s) => s.showInspector);
  const toggleLayers = useWorkspaceStore((s) => s.toggleLayers);
  const toggleInsp   = useWorkspaceStore((s) => s.toggleInspector);

  return (
    <footer className="h-7 shrink-0 flex items-center justify-between px-3 border-t border-[var(--ink-line)] bg-paper-1 text-[10.5px] text-ink-2 font-mono">
      <div className="flex items-center gap-3">
        {/* Panel toggles */}
        <button
          onClick={toggleLayers}
          title={`${showLayers ? 'Hide' : 'Show'} layers (⌘/)`}
          className={`p-0.5 rounded transition-colors ${showLayers ? 'text-ink-0 bg-paper-2' : 'text-ink-2 hover:text-ink-0 hover:bg-paper-2'}`}
        >
          {showLayers ? <PanelLeftClose size={12} /> : <PanelLeft size={12} />}
        </button>
        <button
          onClick={toggleInsp}
          title={`${showInspector ? 'Hide' : 'Show'} inspector (⌘.)`}
          className={`p-0.5 rounded transition-colors ${showInspector ? 'text-ink-0 bg-paper-2' : 'text-ink-2 hover:text-ink-0 hover:bg-paper-2'}`}
        >
          {showInspector ? <PanelRightClose size={12} /> : <PanelRight size={12} />}
        </button>
        <span className="text-ink-3/50">|</span>
        <span className="inline-flex items-center gap-1">
          <span className="h-1 w-1 rounded-full bg-brand" />
          {nodeCount} nodes · depth {maxDepth}
        </span>
        <span className="text-ink-3/50">|</span>
        <span>
          {phase.status === 'IDLE'                    ? 'Ready' :
           phase.status === 'GENERATING_DIVERSE_POOL' ? 'Generating diverse pool…' :
           phase.status === 'SYNTHESIZING_QUALITY'    ? 'Synthesizing quality…' :
           phase.status === 'COMPLETE'                ? 'Generation complete' :
           phase.status === 'ERROR'                   ? 'Error' : 'Ready'}
        </span>
        {selectedN > 0 && (
          <>
            <span className="text-ink-3/50">|</span>
            <span className="text-brand">{selectedN} selected</span>
          </>
        )}
      </div>
      <div className="flex items-center gap-4">
        <span className="capitalize">Tool: <span className="text-ink-0">{tool}</span></span>
        <span>Zoom: <span className="text-ink-0 tabular-nums">{zoom}%</span></span>
        <span className="inline-flex items-center gap-1 text-ink-3">
          <Keyboard size={11} /> Press <kbd className="text-ink-0">?</kbd>
        </span>
      </div>
    </footer>
  );
}
