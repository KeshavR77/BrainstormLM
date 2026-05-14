/**
 * ShellLayout — three-pane workspace shell with TopBar / StatusBar.
 *
 * When either side panel is collapsed, a thin vertical rail with a chevron
 * button appears at that edge so the user can re-open it without keyboard
 * shortcuts or the command palette.
 */
import { Panel, PanelGroup, PanelResizeHandle } from 'react-resizable-panels';
import type { ReactNode } from 'react';
import { ChevronRight, ChevronLeft, Layers, Info } from 'lucide-react';
import { useWorkspaceStore } from '../../stores/workspaceStore';

interface Props {
  topBar:     ReactNode;
  layers:     ReactNode;
  canvas:     ReactNode;
  inspector?: ReactNode;
  statusBar:  ReactNode;
}

export function ShellLayout({ topBar, layers, canvas, inspector, statusBar }: Props) {
  const showLayers       = useWorkspaceStore((s) => s.showLayers);
  const showInspector    = useWorkspaceStore((s) => s.showInspector);
  const layersSize       = useWorkspaceStore((s) => s.layersSize);
  const inspectorSize    = useWorkspaceStore((s) => s.inspectorSize);
  const setLayersSize    = useWorkspaceStore((s) => s.setLayersSize);
  const setInspectorSize = useWorkspaceStore((s) => s.setInspectorSize);
  const toggleLayers     = useWorkspaceStore((s) => s.toggleLayers);
  const toggleInspector  = useWorkspaceStore((s) => s.toggleInspector);

  return (
    <div className="h-screen w-screen flex flex-col bg-paper text-ink-0 font-sans antialiased overflow-hidden">
      {topBar}

      <main className="flex-1 min-h-0">
        <PanelGroup direction="horizontal" className="h-full">
          {showLayers ? (
            <>
              <Panel
                defaultSize={layersSize}
                minSize={12}
                maxSize={32}
                onResize={setLayersSize}
                className="border-r border-[var(--ink-line)]"
              >
                {layers}
              </Panel>
              <PanelResizeHandle className="w-px bg-[var(--ink-line)] hover:bg-brand/40 transition-colors data-[resize-handle-state=drag]:bg-brand" />
            </>
          ) : (
            <CollapsedRail side="left" label="Layers" Icon={Layers} onClick={toggleLayers} />
          )}

          <Panel minSize={32} className="relative">
            {canvas}
          </Panel>

          {showInspector && inspector ? (
            <>
              <PanelResizeHandle className="w-px bg-[var(--ink-line)] hover:bg-brand/40 transition-colors data-[resize-handle-state=drag]:bg-brand" />
              <Panel
                defaultSize={inspectorSize}
                minSize={16}
                maxSize={36}
                onResize={setInspectorSize}
                className="border-l border-[var(--ink-line)]"
              >
                {inspector}
              </Panel>
            </>
          ) : (
            <CollapsedRail side="right" label="Inspector" Icon={Info} onClick={toggleInspector} />
          )}
        </PanelGroup>
      </main>

      {statusBar}
    </div>
  );
}

interface RailProps {
  side: 'left' | 'right';
  label: string;
  Icon: React.ComponentType<{ size?: number; strokeWidth?: number; className?: string }>;
  onClick: () => void;
}

function CollapsedRail({ side, label, Icon, onClick }: RailProps) {
  const Chevron = side === 'left' ? ChevronRight : ChevronLeft;
  const borderCls = side === 'left' ? 'border-r' : 'border-l';
  return (
    <div className={`shrink-0 w-7 bg-paper-1 ${borderCls} border-[var(--ink-line)] flex flex-col items-center pt-3 gap-2 group`}>
      <button
        onClick={onClick}
        title={`Open ${label}`}
        className="flex flex-col items-center gap-1.5 px-1.5 py-2 rounded-md text-ink-2 hover:text-ink-0 hover:bg-paper-2 transition-colors"
      >
        <Chevron size={13} strokeWidth={2} />
        <Icon size={13} strokeWidth={1.8} />
        <span
          className="text-[9.5px] uppercase tracking-[0.18em] text-ink-3 font-medium select-none"
          style={{ writingMode: 'vertical-rl', textOrientation: 'mixed' }}
        >
          {label}
        </span>
      </button>
    </div>
  );
}
