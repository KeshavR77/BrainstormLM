/**
 * CommandPalette — Cmd+K fuzzy palette for nodes, actions, navigation.
 *
 * Built on `cmdk`. Groups: Actions, Nodes, Navigate. Recent searches
 * persist via the workspace store. Closes on Escape or click-outside.
 */
import { Command } from 'cmdk';
import { useEffect, useMemo, useState } from 'react';
import {
  ArrowRight,
  Sparkles,
  Save,
  Search as SearchIcon,
  PanelLeft,
  PanelRight,
  Trash2,
  Maximize2,
  Home,
  Layers as LayersIcon,
} from 'lucide-react';
import { useWorkspaceStore } from '../../stores/workspaceStore';
import type { BrainstormTree } from '../../types';
import { depthHue } from '../../styles/motion';

interface Props {
  tree: BrainstormTree;
  onFocusNode?: (id: string) => void;
  onNavigateHome?: () => void;
  onSnapshot?: () => void;
  onFitView?: () => void;
}

export function CommandPalette({
  tree,
  onFocusNode,
  onNavigateHome,
  onSnapshot,
  onFitView,
}: Props) {
  const open    = useWorkspaceStore((s) => s.commandOpen);
  const close   = useWorkspaceStore((s) => s.closeCommand);
  const layers  = useWorkspaceStore((s) => s.toggleLayers);
  const insp    = useWorkspaceStore((s) => s.toggleInspector);
  const setSelection = useWorkspaceStore((s) => s.setSelection);

  const [query, setQuery] = useState('');

  /* Flattened, searchable node list */
  const nodes = useMemo(
    () => Object.values(tree.nodes).filter((n) => n.summary || n.content),
    [tree.nodes],
  );

  useEffect(() => {
    if (!open) setQuery('');
  }, [open]);

  /* Click-outside backdrop */
  if (!open) return null;

  const runAction = (fn?: () => void) => () => {
    fn?.();
    close();
  };

  return (
    <div
      onClick={close}
      className="fixed inset-0 z-50 flex items-start justify-center bg-ink-0/30 backdrop-blur-sm pt-[12vh] px-4"
    >
      <div
        onClick={(e) => e.stopPropagation()}
        className="w-full max-w-xl rounded-2xl bg-paper border border-[var(--ink-line-bold)] shadow-lift overflow-hidden"
      >
        <Command label="Command palette" className="font-sans">
          <div className="flex items-center gap-2 px-4 py-3 border-b border-[var(--ink-line)]">
            <SearchIcon size={14} className="text-ink-2" />
            <Command.Input
              autoFocus
              value={query}
              onValueChange={setQuery}
              placeholder="Search nodes, actions…"
              className="flex-1 bg-transparent outline-none text-[14px] text-ink-0 placeholder:text-ink-3"
            />
            <kbd className="text-[10px] text-ink-3 tabular-nums">ESC</kbd>
          </div>
          <Command.List className="max-h-[440px] overflow-y-auto p-1.5">
            <Command.Empty className="px-4 py-8 text-center text-[12px] text-ink-3">
              No matches for "{query}"
            </Command.Empty>

            {/* Actions */}
            <Command.Group heading="Actions" className="text-[10px] uppercase tracking-[0.18em] text-ink-3 px-2 py-1.5">
              <Command.Item value="fit view zoom" onSelect={runAction(onFitView)} className="palette-item">
                <Maximize2 size={13} className="text-ink-2" />
                <span>Fit view</span>
                <kbd className="ml-auto text-[10px] text-ink-3">⌘0</kbd>
              </Command.Item>
              <Command.Item value="snapshot save" onSelect={runAction(onSnapshot)} className="palette-item">
                <Save size={13} className="text-ink-2" />
                <span>Save snapshot</span>
                <kbd className="ml-auto text-[10px] text-ink-3">⌘S</kbd>
              </Command.Item>
              <Command.Item value="toggle layers" onSelect={runAction(layers)} className="palette-item">
                <PanelLeft size={13} className="text-ink-2" />
                <span>Toggle layers panel</span>
                <kbd className="ml-auto text-[10px] text-ink-3">⌘/</kbd>
              </Command.Item>
              <Command.Item value="toggle inspector" onSelect={runAction(insp)} className="palette-item">
                <PanelRight size={13} className="text-ink-2" />
                <span>Toggle inspector</span>
                <kbd className="ml-auto text-[10px] text-ink-3">⌘.</kbd>
              </Command.Item>
              <Command.Item value="clear selection" onSelect={runAction(() => setSelection([]))} className="palette-item">
                <Trash2 size={13} className="text-ink-2" />
                <span>Clear selection</span>
              </Command.Item>
            </Command.Group>

            {/* Navigate */}
            <Command.Group heading="Navigate" className="text-[10px] uppercase tracking-[0.18em] text-ink-3 px-2 py-1.5 mt-2">
              <Command.Item value="home dashboard projects" onSelect={runAction(onNavigateHome)} className="palette-item">
                <Home size={13} className="text-ink-2" />
                <span>Back to projects</span>
              </Command.Item>
            </Command.Group>

            {/* Nodes */}
            {nodes.length > 0 && (
              <Command.Group heading="Nodes" className="text-[10px] uppercase tracking-[0.18em] text-ink-3 px-2 py-1.5 mt-2">
                {nodes.map((n) => (
                  <Command.Item
                    key={n.id}
                    value={`${n.summary} ${n.content}`}
                    onSelect={() => {
                      setSelection([n.id]);
                      onFocusNode?.(n.id);
                      close();
                    }}
                    className="palette-item"
                  >
                    <span
                      className="h-1.5 w-1.5 rounded-full shrink-0"
                      style={{ background: `oklch(0.62 0.20 ${depthHue(n.depth)})` }}
                    />
                    <Sparkles size={11} className="text-ink-2 shrink-0" />
                    <span className="truncate">{n.summary || n.content.slice(0, 60)}</span>
                    <span className="ml-auto text-[10px] text-ink-3 tabular-nums">d{n.depth}</span>
                    <ArrowRight size={11} className="text-ink-3" />
                  </Command.Item>
                ))}
              </Command.Group>
            )}
          </Command.List>

          <div className="flex items-center justify-between px-4 py-2 border-t border-[var(--ink-line)] bg-paper-1 text-[10.5px] text-ink-3">
            <div className="flex items-center gap-3">
              <span><kbd className="text-ink-1">↑↓</kbd> navigate</span>
              <span><kbd className="text-ink-1">↵</kbd> select</span>
            </div>
            <span><LayersIcon size={10} className="inline mr-1" />{nodes.length} nodes</span>
          </div>
        </Command>
      </div>

      <style>{`
        .palette-item {
          display: flex;
          align-items: center;
          gap: 8px;
          padding: 8px 10px;
          font-size: 12.5px;
          color: var(--ink-0);
          border-radius: 8px;
          cursor: pointer;
          user-select: none;
        }
        .palette-item[data-selected="true"] {
          background: var(--paper-2);
        }
        .palette-item:hover { background: var(--paper-2); }
      `}</style>
    </div>
  );
}
