/**
 * useHotkeys — single-source keyboard binding for the workspace.
 *
 * Bindings deliberately mirror Figma/Linear conventions: V/E/B/F/C/T for tools,
 * Cmd+K for command palette, Cmd+/ for layers, Cmd+. for inspector, ? for help.
 *
 * Skips when the user is typing in an editable field.
 */
import { useEffect } from 'react';
import { useWorkspaceStore, type Tool } from '../stores/workspaceStore';

function isEditableTarget(e: KeyboardEvent): boolean {
  const t = e.target as HTMLElement | null;
  if (!t) return false;
  const tag = t.tagName;
  if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return true;
  if (t.isContentEditable) return true;
  return false;
}

const TOOL_KEYS: Record<string, Tool> = {
  v: 'select',
  e: 'expand',
  b: 'branch',
  f: 'frame',
  c: 'comment',
  t: 'text',
  h: 'hand',
  z: 'zoom',
};

interface Options {
  onShowShortcuts?: () => void;
  onFocusSearch?: () => void;
  onUndo?: () => void;
  onRedo?: () => void;
  onSaveSnapshot?: () => void;
  onFitView?: () => void;
}

export function useHotkeys(opts: Options = {}) {
  const setTool         = useWorkspaceStore((s) => s.setTool);
  const openCommand     = useWorkspaceStore((s) => s.openCommand);
  const toggleLayers    = useWorkspaceStore((s) => s.toggleLayers);
  const toggleInspector = useWorkspaceStore((s) => s.toggleInspector);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      // Cmd/Ctrl+K — command palette (active even in inputs)
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault();
        openCommand();
        return;
      }

      // Cmd/Ctrl+Z / Cmd+Shift+Z — undo/redo (active everywhere)
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'z') {
        e.preventDefault();
        if (e.shiftKey) opts.onRedo?.();
        else            opts.onUndo?.();
        return;
      }

      // Cmd/Ctrl+S — snapshot
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 's') {
        e.preventDefault();
        opts.onSaveSnapshot?.();
        return;
      }

      // Cmd/Ctrl+0 — fit view
      if ((e.metaKey || e.ctrlKey) && e.key === '0') {
        e.preventDefault();
        opts.onFitView?.();
        return;
      }

      if (isEditableTarget(e)) return;

      // Cmd/Ctrl+/ — toggle layers
      if ((e.metaKey || e.ctrlKey) && e.key === '/') {
        e.preventDefault();
        toggleLayers();
        return;
      }
      // Cmd/Ctrl+. — toggle inspector
      if ((e.metaKey || e.ctrlKey) && e.key === '.') {
        e.preventDefault();
        toggleInspector();
        return;
      }
      // ? — shortcut sheet
      if (e.key === '?' && !e.metaKey && !e.ctrlKey) {
        e.preventDefault();
        opts.onShowShortcuts?.();
        return;
      }
      // / — focus search
      if (e.key === '/' && !e.metaKey && !e.ctrlKey) {
        e.preventDefault();
        opts.onFocusSearch?.();
        return;
      }
      // Single-letter tool bindings
      const tool = TOOL_KEYS[e.key.toLowerCase()];
      if (tool && !e.metaKey && !e.ctrlKey && !e.altKey && !e.shiftKey) {
        e.preventDefault();
        setTool(tool);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [setTool, openCommand, toggleLayers, toggleInspector, opts]);
}
