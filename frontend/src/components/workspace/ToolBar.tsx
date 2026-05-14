/**
 * ToolBar — floating tool selector, pill-shaped, top-center of canvas.
 *
 * Active tool gets a brand magenta ring; hover gets a soft OKLCH glow.
 * Keyboard shortcuts (single-letter) are stubbed for Phase 3 useHotkeys.
 */
import {
  MousePointer2,
  Sparkles,
  GitBranch,
  Square,
  MessageSquare,
  Type,
  Search,
  Hand,
  ZoomIn,
} from 'lucide-react';
import type { LucideIcon } from 'lucide-react';
import { useWorkspaceStore, type Tool } from '../../stores/workspaceStore';

interface ToolDef {
  id: Tool;
  label: string;
  hint: string;
  Icon: LucideIcon;
}

const TOOLS: ToolDef[] = [
  { id: 'select',  label: 'Select',  hint: 'V', Icon: MousePointer2 },
  { id: 'expand',  label: 'Expand',  hint: 'E', Icon: Sparkles },
  { id: 'branch',  label: 'Branch',  hint: 'B', Icon: GitBranch },
  { id: 'frame',   label: 'Frame',   hint: 'F', Icon: Square },
  { id: 'comment', label: 'Comment', hint: 'C', Icon: MessageSquare },
  { id: 'text',    label: 'Text',    hint: 'T', Icon: Type },
];
const SECONDARY: ToolDef[] = [
  { id: 'search', label: 'Search', hint: '/', Icon: Search },
  { id: 'hand',   label: 'Hand',   hint: 'H', Icon: Hand },
  { id: 'zoom',   label: 'Zoom',   hint: 'Z', Icon: ZoomIn },
];

export function ToolBar() {
  const activeTool = useWorkspaceStore((s) => s.activeTool);
  const setTool    = useWorkspaceStore((s) => s.setTool);

  const renderTool = (t: ToolDef) => {
    const active = activeTool === t.id;
    return (
      <button
        key={t.id}
        onClick={() => setTool(t.id)}
        title={`${t.label} · ${t.hint}`}
        className={`group relative flex items-center justify-center h-8 w-8 rounded-full transition-all duration-150 ${
          active
            ? 'bg-ink-0 text-paper'
            : 'text-ink-1 hover:text-ink-0 hover:bg-paper-2'
        }`}
      >
        <t.Icon size={14} strokeWidth={active ? 2.4 : 1.8} />
        {active && (
          <span
            className="absolute -inset-[3px] rounded-full pointer-events-none"
            style={{ boxShadow: '0 0 0 1.5px var(--brand)' }}
          />
        )}
        <span className="pointer-events-none absolute -bottom-7 left-1/2 -translate-x-1/2 opacity-0 group-hover:opacity-100 transition-opacity duration-150 whitespace-nowrap rounded-md bg-ink-0 text-paper text-[10px] px-1.5 py-1 font-mono">
          {t.hint}
        </span>
      </button>
    );
  };

  return (
    <div className="absolute top-4 left-1/2 -translate-x-1/2 z-30 pointer-events-auto">
      <div className="flex items-center gap-1 p-1 rounded-full bg-paper/95 backdrop-blur-md border border-[var(--ink-line)] shadow-lift">
        {TOOLS.map(renderTool)}
        <div className="h-5 w-px bg-[var(--ink-line)] mx-0.5" />
        {SECONDARY.map(renderTool)}
      </div>
    </div>
  );
}
