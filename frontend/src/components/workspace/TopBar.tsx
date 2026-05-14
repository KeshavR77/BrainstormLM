/**
 * TopBar — workspace top navigation. Breadcrumb, page tabs (placeholder),
 * share + presence avatar stubs (no real collab this round), open command palette.
 */
import { useCallback, useState } from 'react';
import { ArrowLeft, ChevronRight, Share2, Command, Undo2, Redo2, Clock, MessageSquare } from 'lucide-react';
import type { BrainstormPhase } from '../../types';
import { useWorkspaceStore } from '../../stores/workspaceStore';
import { Logo } from '../Logo';

interface Props {
  projectName: string;
  phase: BrainstormPhase;
  onBack: () => void;
  onUndo?: () => void;
  onRedo?: () => void;
  canUndo?: boolean;
  canRedo?: boolean;
  onOpenHistory?: () => void;
  onOpenComments?: () => void;
  commentsCount?: number;
}

export function TopBar({
  projectName, phase, onBack, onUndo, onRedo, canUndo, canRedo,
  onOpenHistory, onOpenComments, commentsCount,
}: Props) {
  const openCommand = useWorkspaceStore((s) => s.openCommand);

  const phaseLabel: Record<BrainstormPhase['status'], string> = {
    IDLE: 'Idle',
    GENERATING_DIVERSE_POOL: 'Exploring',
    SYNTHESIZING_QUALITY: 'Refining',
    COMPLETE: 'Complete',
    ERROR: 'Error',
  };

  return (
    <header className="h-12 shrink-0 flex items-center justify-between gap-3 px-4 border-b border-[var(--ink-line)] bg-paper/90 backdrop-blur-md">
      {/* Left: nav + breadcrumb */}
      <div className="flex items-center gap-3 min-w-0">
        <a href="#/" className="shrink-0 inline-flex" title="Home">
          <Logo size={18} />
        </a>
        <ChevronRight size={11} className="text-ink-3 shrink-0" />
        <button
          onClick={onBack}
          className="inline-flex items-center gap-1 text-ink-1 hover:text-ink-0 text-[12.5px] px-1.5 py-1 rounded transition-colors shrink-0"
          title="Back to dashboard"
        >
          <ArrowLeft size={11} /> Projects
        </button>
        <ChevronRight size={11} className="text-ink-3 shrink-0" />
        <div className="flex items-center gap-2 min-w-0">
          <span
            className="font-display text-ink-0 text-[14.5px] tracking-tight truncate"
            style={{ fontVariationSettings: '"wght" 540, "opsz" 24' }}
          >
            {projectName}
          </span>
        </div>
      </div>

      {/* Center: phase indicator */}
      <div className="hidden md:inline-flex items-center gap-2 px-3 py-1 rounded-full bg-paper-1 border border-[var(--ink-line)] text-[11px] text-ink-1">
        <span
          className="h-1.5 w-1.5 rounded-full"
          style={{
            background:
              phase.status === 'COMPLETE' ? 'var(--brand)'
              : phase.status === 'GENERATING_DIVERSE_POOL' ? 'oklch(0.62 0.20 140)'
              : phase.status === 'SYNTHESIZING_QUALITY' ? 'oklch(0.62 0.20 280)'
              : phase.status === 'ERROR' ? 'var(--accent-red)'
              : 'var(--ink-3)',
            animation: phase.status === 'GENERATING_DIVERSE_POOL' || phase.status === 'SYNTHESIZING_QUALITY' ? 'pulse 1.4s ease-in-out infinite' : 'none',
          }}
        />
        {phaseLabel[phase.status]}
      </div>

      {/* Right: undo/redo + history + command + share + avatars */}
      <div className="flex items-center gap-2">
        {/* Undo/Redo */}
        <div className="hidden md:inline-flex items-center gap-0.5 mr-1">
          <button
            onClick={onUndo}
            disabled={!canUndo}
            title="Undo (⌘Z)"
            className="p-1.5 rounded-md text-ink-1 hover:text-ink-0 hover:bg-paper-2 disabled:opacity-30 disabled:hover:bg-transparent transition-colors"
          >
            <Undo2 size={13} />
          </button>
          <button
            onClick={onRedo}
            disabled={!canRedo}
            title="Redo (⌘⇧Z)"
            className="p-1.5 rounded-md text-ink-1 hover:text-ink-0 hover:bg-paper-2 disabled:opacity-30 disabled:hover:bg-transparent transition-colors"
          >
            <Redo2 size={13} />
          </button>
          <button
            onClick={onOpenHistory}
            title="Version history"
            className="p-1.5 rounded-md text-ink-1 hover:text-ink-0 hover:bg-paper-2 transition-colors"
          >
            <Clock size={13} />
          </button>
          <button
            onClick={onOpenComments}
            title="Comments"
            className="relative p-1.5 rounded-md text-ink-1 hover:text-ink-0 hover:bg-paper-2 transition-colors"
          >
            <MessageSquare size={13} />
            {commentsCount && commentsCount > 0 ? (
              <span className="absolute -top-0.5 -right-0.5 min-w-[14px] h-[14px] rounded-full bg-brand text-white text-[9px] font-medium flex items-center justify-center px-1">
                {commentsCount}
              </span>
            ) : null}
          </button>
        </div>

        <button
          onClick={openCommand}
          className="inline-flex items-center gap-1.5 text-ink-1 hover:text-ink-0 text-[12px] px-2 py-1 rounded-md bg-paper-1 border border-[var(--ink-line)] hover:bg-paper-2 transition-colors"
          title="Command palette (⌘K)"
        >
          <Command size={11} /> <span className="hidden sm:inline">Search</span>
          <kbd className="hidden sm:inline ml-1 text-[10px] text-ink-3 tabular-nums">⌘K</kbd>
        </button>

        <ShareButton projectName={projectName} />
      </div>

      <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 0.5; transform: scale(1); }
          50% { opacity: 1; transform: scale(1.3); }
        }
        @keyframes toastIn {
          0%   { opacity: 0; transform: translateY(-6px); }
          100% { opacity: 1; transform: translateY(0); }
        }
      `}</style>
    </header>
  );
}

/* Share button writes a hash link to clipboard and shows a 2s toast. */
function ShareButton({ projectName }: { projectName: string }) {
  const [copied, setCopied] = useState(false);
  const onShare = useCallback(async () => {
    const url = window.location.href;
    try {
      await navigator.clipboard.writeText(url);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 2000);
    } catch {
      setCopied(false);
    }
  }, []);
  return (
    <div className="relative">
      <button
        onClick={onShare}
        title={`Share ${projectName}`}
        className="inline-flex items-center gap-1.5 bg-ink-0 text-paper text-[12px] font-medium px-3 py-1.5 rounded-full hover:bg-ink-1 transition-colors"
      >
        <Share2 size={11} /> Share
      </button>
      {copied && (
        <div
          className="absolute top-full right-0 mt-1.5 px-2.5 py-1 rounded-md bg-ink-0 text-paper text-[11px] whitespace-nowrap shadow-lg"
          style={{ animation: 'toastIn 220ms cubic-bezier(0.22, 1, 0.36, 1)' }}
        >
          ✓ Link copied
        </div>
      )}
    </div>
  );
}
