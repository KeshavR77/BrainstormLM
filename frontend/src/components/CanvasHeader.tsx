/**
 * CanvasHeader.tsx — Top bar for the brainstorm canvas page.
 */

import { ArrowLeft, GitBranch, Layers, Loader2, Zap } from 'lucide-react';

interface CanvasHeaderProps {
  projectName: string;
  nodeCount: number;
  maxDepth: number;
  loading: boolean;
  onBack: () => void;
}

export function CanvasHeader({
  projectName,
  nodeCount,
  maxDepth,
  loading,
  onBack,
}: CanvasHeaderProps) {
  return (
    <div className="canvas-header">
      <div className="canvas-header__left">
        <button
          className="btn btn-ghost btn-icon canvas-header__back"
          onClick={onBack}
          aria-label="Back to dashboard"
        >
          <ArrowLeft size={18} />
        </button>
        <div className="canvas-header__divider" />
        <div className="canvas-header__brand">
          <div className="canvas-header__logo"><Zap size={12} /></div>
        </div>
        <div className="canvas-header__divider" />
        <h1 className="canvas-header__title">{projectName}</h1>
      </div>

      <div className="canvas-header__right">
        {loading && (
          <span className="canvas-header__loading">
            <Loader2 size={14} className="spin" />
            Generating…
          </span>
        )}
        <span className="canvas-header__stat">
          <GitBranch size={13} />
          {nodeCount} nodes
        </span>
        <span className="canvas-header__stat">
          <Layers size={13} />
          Depth {maxDepth}
        </span>
      </div>
    </div>
  );
}
