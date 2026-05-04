import { ArrowLeft, GitBranch, Layers, Zap } from 'lucide-react';
import type { BrainstormPhase } from '../types';
import { PhaseBanner } from './PhaseBanner';

interface CanvasHeaderProps {
  projectName: string;
  nodeCount: number;
  maxDepth: number;
  phase: BrainstormPhase;
  onBack: () => void;
}

export function CanvasHeader({
  projectName,
  nodeCount,
  maxDepth,
  phase,
  onBack,
}: CanvasHeaderProps) {
  return (
    <>
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

      <PhaseBanner phase={phase} />
    </>
  );
}
