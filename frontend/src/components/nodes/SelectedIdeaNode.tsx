/**
 * SelectedIdeaNode.tsx — An idea the user chose to pursue.
 *
 * When not yet expanded, shows an inline prompt input so the user
 * can guide what gets generated next. They can also auto-generate
 * or add their own custom idea.
 */

import { Handle, Position } from '@xyflow/react';
import type { NodeProps } from '@xyflow/react';
import { CheckCircle2, ArrowRight, Zap, Plus } from 'lucide-react';
import { useState } from 'react';

export interface SelectedIdeaNodeData {
  summary: string;
  content: string;
  depth: number;
  isExpanded: boolean;
  onExpandWithPrompt: (nodeId: string, prompt: string) => void;
  onAutoExpand: (nodeId: string) => void;
  onAddCustom: (parentId: string, content: string, summary: string) => void;
  nodeId: string;
  [key: string]: unknown;
}

export function SelectedIdeaNode({ data }: NodeProps) {
  const {
    summary,
    content,
    depth,
    isExpanded,
    onExpandWithPrompt,
    onAutoExpand,
    onAddCustom,
    nodeId,
  } = data as unknown as SelectedIdeaNodeData;

  const [prompt, setPrompt] = useState('');
  const [showCustom, setShowCustom] = useState(false);
  const [customSummary, setCustomSummary] = useState('');
  const [customContent, setCustomContent] = useState('');

  const handleSubmitPrompt = () => {
    if (prompt.trim()) {
      onExpandWithPrompt(nodeId, prompt.trim());
    }
  };

  const handleAddCustom = () => {
    if (customContent.trim()) {
      onAddCustom(nodeId, customContent.trim(), customSummary.trim());
      setCustomSummary('');
      setCustomContent('');
      setShowCustom(false);
    }
  };

  return (
    <div className="bs-node bs-node--selected">
      <Handle type="target" position={Position.Top} />

      <div className="bs-node__header">
        <div className="bs-node__icon bs-node__icon--selected">
          <CheckCircle2 size={13} />
        </div>
        <span className="bs-node__label">{summary}</span>
        <span className="bs-node__depth">D{depth}</span>
      </div>

      <div className="bs-node__body">
        <p className="bs-node__content">{content}</p>
      </div>

      {/* Inline prompt — only shown when node hasn't been expanded yet */}
      {!isExpanded && !showCustom && (
        <div className="bs-node__prompt">
          <div className="bs-node__prompt-label">What should we explore from here?</div>
          <textarea
            className="input bs-node__prompt-input"
            placeholder="e.g. Focus on B2B use cases, or explore pricing models..."
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && e.metaKey) handleSubmitPrompt();
            }}
            rows={2}
          />
          <div className="bs-node__prompt-actions">
            <button
              className="btn btn-sm btn-secondary"
              onClick={() => setShowCustom(true)}
              title="Add your own idea instead"
            >
              <Plus size={12} />
              Add Your Own
            </button>
            <button
              className="btn btn-sm btn-ghost"
              onClick={() => onAutoExpand(nodeId)}
            >
              <Zap size={12} />
              Auto-generate
            </button>
            <button
              className="btn btn-sm btn-primary"
              onClick={handleSubmitPrompt}
              disabled={!prompt.trim()}
            >
              <ArrowRight size={12} />
              Generate
            </button>
          </div>
        </div>
      )}

      {/* Add custom idea form */}
      {!isExpanded && showCustom && (
        <div className="bs-node__prompt">
          <div className="bs-node__prompt-label">Add your own idea</div>
          <input
            className="input bs-node__prompt-input"
            placeholder="Title (e.g. Enterprise API Layer)"
            value={customSummary}
            onChange={(e) => setCustomSummary(e.target.value)}
          />
          <textarea
            className="input bs-node__prompt-input"
            placeholder="Describe the idea..."
            value={customContent}
            onChange={(e) => setCustomContent(e.target.value)}
            rows={2}
          />
          <div className="bs-node__prompt-actions">
            <button
              className="btn btn-sm btn-ghost"
              onClick={() => setShowCustom(false)}
            >
              Back
            </button>
            <button
              className="btn btn-sm btn-primary"
              onClick={handleAddCustom}
              disabled={!customContent.trim()}
            >
              <Plus size={12} />
              Add Idea
            </button>
          </div>
        </div>
      )}

      {isExpanded && (
        <div className="bs-node__selected-badge">
          <CheckCircle2 size={11} />
          Chosen Path
        </div>
      )}

      <Handle type="source" position={Position.Bottom} />
    </div>
  );
}
