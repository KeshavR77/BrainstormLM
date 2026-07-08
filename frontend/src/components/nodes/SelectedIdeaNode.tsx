/**
 * SelectedIdeaNode.tsx — An idea the user chose to pursue.
 *
 * When not yet expanded, shows an inline prompt input so the user
 * can guide what gets generated next. They can also auto-generate
 * or add their own custom idea.
 *
 * Also shows a collapsible critique note so users can annotate why
 * they chose this direction (feeds the RLHF critique field).
 */

import { Handle, Position } from '@xyflow/react';
import type { NodeProps } from '@xyflow/react';
import { CheckCircle2, ArrowRight, Zap, Plus, Pencil } from 'lucide-react';
import { useState } from 'react';

interface SelectedIdeaNodeData {
  summary: string;
  content: string;
  depth: number;
  isExpanded: boolean;
  onExpandWithPrompt: (nodeId: string, prompt: string) => void;
  onAutoExpand: (nodeId: string) => void;
  onAddCustom: (parentId: string, content: string, summary: string) => void;
  onCritique?: (nodeId: string, critique: string) => void;
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
    onCritique,
    nodeId,
  } = data as unknown as SelectedIdeaNodeData;

  const [prompt, setPrompt] = useState('');
  const [showCustom, setShowCustom] = useState(false);
  const [showCritique, setShowCritique] = useState(false);
  const [critique, setCritique] = useState('');

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

  const handleSaveCritique = () => {
    if (critique.trim()) {
      onCritique?.(nodeId, critique.trim());
      setShowCritique(false);
    }
  };

  const [customSummary, setCustomSummary] = useState('');
  const [customContent, setCustomContent] = useState('');

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
            placeholder="e.g. Push toward enterprise, or explore the API-first angle..."
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

      {/* Critique note — collapsible, visible regardless of expanded state */}
      {onCritique && (
        <div className="bs-node__critique">
          {!showCritique ? (
            <button
              className="btn btn-sm btn-ghost bs-node__critique-toggle"
              onClick={() => setShowCritique(true)}
            >
              <Pencil size={10} />
              Add a note
            </button>
          ) : (
            <>
              <textarea
                className="input bs-node__critique-input"
                placeholder="Why did you choose this direction? (optional)"
                value={critique}
                onChange={(e) => setCritique(e.target.value)}
                rows={2}
                autoFocus
              />
              <div className="bs-node__critique-actions">
                <button
                  className="btn btn-sm btn-ghost"
                  onClick={() => { setShowCritique(false); setCritique(''); }}
                >
                  Cancel
                </button>
                <button
                  className="btn btn-sm btn-primary"
                  onClick={handleSaveCritique}
                  disabled={!critique.trim()}
                >
                  Save note
                </button>
              </div>
            </>
          )}
        </div>
      )}

      <Handle type="source" position={Position.Bottom} />
    </div>
  );
}
