/**
 * SeedNode.tsx — The root brainstorm node where the user starts.
 */

import { Handle, Position } from '@xyflow/react';
import type { NodeProps } from '@xyflow/react';
import { Sparkles, ArrowRight } from 'lucide-react';
import { useState } from 'react';

interface SeedNodeData {
  content: string;
  submitted: boolean;
  onSubmit: (text: string) => void;
  [key: string]: unknown;
}

export function SeedNode({ data }: NodeProps) {
  const { content, submitted, onSubmit } = data as unknown as SeedNodeData;
  const [text, setText] = useState(content ?? '');

  const handleSubmit = () => {
    if (text.trim() && onSubmit) onSubmit(text.trim());
  };

  return (
    <div className="bs-node bs-node--seed">
      <div className="bs-node__accent" />
      <div className="bs-node__header">
        <div className="bs-node__icon bs-node__icon--seed">
          <Sparkles size={14} />
        </div>
        <span className="bs-node__label">Starting Idea</span>
      </div>

      <div className="bs-node__body">
        {submitted ? (
          <p className="bs-node__content">{content}</p>
        ) : (
          <>
            <textarea
              className="input bs-seed-input"
              placeholder="What strategic question should we explore?"
              value={text}
              onChange={(e) => setText(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && e.metaKey) handleSubmit();
              }}
              rows={3}
            />
            <button
              className="btn btn-primary btn-lg bs-seed-submit"
              onClick={handleSubmit}
              disabled={!text.trim()}
            >
              Start Brainstorming
              <ArrowRight size={16} />
            </button>
          </>
        )}
      </div>

      <Handle type="source" position={Position.Bottom} />
    </div>
  );
}
