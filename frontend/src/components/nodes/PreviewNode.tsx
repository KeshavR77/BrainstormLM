import { Handle, Position } from '@xyflow/react';
import type { NodeProps } from '@xyflow/react';
import { Sparkles } from 'lucide-react';

export interface PreviewNodeData {
  summary: string;
  content: string;
  [key: string]: unknown;
}

export function PreviewNode({ data }: NodeProps) {
  const { summary, content } = data as unknown as PreviewNodeData;

  return (
    <div className="bs-node bs-node--preview">
      <Handle type="target" position={Position.Top} />

      <div className="bs-node__accent" />

      <div className="bs-node__header">
        <div className="bs-node__icon bs-node__icon--preview">
          <Sparkles size={13} />
        </div>
        <span className="bs-node__label">{summary}</span>
      </div>

      <div className="bs-node__body">
        <p className="bs-node__content">
          {content.length > 120 ? content.slice(0, 117) + '...' : content}
        </p>
      </div>
    </div>
  );
}
