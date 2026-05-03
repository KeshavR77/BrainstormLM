/**
 * IdeaNode.tsx — A generated brainstorm direction.
 *
 * Can be selected, expanded, or edited inline.
 * Double-click the content to edit it.
 */

import { Handle, Position } from '@xyflow/react';
import type { NodeProps } from '@xyflow/react';
import { Lightbulb, ChevronRight, Expand, Pencil, Check, X } from 'lucide-react';
import { useState, useRef, useEffect } from 'react';

export interface IdeaNodeData {
  summary: string;
  content: string;
  depth: number;
  isExpanded: boolean;
  hasChildren: boolean;
  onSelect: (nodeId: string) => void;
  onExpand: (nodeId: string) => void;
  onEdit: (nodeId: string, content: string, summary: string) => void;
  nodeId: string;
  [key: string]: unknown;
}

export function IdeaNode({ data }: NodeProps) {
  const {
    summary,
    content,
    depth,
    isExpanded,
    hasChildren,
    onSelect,
    onExpand,
    onEdit,
    nodeId,
  } = data as unknown as IdeaNodeData;

  const [editing, setEditing] = useState(false);
  const [editSummary, setEditSummary] = useState(summary);
  const [editContent, setEditContent] = useState(content);
  const summaryRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (editing) {
      summaryRef.current?.focus();
    }
  }, [editing]);

  const handleSave = () => {
    if (editContent.trim()) {
      onEdit(nodeId, editContent.trim(), editSummary.trim());
      setEditing(false);
    }
  };

  const handleCancel = () => {
    setEditSummary(summary);
    setEditContent(content);
    setEditing(false);
  };

  return (
    <div className="bs-node bs-node--idea">
      <Handle type="target" position={Position.Top} />

      <div className="bs-node__header">
        <div className="bs-node__icon bs-node__icon--idea">
          <Lightbulb size={13} />
        </div>
        {editing ? (
          <input
            ref={summaryRef}
            className="input bs-node__edit-title"
            value={editSummary}
            onChange={(e) => setEditSummary(e.target.value)}
            placeholder="Title"
            onKeyDown={(e) => {
              if (e.key === 'Enter') handleSave();
              if (e.key === 'Escape') handleCancel();
            }}
          />
        ) : (
          <span className="bs-node__label">{summary}</span>
        )}
        <span className="bs-node__depth">D{depth}</span>
      </div>

      <div className="bs-node__body">
        {editing ? (
          <textarea
            className="input bs-node__edit-content"
            value={editContent}
            onChange={(e) => setEditContent(e.target.value)}
            placeholder="Describe the idea..."
            rows={3}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && e.metaKey) handleSave();
              if (e.key === 'Escape') handleCancel();
            }}
          />
        ) : (
          <p
            className="bs-node__content bs-node__content--editable"
            onDoubleClick={() => setEditing(true)}
            title="Double-click to edit"
          >
            {content}
          </p>
        )}
      </div>

      <div className="bs-node__actions">
        {editing ? (
          <>
            <button
              className="btn btn-sm btn-ghost bs-node__action-btn"
              onClick={handleCancel}
            >
              <X size={12} />
              Cancel
            </button>
            <button
              className="btn btn-sm btn-primary bs-node__action-btn"
              onClick={handleSave}
              disabled={!editContent.trim()}
            >
              <Check size={12} />
              Save
            </button>
          </>
        ) : (
          <>
            <button
              className="btn btn-sm btn-ghost bs-node__action-btn"
              onClick={(e) => {
                e.stopPropagation();
                setEditing(true);
              }}
              title="Edit this idea"
            >
              <Pencil size={11} />
              Edit
            </button>
            {!isExpanded && (
              <button
                className="btn btn-sm btn-ghost bs-node__action-btn"
                onClick={(e) => {
                  e.stopPropagation();
                  onExpand(nodeId);
                }}
                title="Expand — generate more ideas from this direction"
              >
                <Expand size={12} />
                Expand
              </button>
            )}
            <button
              className="btn btn-sm btn-primary bs-node__action-btn"
              onClick={(e) => {
                e.stopPropagation();
                onSelect(nodeId);
              }}
            >
              <ChevronRight size={13} />
              Choose This
            </button>
          </>
        )}
      </div>

      {(isExpanded || hasChildren) && (
        <Handle type="source" position={Position.Bottom} />
      )}
    </div>
  );
}
