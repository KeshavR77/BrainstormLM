/**
 * NewProjectModal.tsx — Modal dialog for creating a new brainstorm project.
 */

import { useState, useRef, useEffect } from 'react';
import { X, Plus } from 'lucide-react';

interface NewProjectModalProps {
  isOpen: boolean;
  onClose: () => void;
  onCreate: (name: string, description: string) => void;
}

export function NewProjectModal({ isOpen, onClose, onCreate }: NewProjectModalProps) {
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const nameRef = useRef<HTMLInputElement>(null);

  // Focus the name input when modal opens
  useEffect(() => {
    if (isOpen) {
      setName('');
      setDescription('');
      setTimeout(() => nameRef.current?.focus(), 100);
    }
  }, [isOpen]);

  // Close on Escape
  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isOpen) onClose();
    };
    window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (name.trim()) {
      onCreate(name.trim(), description.trim());
      onClose();
    }
  };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
            <div>
              <h2>New Project</h2>
              <p>Start a new brainstorming session</p>
            </div>
            <button className="btn btn-ghost btn-icon" onClick={onClose} aria-label="Close">
              <X size={18} />
            </button>
          </div>
        </div>

        <form onSubmit={handleSubmit}>
          <div className="modal-body">
            <div>
              <label className="input-label" htmlFor="project-name">Project Name *</label>
              <input
                ref={nameRef}
                id="project-name"
                className="input"
                type="text"
                placeholder="e.g. Q3 Go-to-Market Strategy"
                value={name}
                onChange={(e) => setName(e.target.value)}
                maxLength={80}
              />
            </div>
            <div>
              <label className="input-label" htmlFor="project-desc">Description</label>
              <textarea
                id="project-desc"
                className="input"
                placeholder="What are you brainstorming about? (optional)"
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                rows={3}
                maxLength={300}
              />
            </div>
          </div>
          <div className="modal-footer">
            <button type="button" className="btn btn-secondary" onClick={onClose}>
              Cancel
            </button>
            <button type="submit" className="btn btn-primary" disabled={!name.trim()}>
              <Plus size={15} />
              Create Project
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
