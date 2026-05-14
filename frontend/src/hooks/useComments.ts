/**
 * useComments — per-project, per-node threaded comments.
 *
 * Single-user this round (no real-time sync). Persists to localStorage
 * under `brainstorm-comments-${projectId}`.
 */
import { useCallback, useEffect, useState } from 'react';

export interface Comment {
  id:        string;
  author:    string;
  body:      string;
  createdAt: string;
}

export interface Thread {
  nodeId:   string;
  comments: Comment[];
  resolved: boolean;
}

const KEY = 'brainstorm-comments-';

function load(projectId: string): Thread[] {
  try {
    const raw = localStorage.getItem(KEY + projectId);
    return raw ? (JSON.parse(raw) as Thread[]) : [];
  } catch { return []; }
}
function save(projectId: string, t: Thread[]) {
  localStorage.setItem(KEY + projectId, JSON.stringify(t));
}

export function useComments(projectId: string, author = 'You') {
  const [threads, setThreads] = useState<Thread[]>(() => load(projectId));

  useEffect(() => { setThreads(load(projectId)); }, [projectId]);

  const addComment = useCallback((nodeId: string, body: string) => {
    if (!body.trim()) return;
    setThreads((prev) => {
      const existing = prev.find((t) => t.nodeId === nodeId);
      const newComment: Comment = {
        id: crypto.randomUUID(),
        author,
        body: body.trim(),
        createdAt: new Date().toISOString(),
      };
      let next: Thread[];
      if (existing) {
        next = prev.map((t) =>
          t.nodeId === nodeId ? { ...t, comments: [...t.comments, newComment], resolved: false } : t,
        );
      } else {
        next = [...prev, { nodeId, comments: [newComment], resolved: false }];
      }
      save(projectId, next);
      return next;
    });
  }, [projectId, author]);

  const toggleResolved = useCallback((nodeId: string) => {
    setThreads((prev) => {
      const next = prev.map((t) =>
        t.nodeId === nodeId ? { ...t, resolved: !t.resolved } : t,
      );
      save(projectId, next);
      return next;
    });
  }, [projectId]);

  const deleteThread = useCallback((nodeId: string) => {
    setThreads((prev) => {
      const next = prev.filter((t) => t.nodeId !== nodeId);
      save(projectId, next);
      return next;
    });
  }, [projectId]);

  const getThread = useCallback(
    (nodeId: string) => threads.find((t) => t.nodeId === nodeId),
    [threads],
  );

  const openCount = threads.filter((t) => !t.resolved && t.comments.length > 0).length;

  return { threads, addComment, toggleResolved, deleteThread, getThread, openCount };
}
