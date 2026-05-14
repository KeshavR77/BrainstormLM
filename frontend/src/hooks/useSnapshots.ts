/**
 * useSnapshots — named tree snapshots, compressed and persisted per project.
 *
 * Uses lz-string for storage compression so snapshots don't blow past
 * localStorage 5MB. Caps at 20 entries/project (oldest evicted).
 */
import { useCallback, useEffect, useState } from 'react';
import lz from 'lz-string';
import type { BrainstormTree } from '../types';

const KEY_PREFIX = 'brainstorm-snapshots-';
const MAX_SNAPSHOTS = 20;

export interface Snapshot {
  id: string;
  label: string;
  createdAt: string;
  nodeCount: number;
  /** lz-compressed JSON of the tree */
  payload: string;
}

function keyFor(projectId: string) {
  return KEY_PREFIX + projectId;
}

function load(projectId: string): Snapshot[] {
  try {
    const raw = localStorage.getItem(keyFor(projectId));
    if (!raw) return [];
    return JSON.parse(raw) as Snapshot[];
  } catch { return []; }
}

function save(projectId: string, snaps: Snapshot[]) {
  localStorage.setItem(keyFor(projectId), JSON.stringify(snaps));
}

export function useSnapshots(projectId: string) {
  const [snapshots, setSnapshots] = useState<Snapshot[]>(() => load(projectId));

  useEffect(() => {
    setSnapshots(load(projectId));
  }, [projectId]);

  const create = useCallback((label: string, tree: BrainstormTree) => {
    const snap: Snapshot = {
      id: crypto.randomUUID(),
      label: label.trim() || `Snapshot ${new Date().toLocaleString()}`,
      createdAt: new Date().toISOString(),
      nodeCount: Object.keys(tree.nodes).length,
      payload: lz.compressToUTF16(JSON.stringify(tree)),
    };
    setSnapshots((prev) => {
      const next = [snap, ...prev].slice(0, MAX_SNAPSHOTS);
      save(projectId, next);
      return next;
    });
    return snap;
  }, [projectId]);

  const restore = useCallback((id: string): BrainstormTree | null => {
    const s = snapshots.find((x) => x.id === id);
    if (!s) return null;
    try {
      const decompressed = lz.decompressFromUTF16(s.payload);
      if (!decompressed) return null;
      return JSON.parse(decompressed) as BrainstormTree;
    } catch { return null; }
  }, [snapshots]);

  const remove = useCallback((id: string) => {
    setSnapshots((prev) => {
      const next = prev.filter((x) => x.id !== id);
      save(projectId, next);
      return next;
    });
  }, [projectId]);

  return { snapshots, create, restore, remove };
}
