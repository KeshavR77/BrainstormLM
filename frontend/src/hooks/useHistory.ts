/**
 * useHistory — undo/redo wrapper around useBrainstorm.
 *
 * Observes the tree state and maintains a bounded history stack. Streaming
 * generations are coalesced into a single undo step by ignoring intermediate
 * changes while phase !== IDLE/COMPLETE.
 *
 * This is a deliberately small/non-invasive Phase 4 — full zustand+immer
 * patches migration deferred. Trade-off: each undo entry stores a full tree
 * snapshot (cheap until 100s of nodes; revisit when that lands).
 */
import { useCallback, useEffect, useRef, useState } from 'react';
import type { BrainstormTree } from '../types';
import type { BrainstormPhase } from '../lib/brainstormMachine';

const MAX_HISTORY = 50;

interface Options {
  tree: BrainstormTree;
  phase: BrainstormPhase;
  replaceTree: (next: BrainstormTree) => void;
}

export function useHistory({ tree, phase, replaceTree }: Options) {
  const [past, setPast]     = useState<BrainstormTree[]>([]);
  const [future, setFuture] = useState<BrainstormTree[]>([]);
  const lastCommitted = useRef<BrainstormTree>(tree);
  const restoring = useRef(false);

  /* Commit a new history entry when tree changes — but only when phase
     is settled (IDLE/COMPLETE/ERROR), so streaming = single undo step. */
  useEffect(() => {
    if (restoring.current) {
      restoring.current = false;
      lastCommitted.current = tree;
      return;
    }
    const settled = phase.status === 'IDLE' || phase.status === 'COMPLETE' || phase.status === 'ERROR';
    if (!settled) return;
    if (tree === lastCommitted.current) return;
    // Skip the very first commit (load).
    if (lastCommitted.current && lastCommitted.current !== tree) {
      setPast((p) => {
        const next = [...p, lastCommitted.current];
        return next.length > MAX_HISTORY ? next.slice(-MAX_HISTORY) : next;
      });
      setFuture([]);
    }
    lastCommitted.current = tree;
  }, [tree, phase.status]);

  const undo = useCallback(() => {
    if (past.length === 0) return;
    const prev = past[past.length - 1];
    setPast((p) => p.slice(0, -1));
    setFuture((f) => [tree, ...f].slice(0, MAX_HISTORY));
    restoring.current = true;
    replaceTree(prev);
  }, [past, tree, replaceTree]);

  const redo = useCallback(() => {
    if (future.length === 0) return;
    const next = future[0];
    setFuture((f) => f.slice(1));
    setPast((p) => [...p, tree].slice(-MAX_HISTORY));
    restoring.current = true;
    replaceTree(next);
  }, [future, tree, replaceTree]);

  return {
    undo,
    redo,
    canUndo: past.length > 0,
    canRedo: future.length > 0,
    pastSize: past.length,
    futureSize: future.length,
  };
}
