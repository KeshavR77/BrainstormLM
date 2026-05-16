import { useCallback, useEffect, useReducer, useRef, useState } from 'react';
import type {
  BrainstormNode,
  BrainstormEdge,
  BrainstormTree,
  ProjectMemoryCounts,
} from '../types';
import {
  fetchMemoryCounts,
  generateIdeasPhased,
  resetProjectMemory,
  submitFeedback,
} from '../lib/api';
import { brainstormReducer, IDLE, isLoading } from '../lib/brainstormMachine';
import { DEMO_PROJECT_ID, DEMO_TREE } from '../lib/demoProject';

function storageKey(projectId: string) {
  return `brainstorm-tree-${projectId}`;
}

function isEmptyTree(t: BrainstormTree): boolean {
  return t.rootId == null || Object.keys(t.nodes).length === 0;
}

function loadTree(projectId: string): BrainstormTree {
  try {
    const raw = localStorage.getItem(storageKey(projectId));
    if (raw) {
      const parsed = JSON.parse(raw) as BrainstormTree;
      if (projectId === DEMO_PROJECT_ID && isEmptyTree(parsed)) return DEMO_TREE;
      return parsed;
    }
    if (projectId === DEMO_PROJECT_ID) return DEMO_TREE;
    return { nodes: {}, edges: [], rootId: null };
  } catch {
    if (projectId === DEMO_PROJECT_ID) return DEMO_TREE;
    return { nodes: {}, edges: [], rootId: null };
  }
}

function saveTree(projectId: string, tree: BrainstormTree) {
  localStorage.setItem(storageKey(projectId), JSON.stringify(tree));
}

export function useBrainstorm(projectId: string) {
  const [tree, setTree] = useState<BrainstormTree>(() => loadTree(projectId));
  const [phase, dispatch] = useReducer(brainstormReducer, IDLE);
  const [activeNodeId, setActiveNodeId] = useState<string | null>(null);
  const [memoryCounts, setMemoryCounts] = useState<ProjectMemoryCounts>({
    projectId,
    seeded: 0,
    generated: 0,
    total: 0,
  });
  const treeRef = useRef(tree);

  const loading = isLoading(phase);

  useEffect(() => {
    treeRef.current = tree;
  }, [tree]);

  useEffect(() => {
    saveTree(projectId, tree);
  }, [projectId, tree]);

  /* Fetch memory counts on mount + when projectId changes */
  useEffect(() => {
    let alive = true;
    fetchMemoryCounts(projectId).then((c) => {
      if (alive && c) setMemoryCounts(c);
    });
    return () => {
      alive = false;
    };
  }, [projectId]);

  /* Intercept MEMORY_UPDATED events from the SSE stream so the chip ticks live */
  const dispatchWithMemory = useCallback(
    (event: Parameters<typeof dispatch>[0]) => {
      if (event.type === 'MEMORY_UPDATED') {
        setMemoryCounts(event.counts);
      }
      dispatch(event);
    },
    [],
  );

  const resetMemory = useCallback(async () => {
    await resetProjectMemory(projectId);
    setMemoryCounts({ projectId, seeded: 0, generated: 0, total: 0 });
  }, [projectId]);

  // ── Expand a node (generate children) ─────────────────────────

  const expandNodeInner = useCallback(
    async (nodeId: string, userPrompt?: string, currentTree?: BrainstormTree) => {
      const t = currentTree ?? treeRef.current;
      const node = t.nodes[nodeId];
      if (!node || node.isExpanded) return;

      dispatchWithMemory({ type: 'START_GENERATION', prompt: node.content });
      try {
        const ideas = await generateIdeasPhased(node.content, userPrompt, dispatchWithMemory, projectId);
        const newNodes: Record<string, BrainstormNode> = {};
        const newEdges: BrainstormEdge[] = [];
        const childIds: string[] = [];

        for (const idea of ideas) {
          const childId = crypto.randomUUID();
          childIds.push(childId);

          newNodes[childId] = {
            id: childId,
            parentId: nodeId,
            type: 'idea',
            content: idea.content,
            summary: idea.summary,
            depth: node.depth + 1,
            createdAt: new Date().toISOString(),
            isExpanded: false,
            children: [],
            trajectoryId: idea.trajectory_id,
            ideaId: idea.idea_id,
            feedbackGiven: undefined,
          };

          newEdges.push({
            id: `e-${nodeId}-${childId}`,
            source: nodeId,
            target: childId,
          });
        }

        setTree((prev) => ({
          ...prev,
          nodes: {
            ...prev.nodes,
            [nodeId]: { ...prev.nodes[nodeId], isExpanded: true, children: childIds },
            ...newNodes,
          },
          edges: [...prev.edges, ...newEdges],
        }));
      } catch (err) {
        dispatchWithMemory({
          type: 'ERROR',
          message: err instanceof Error ? err.message : 'Generation failed',
        });
      }
    },
    [dispatchWithMemory, projectId],
  );

  // ── Submit seed ────────────────────────────────────────────────

  const submitSeed = useCallback(
    async (text: string) => {
      const seedId = crypto.randomUUID();
      const seed: BrainstormNode = {
        id: seedId,
        parentId: null,
        type: 'seed',
        content: text.trim(),
        summary: text.trim().slice(0, 60) + (text.length > 60 ? '…' : ''),
        depth: 0,
        createdAt: new Date().toISOString(),
        isExpanded: false,
        children: [],
      };

      setTree({ nodes: { [seedId]: seed }, edges: [], rootId: seedId });
      setActiveNodeId(seedId);

      await expandNodeInner(seedId, undefined, {
        nodes: { [seedId]: seed },
        edges: [],
        rootId: seedId,
      });
    },
    [expandNodeInner],
  );

  const expandNode = useCallback(
    async (nodeId: string) => {
      await expandNodeInner(nodeId);
    },
    [expandNodeInner],
  );

  // ── Expand with user prompt ───────────────────────────────────

  const expandWithPrompt = useCallback(
    async (nodeId: string, userPrompt: string) => {
      await expandNodeInner(nodeId, userPrompt);
    },
    [expandNodeInner],
  );

  // ── Select an idea — fires DPO preference pairs automatically ─

  const selectIdea = useCallback(
    (nodeId: string) => {
      const node = treeRef.current.nodes[nodeId];
      if (!node || node.type === 'seed') return;

      if (node.trajectoryId) {
        submitFeedback(node.trajectoryId, 'up');
      }

      const parentNode = node.parentId ? treeRef.current.nodes[node.parentId] : null;
      if (parentNode) {
        for (const sibId of parentNode.children) {
          if (sibId !== nodeId) {
            const sib = treeRef.current.nodes[sibId];
            if (sib?.trajectoryId && !sib.feedbackGiven) {
              submitFeedback(sib.trajectoryId, 'down');
            }
          }
        }
      }

      setTree((prev) => {
        const updated = { ...prev, nodes: { ...prev.nodes } };
        updated.nodes[nodeId] = {
          ...updated.nodes[nodeId],
          type: 'selected',
          feedbackGiven: 'up',
        };
        if (parentNode) {
          for (const sibId of parentNode.children) {
            if (sibId !== nodeId && updated.nodes[sibId] && !updated.nodes[sibId].feedbackGiven) {
              updated.nodes[sibId] = { ...updated.nodes[sibId], feedbackGiven: 'down' };
            }
          }
        }
        return updated;
      });

      setActiveNodeId(nodeId);
    },
    [],
  );

  // ── Manual feedback (explicit thumb press on a node) ──────────

  const submitNodeFeedback = useCallback(
    (nodeId: string, thumbs: 'up' | 'down') => {
      const node = treeRef.current.nodes[nodeId];
      if (!node?.trajectoryId || node.feedbackGiven) return;

      submitFeedback(node.trajectoryId, thumbs);

      setTree((prev) => ({
        ...prev,
        nodes: {
          ...prev.nodes,
          [nodeId]: { ...prev.nodes[nodeId], feedbackGiven: thumbs },
        },
      }));
    },
    [],
  );

  // ── Submit a critique note for a selected node ────────────────

  const submitCritique = useCallback(
    (nodeId: string, critique: string) => {
      const node = treeRef.current.nodes[nodeId];
      if (!node?.trajectoryId || !critique.trim()) return;
      submitFeedback(node.trajectoryId, 'up', undefined, critique.trim());
    },
    [],
  );

  // ── Edit a node's content ─────────────────────────────────────

  const editNode = useCallback(
    (nodeId: string, content: string, summary?: string) => {
      setTree((prev) => ({
        ...prev,
        nodes: {
          ...prev.nodes,
          [nodeId]: {
            ...prev.nodes[nodeId],
            content,
            summary: summary ?? content.slice(0, 60) + (content.length > 60 ? '…' : ''),
          },
        },
      }));
    },
    [],
  );

  // ── Add a custom idea as a child ──────────────────────────────

  const addCustomIdea = useCallback(
    (parentId: string, content: string, summary: string) => {
      const parent = treeRef.current.nodes[parentId];
      if (!parent) return;

      const childId = crypto.randomUUID();

      setTree((prev) => ({
        ...prev,
        nodes: {
          ...prev.nodes,
          [parentId]: {
            ...prev.nodes[parentId],
            isExpanded: true,
            children: [...prev.nodes[parentId].children, childId],
          },
          [childId]: {
            id: childId,
            parentId,
            type: 'idea' as const,
            content: content.trim(),
            summary: summary.trim() || content.trim().slice(0, 40),
            depth: parent.depth + 1,
            createdAt: new Date().toISOString(),
            isExpanded: false,
            children: [],
          },
        },
        edges: [
          ...prev.edges,
          { id: `e-${parentId}-${childId}`, source: parentId, target: childId },
        ],
      }));
    },
    [],
  );

  // ── Computed values ───────────────────────────────────────────

  const nodeCount = Object.keys(tree.nodes).length;
  const maxDepth = Object.values(tree.nodes).reduce(
    (max, n) => Math.max(max, n.depth),
    0,
  );

  /** Replace the entire tree (used by history undo/redo and snapshot restore). */
  const replaceTree = useCallback((next: BrainstormTree) => {
    setTree(next);
  }, []);

  /* Toggle a node's "explored" flag — UI dims it to communicate that
     the user has consciously moved past this branch. */
  const toggleExplored = useCallback((nodeId: string) => {
    setTree((prev) => {
      const node = prev.nodes[nodeId];
      if (!node) return prev;
      return {
        ...prev,
        nodes: {
          ...prev.nodes,
          [nodeId]: { ...node, isExplored: !node.isExplored },
        },
      };
    });
  }, []);

  return {
    tree,
    phase,
    loading,
    activeNodeId,
    setActiveNodeId,
    submitSeed,
    expandNode,
    expandWithPrompt,
    selectIdea,
    editNode,
    addCustomIdea,
    submitNodeFeedback,
    submitCritique,
    toggleExplored,
    replaceTree,
    memoryCounts,
    resetMemory,
    nodeCount,
    maxDepth,
    hasRoot: tree.rootId !== null,
  };
}
