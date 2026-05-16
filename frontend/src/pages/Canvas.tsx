/**
 * Canvas — Conceptial design workspace.
 *
 * Wraps the React Flow brainstorm canvas in the new ShellLayout
 * (TopBar / LayersPanel / Canvas / InspectorStub / StatusBar) and
 * the floating ToolBar pill. Selection + tool state lives in
 * useWorkspaceStore (zustand). Tree state stays in useBrainstorm
 * (migrated to zustand in Phase 4).
 */
import {
  ReactFlow,
  ReactFlowProvider,
  Controls,
  Background,
  BackgroundVariant,
  MiniMap,
  useReactFlow,
  applyNodeChanges,
  applyEdgeChanges,
  type Node as FlowNode,
  type Edge as FlowEdge,
  type OnNodesChange,
  type OnEdgesChange,
} from '@xyflow/react';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import { nodeTypes } from '../components/nodes';
import { ShellLayout } from '../components/workspace/ShellLayout';
import { TopBar } from '../components/workspace/TopBar';
import { ToolBar } from '../components/workspace/ToolBar';
import { LayersPanel } from '../components/workspace/LayersPanel';
import { StatusBar } from '../components/workspace/StatusBar';
import { PropertiesPanel } from '../components/workspace/PropertiesPanel';
import { CommandPalette } from '../components/workspace/CommandPalette';
import { ShortcutsSheet } from '../components/workspace/ShortcutsSheet';
import { HistoryDrawer } from '../components/workspace/HistoryDrawer';
import { CommentsPanel } from '../components/workspace/CommentsPanel';
import { EmptyCanvas } from '../components/workspace/EmptyCanvas';
import { useHotkeys } from '../hooks/useHotkeys';
import { useHistory } from '../hooks/useHistory';
import { useSnapshots } from '../hooks/useSnapshots';
import { useComments } from '../hooks/useComments';
import { SteeringTelemetry } from '../components/SteeringTelemetry';
import { MemoryChip } from '../components/MemoryChip';
import { useBrainstorm } from '../hooks/useBrainstorm';
import { useProjects } from '../hooks/useProjects';
import { useWorkspaceStore } from '../stores/workspaceStore';
import { layoutTree } from '../layoutTree';
import { depthHue } from '../styles/motion';

interface CanvasPageProps {
  projectId: string;
  onBack: () => void;
}

function CanvasInner({ projectId, onBack }: CanvasPageProps) {
  const {
    tree, phase,
    submitSeed, expandNode, expandWithPrompt, selectIdea,
    editNode, addCustomIdea, submitNodeFeedback, submitCritique,
    replaceTree,
    memoryCounts, resetMemory,
    nodeCount, maxDepth, hasRoot,
  } = useBrainstorm(projectId);

  /* Undo/redo + named snapshots + comments */
  const history   = useHistory({ tree, phase, replaceTree });
  const snapshots = useSnapshots(projectId);
  const comments  = useComments(projectId);
  const [historyOpen, setHistoryOpen]   = useState(false);
  const [commentsOpen, setCommentsOpen] = useState(false);

  const { getProject, updateProject } = useProjects();
  const project = getProject(projectId);
  const { fitView, getNode } = useReactFlow();

  const setSelection      = useWorkspaceStore((s) => s.setSelection);
  const setZoom           = useWorkspaceStore((s) => s.setZoom);
  const selectedNodeIds   = useWorkspaceStore((s) => s.selectedNodeIds);
  const showInspector     = useWorkspaceStore((s) => s.showInspector);
  const showLayers        = useWorkspaceStore((s) => s.showLayers);
  const setShowInspector  = useWorkspaceStore.setState;
  const activeTool        = useWorkspaceStore((s) => s.activeTool);

  const [flowNodes, setFlowNodes] = useState<FlowNode[]>([]);
  const [flowEdges, setFlowEdges] = useState<FlowEdge[]>([]);
  const prevNodeCount = useRef(0);

  /* Persist node count to project */
  useEffect(() => {
    if (nodeCount > 0) updateProject(projectId, { nodeCount });
  }, [nodeCount, projectId, updateProject]);

  /* Build React Flow nodes/edges from brainstorm tree */
  const { builtNodes, builtEdges } = useMemo(() => {
    const fNodes: FlowNode[] = [];
    const fEdges: FlowEdge[] = [];

    for (const node of Object.values(tree.nodes)) {
      const base = { id: node.id, position: { x: 0, y: 0 } };
      switch (node.type) {
        case 'seed':
          fNodes.push({ ...base, type: 'seed', data: { content: node.content, submitted: true, onSubmit: submitSeed }});
          break;
        case 'idea':
          fNodes.push({ ...base, type: 'idea', data: {
            summary: node.summary, content: node.content, depth: node.depth,
            isExpanded: node.isExpanded, hasChildren: node.children.length > 0,
            onSelect: selectIdea, onExpand: expandNode, onEdit: editNode,
            onFeedback: submitNodeFeedback,
            nodeId: node.id, trajectoryId: node.trajectoryId, feedbackGiven: node.feedbackGiven,
          }});
          break;
        case 'selected':
          fNodes.push({ ...base, type: 'selected', data: {
            summary: node.summary, content: node.content, depth: node.depth,
            isExpanded: node.isExpanded, onExpandWithPrompt: expandWithPrompt,
            onAutoExpand: expandNode, onAddCustom: addCustomIdea,
            onCritique: submitCritique, nodeId: node.id,
          }});
          break;
      }
    }

    for (const edge of tree.edges) {
      const targetNode = tree.nodes[edge.target];
      const hue = depthHue(targetNode?.depth ?? 1);
      fEdges.push({
        id: edge.id,
        source: edge.source,
        target: edge.target,
        animated: true,
        style: { stroke: `oklch(0.62 0.20 ${hue})`, strokeWidth: 1.4, opacity: 0.55 },
      });
    }
    return { builtNodes: fNodes, builtEdges: fEdges };
  }, [tree, submitSeed, selectIdea, expandNode, expandWithPrompt, editNode, addCustomIdea, submitNodeFeedback, submitCritique]);

  /* Preview nodes during diverse pool phase */
  const previewNodes = useMemo<FlowNode[]>(() => {
    if (phase.status !== 'GENERATING_DIVERSE_POOL' || !tree.rootId) return [];
    return phase.rawIdeas.map((idea, i) => ({
      id: `preview-${i}`,
      type: 'preview' as const,
      position: { x: 0, y: 0 },
      data: { summary: idea.summary, content: idea.content },
    }));
  }, [phase, tree.rootId]);

  const previewEdges = useMemo<FlowEdge[]>(() => {
    if (phase.status !== 'GENERATING_DIVERSE_POOL' || !tree.rootId) return [];
    const lastExpandedNodeId = Object.values(tree.nodes)
      .filter((n) => n.type === 'seed' || n.type === 'selected')
      .sort((a, b) => b.depth - a.depth)[0]?.id ?? tree.rootId;
    return previewNodes.map((pn) => ({
      id: `pe-${lastExpandedNodeId}-${pn.id}`,
      source: lastExpandedNodeId,
      target: pn.id,
      animated: true,
      style: { stroke: 'var(--brand)', strokeDasharray: '4 4', opacity: 0.4 },
    }));
  }, [previewNodes, tree.rootId, tree.nodes, phase.status]);

  /* Apply layout */
  useEffect(() => {
    const allNodes = [...builtNodes, ...previewNodes];
    const allEdges = [...builtEdges, ...previewEdges];
    if (allNodes.length > 0) {
      const laid = layoutTree(allNodes, allEdges);
      setFlowNodes(laid);
      setFlowEdges(allEdges);
    }
  }, [builtNodes, builtEdges, previewNodes, previewEdges]);

  /* Auto-fit on node count change. Clamp minZoom so a single small node
     can't shrink the viewport to ~10% (Bug #1). */
  useEffect(() => {
    if (flowNodes.length === 0) return;
    const grew = flowNodes.length !== prevNodeCount.current;
    if (!grew) return;
    prevNodeCount.current = flowNodes.length;
    const t1 = window.setTimeout(() => {
      fitView({ padding: 0.35, duration: 500, minZoom: 0.45, maxZoom: 1.4 });
    }, 200);
    /* Second pass after layout settles (React Flow sometimes measures
       node bounds incorrectly on the first paint). */
    const t2 = window.setTimeout(() => {
      fitView({ padding: 0.35, duration: 400, minZoom: 0.45, maxZoom: 1.4 });
    }, 700);
    return () => { window.clearTimeout(t1); window.clearTimeout(t2); };
  }, [flowNodes, fitView]);

  const onNodesChange: OnNodesChange = useCallback(
    (changes) => setFlowNodes((nds) => applyNodeChanges(changes, nds)),
    [],
  );
  const onEdgesChange: OnEdgesChange = useCallback(
    (changes) => setFlowEdges((eds) => applyEdgeChanges(changes, eds)),
    [],
  );

  const onSelectionChange = useCallback(
    ({ nodes }: { nodes: FlowNode[] }) => {
      setSelection(nodes.map((n) => n.id));
    },
    [setSelection],
  );

  /* Layers panel → focus a node */
  const focusNode = useCallback(
    (id: string) => {
      const flowNode = getNode(id);
      if (flowNode) {
        fitView({ nodes: [flowNode], padding: 0.4, duration: 400, maxZoom: 1.4 });
      }
    },
    [fitView, getNode],
  );

  /* Hotkeys + shortcut sheet */
  const [shortcutsOpen, setShortcutsOpen] = useState(false);
  const saveSnapshot = useCallback(() => {
    const label = window.prompt('Snapshot name:', `Snapshot ${new Date().toLocaleTimeString()}`);
    if (label !== null) snapshots.create(label, tree);
  }, [snapshots, tree]);

  useHotkeys({
    onShowShortcuts: () => setShortcutsOpen(true),
    onUndo: history.undo,
    onRedo: history.redo,
    onSaveSnapshot: saveSnapshot,
    onFitView: () => fitView({ padding: 0.3, duration: 400 }),
  });

  /* Auto-show inspector when a node gets selected from a closed state. */
  useEffect(() => {
    if (selectedNodeIds.length > 0 && !showInspector) {
      setShowInspector({ showInspector: true });
    }
  }, [selectedNodeIds, showInspector, setShowInspector]);

  /* Picking the "C" tool opens the Comments panel. Snap tool back to select. */
  useEffect(() => {
    if (activeTool === 'comment') {
      setCommentsOpen(true);
      useWorkspaceStore.setState({ activeTool: 'select' });
    }
  }, [activeTool]);

  /* Refit canvas after side-panel visibility changes so the tree recenters
     in the (now larger or smaller) canvas area. */
  useEffect(() => {
    if (!hasRoot) return;
    const t = window.setTimeout(() => {
      fitView({ padding: 0.35, duration: 300, minZoom: 0.45, maxZoom: 1.4 });
    }, 320); // give react-resizable-panels time to settle
    return () => window.clearTimeout(t);
  }, [showLayers, showInspector, hasRoot, fitView]);

  /* Display nodes are only ever the real flow nodes — the empty state is
     rendered separately so React Flow doesn't shrink a single seed to 10%. */
  const displayNodes: FlowNode[] = flowNodes;

  const fitViewAll = useCallback(() => fitView({ padding: 0.3, duration: 400 }), [fitView]);

  return (
    <>
    <ShellLayout
      topBar={
        <TopBar
          projectName={project?.name ?? 'Untitled'}
          phase={phase}
          onBack={onBack}
          onUndo={history.undo}
          onRedo={history.redo}
          canUndo={history.canUndo}
          canRedo={history.canRedo}
          onOpenHistory={() => setHistoryOpen(true)}
          onOpenComments={() => setCommentsOpen(true)}
          commentsCount={comments.openCount}
        />
      }
      layers={<LayersPanel tree={tree} onFocusNode={focusNode} />}
      inspector={
        <PropertiesPanel
          tree={tree}
          onEditNode={editNode}
          onRegenerate={expandNode}
          onFeedback={submitNodeFeedback}
        />
      }
      statusBar={<StatusBar phase={phase} nodeCount={nodeCount} maxDepth={maxDepth} />}
      canvas={
        <>
          {hasRoot && <ToolBar />}
          {!hasRoot ? (
            <EmptyCanvas
              defaultText={project?.description ?? ''}
              onSubmit={submitSeed}
            />
          ) : (
          <div className="h-full w-full conceptial-canvas">
            <ReactFlow
              nodes={displayNodes}
              edges={flowEdges}
              nodeTypes={nodeTypes}
              onNodesChange={onNodesChange}
              onEdgesChange={onEdgesChange}
              onSelectionChange={onSelectionChange}
              onMove={(_, v) => setZoom(Math.round(v.zoom * 100))}
              minZoom={0.1}
              maxZoom={1.8}
              fitView
              fitViewOptions={{ padding: 0.5 }}
              proOptions={{ hideAttribution: true }}
              className="bg-paper"
            >
              <Background
                variant={BackgroundVariant.Dots}
                gap={28}
                size={1.1}
                color="rgba(22, 19, 24, 0.10)"
              />
              <Controls position="bottom-left" showInteractive={false} />
              <MiniMap
                position="bottom-right"
                pannable
                zoomable
                nodeColor={(n) => {
                  const treeNode = tree.nodes[n.id];
                  if (!treeNode) return 'var(--ink-3)';
                  return `oklch(0.62 0.20 ${depthHue(treeNode.depth)})`;
                }}
                maskColor="rgba(244, 239, 230, 0.7)"
                style={{ background: 'var(--paper-1)', border: '1px solid var(--ink-line)', borderRadius: 8 }}
              />
            </ReactFlow>

            <SteeringTelemetry phase={phase} />
            <MemoryChip counts={memoryCounts} onReset={resetMemory} />
          </div>
          )}
        </>
      }
    />

    <CommandPalette
      tree={tree}
      onFocusNode={focusNode}
      onNavigateHome={onBack}
      onFitView={fitViewAll}
      onSnapshot={saveSnapshot}
    />
    <ShortcutsSheet open={shortcutsOpen} onClose={() => setShortcutsOpen(false)} />
    <HistoryDrawer
      open={historyOpen}
      onClose={() => setHistoryOpen(false)}
      snapshots={snapshots.snapshots}
      onCreate={(label) => snapshots.create(label, tree)}
      onRestore={(id) => { const t = snapshots.restore(id); if (t) replaceTree(t); }}
      onDelete={snapshots.remove}
    />
    <CommentsPanel
      open={commentsOpen}
      onClose={() => setCommentsOpen(false)}
      tree={tree}
      threads={comments.threads}
      onAdd={comments.addComment}
      onResolve={comments.toggleResolved}
      onDelete={comments.deleteThread}
      onFocusNode={focusNode}
    />
    </>
  );
}

export function Canvas(props: CanvasPageProps) {
  return (
    <ReactFlowProvider>
      <CanvasInner {...props} />
    </ReactFlowProvider>
  );
}
