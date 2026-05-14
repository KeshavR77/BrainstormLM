/**
 * workspaceStore — zustand store for the Conceptial workspace shell.
 *
 * Holds UI-only state shared across the shell: selected node id, active tool,
 * panel visibility, panel sizes, viewport zoom. Tree state stays in
 * useBrainstorm for now; Phase 4 migrates that into a brainstormStore.
 */
import { create } from 'zustand';
import { persist } from 'zustand/middleware';

export type Tool = 'select' | 'expand' | 'branch' | 'frame' | 'comment' | 'text' | 'search' | 'hand' | 'zoom';

interface WorkspaceState {
  // Selection
  selectedNodeIds: string[];
  hoveredNodeId: string | null;

  // Active tool
  activeTool: Tool;

  // Panel visibility
  showLayers: boolean;
  showInspector: boolean;

  // Panel sizes (percentages)
  layersSize: number;
  inspectorSize: number;

  // Viewport
  zoomPercent: number;

  // Command palette
  commandOpen: boolean;

  // Actions
  setSelection: (ids: string[]) => void;
  toggleSelection: (id: string) => void;
  clearSelection: () => void;
  setHovered: (id: string | null) => void;
  setTool: (t: Tool) => void;
  toggleLayers: () => void;
  toggleInspector: () => void;
  setLayersSize: (n: number) => void;
  setInspectorSize: (n: number) => void;
  setZoom: (n: number) => void;
  openCommand: () => void;
  closeCommand: () => void;
}

export const useWorkspaceStore = create<WorkspaceState>()(
  persist(
    (set) => ({
      selectedNodeIds: [],
      hoveredNodeId:   null,
      activeTool:      'select',
      showLayers:      true,
      showInspector:   true,
      layersSize:      18,
      inspectorSize:   22,
      zoomPercent:     100,
      commandOpen:     false,

      setSelection:    (ids) => set({ selectedNodeIds: ids }),
      toggleSelection: (id)  => set((s) => ({
        selectedNodeIds: s.selectedNodeIds.includes(id)
          ? s.selectedNodeIds.filter((x) => x !== id)
          : [...s.selectedNodeIds, id],
      })),
      clearSelection:  ()    => set({ selectedNodeIds: [] }),
      setHovered:      (id)  => set({ hoveredNodeId: id }),
      setTool:         (t)   => set({ activeTool: t }),
      toggleLayers:    ()    => set((s) => ({ showLayers: !s.showLayers })),
      toggleInspector: ()    => set((s) => ({ showInspector: !s.showInspector })),
      setLayersSize:   (n)   => set({ layersSize: n }),
      setInspectorSize:(n)   => set({ inspectorSize: n }),
      setZoom:         (n)   => set({ zoomPercent: n }),
      openCommand:     ()    => set({ commandOpen: true }),
      closeCommand:    ()    => set({ commandOpen: false }),
    }),
    {
      name: 'conceptial-workspace',
      partialize: (s) => ({
        showLayers:    s.showLayers,
        showInspector: s.showInspector,
        layersSize:    s.layersSize,
        inspectorSize: s.inspectorSize,
        activeTool:    s.activeTool,
      }),
    },
  ),
);
