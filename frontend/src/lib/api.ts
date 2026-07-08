import type { BrainstormEvent, ProjectMemoryCounts } from '../types';
import { mockGeneratePhased } from './mockEngine';

const BASE = import.meta.env.VITE_API_URL ?? '';
const USE_MOCK = import.meta.env.VITE_USE_MOCK_DATA === 'true';

async function generateIdeas(
  prompt: string,
  userPrompt?: string,
  nIdeas = 4,
  projectId?: string,
): Promise<{ summary: string; content: string; trajectory_id: string }[]> {
  const res = await fetch(`${BASE}/brainstorm`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      prompt,
      context: userPrompt ?? null,
      n_ideas: nIdeas,
      use_memory: true,
      project_id: projectId ?? null,
    }),
  });
  if (!res.ok) throw new Error(`API error ${res.status}`);
  const data = await res.json();
  return data.ideas;
}

export async function generateIdeasPhased(
  prompt: string,
  context: string | undefined,
  dispatch: (event: BrainstormEvent) => void,
  projectId?: string,
): Promise<{ summary: string; content: string; trajectory_id: string; idea_id?: string }[]> {
  if (USE_MOCK) {
    return mockGeneratePhased(prompt, context, dispatch, projectId);
  }
  return liveGeneratePhased(prompt, context, dispatch, projectId);
}

async function liveGeneratePhased(
  prompt: string,
  context: string | undefined,
  dispatch: (event: BrainstormEvent) => void,
  projectId?: string,
): Promise<{ summary: string; content: string; trajectory_id: string; idea_id?: string }[]> {
  try {
    const res = await fetch(`${BASE}/brainstorm/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        prompt,
        context: context ?? null,
        n_ideas: 4,
        use_memory: true,
        project_id: projectId ?? null,
      }),
    });

    if (!res.ok) throw new Error(`Stream API error ${res.status}`);

    const reader = res.body!.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    const finalIdeas: { summary: string; content: string; trajectory_id: string; idea_id?: string }[] = [];

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const parts = buffer.split('\n\n');
      buffer = parts.pop() ?? '';

      for (const part of parts) {
        const lines = part.trim().split('\n');
        let eventType = '';
        let data = '';

        for (const line of lines) {
          if (line.startsWith('event: ')) eventType = line.slice(7);
          else if (line.startsWith('data: ')) data = line.slice(6);
        }

        if (!eventType || !data) continue;
        const parsed = JSON.parse(data);

        switch (eventType) {
          case 'raw_idea':
            dispatch({
              type: 'RAW_IDEA_ARRIVED',
              idea: {
                summary: parsed.summary,
                content: parsed.content,
                ideaId: parsed.idea_id,
                pack: parsed.pack,
              },
            });
            break;
          case 'memory':
            dispatch({
              type: 'MEMORY_UPDATED',
              counts: {
                projectId: parsed.project_id,
                seeded: parsed.seeded ?? 0,
                generated: parsed.generated ?? 0,
                total: parsed.total ?? 0,
              },
            });
            break;
          case 'phase':
            if (parsed.phase === 'synthesizing') {
              dispatch({ type: 'DIVERSE_POOL_DONE', count: parsed.pool_size });
            }
            break;
          case 'progress':
            dispatch({ type: 'SYNTHESIS_PROGRESS', progress: parsed.progress });
            break;
          case 'final_idea':
            finalIdeas.push({
              summary: parsed.summary,
              content: parsed.content,
              trajectory_id: parsed.trajectory_id,
              idea_id: parsed.idea_id,
            });
            break;
          case 'done':
            dispatch({ type: 'SYNTHESIS_COMPLETE', ideas: finalIdeas });
            break;
        }
      }
    }

    return finalIdeas;
  } catch {
    // Fall back to non-streaming endpoint
    const ideas = await generateIdeas(prompt, context, 4, projectId);
    dispatch({ type: 'SYNTHESIS_COMPLETE', ideas });
    return ideas;
  }
}

/** Read project memory counts directly (for the chip on canvas mount). */
export async function fetchMemoryCounts(projectId: string): Promise<ProjectMemoryCounts | null> {
  if (USE_MOCK) {
    return readMockMemory(projectId);
  }
  try {
    const res = await fetch(`${BASE}/v1/memory/count?project_id=${encodeURIComponent(projectId)}`);
    if (!res.ok) return null;
    const data = await res.json();
    return {
      projectId: data.project_id,
      seeded: data.seeded ?? 0,
      generated: data.generated ?? 0,
      total: data.total ?? 0,
    };
  } catch {
    return null;
  }
}

/** Reset a project's memory (clear all seeded + generated entries). */
export async function resetProjectMemory(projectId: string): Promise<number> {
  if (USE_MOCK) {
    clearMockMemory(projectId);
    return 0;
  }
  try {
    const res = await fetch(`${BASE}/v1/memory/reset?project_id=${encodeURIComponent(projectId)}`, {
      method: 'POST',
    });
    if (!res.ok) return 0;
    const data = await res.json();
    return data.deleted ?? 0;
  } catch {
    return 0;
  }
}

const MOCK_MEMORY_KEY = (projectId: string) => `mock-memory-${projectId}`;

// Mirror of the server-side seeded counts in artifacts/conceptial/seeds/*.json
// so the chip shows the right number when running offline (Render mock build).
const SEEDED_MOCK_COUNTS: Record<string, number> = {
  'demo-skincare-identity': 12,
  'demo-footwear-campaign': 8,
  'demo-oatmilk-packaging': 8,
};

function readMockMemory(projectId: string): ProjectMemoryCounts {
  const seededFallback = SEEDED_MOCK_COUNTS[projectId] ?? 0;
  try {
    const raw = localStorage.getItem(MOCK_MEMORY_KEY(projectId));
    if (raw) {
      const parsed = JSON.parse(raw);
      const seeded = parsed.seeded ?? seededFallback;
      const generated = parsed.generated ?? 0;
      return { projectId, seeded, generated, total: seeded + generated };
    }
  } catch {
    /* ignore */
  }
  return {
    projectId,
    seeded: seededFallback,
    generated: 0,
    total: seededFallback,
  };
}

function clearMockMemory(projectId: string): void {
  try {
    localStorage.removeItem(MOCK_MEMORY_KEY(projectId));
  } catch {
    /* ignore */
  }
}

export async function submitFeedback(
  trajectoryId: string,
  thumbs: 'up' | 'down',
  rating?: number,
  critique?: string,
): Promise<void> {
  if (USE_MOCK) return;
  try {
    await fetch(`${BASE}/feedback`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        trajectory_id: trajectoryId,
        thumbs,
        rating: rating ?? null,
        critique: critique ?? null,
      }),
    });
  } catch {
    // Feedback is best-effort — never block the UI on a network failure
  }
}
