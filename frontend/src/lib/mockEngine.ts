/**
 * mockEngine — drives the canvas without a backend.
 *
 * Mirrors the live SSE protocol shape: emits raw_idea + memory + final ideas
 * so the canvas UI behaves identically in mock mode and live mode. Project
 * memory persists across regenerations via localStorage so the "explored set"
 * effect is real even in mock mode (critical for Render deploys where
 * VITE_USE_MOCK_DATA=true is the default).
 */

import type { BrainstormEvent } from '../types';
import packsData from '../mocks/brainstorm-packs.json';

interface MockPack {
  theme: string;
  diversePool: { summary: string; content: string; temperature: number; creativityScore: number }[];
  synthesizedIndices: number[];
}

const packs = packsData.packs as MockPack[];

function simpleHash(text: string): number {
  let hash = 0;
  for (let i = 0; i < text.length; i++) {
    hash = ((hash << 5) - hash + text.charCodeAt(i)) | 0;
  }
  return Math.abs(hash);
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// ─── Mock project memory (localStorage) ──────────────────────────────────

interface MockMemoryEntry {
  seeded: number;
  generated: number;
  seenSummaries: string[];
}

const MEMORY_KEY = (projectId: string) => `mock-memory-${projectId}`;

function loadMemory(projectId: string): MockMemoryEntry {
  try {
    const raw = localStorage.getItem(MEMORY_KEY(projectId));
    if (raw) {
      const parsed = JSON.parse(raw);
      return {
        seeded: parsed.seeded ?? 0,
        generated: parsed.generated ?? 0,
        seenSummaries: parsed.seenSummaries ?? [],
      };
    }
  } catch {
    /* ignore */
  }
  return { seeded: 0, generated: 0, seenSummaries: [] };
}

function saveMemory(projectId: string, entry: MockMemoryEntry): void {
  try {
    localStorage.setItem(MEMORY_KEY(projectId), JSON.stringify(entry));
  } catch {
    /* ignore */
  }
}

function norm(s: string): string {
  return s.trim().toLowerCase().replace(/\s+/g, ' ');
}

export async function mockGeneratePhased(
  prompt: string,
  _context: string | undefined,
  dispatch: (event: BrainstormEvent) => void,
  projectId?: string,
): Promise<{ summary: string; content: string; trajectory_id: string; idea_id?: string }[]> {
  const packIdx = simpleHash(prompt) % packs.length;
  const pack = packs[packIdx];
  const pid = projectId ?? `anon-${simpleHash(prompt)}`;
  const memory = loadMemory(pid);
  const seen = new Set(memory.seenSummaries.map(norm));

  // Emit initial memory state (matches live SSE shape)
  dispatch({
    type: 'MEMORY_UPDATED',
    counts: {
      projectId: pid,
      seeded: memory.seeded,
      generated: memory.generated,
      total: memory.seeded + memory.generated,
    },
  });

  // Filter the pool against what the project has already seen — anti-mode
  // exclusion at the mock layer.
  const fresh = pack.diversePool.filter((item) => !seen.has(norm(item.summary)));
  // If we've exhausted the pack, allow re-use of the originals to keep the
  // demo flowing — but mark them with a different temperature so the user
  // can see they're rerolls.
  const queue = fresh.length > 0 ? fresh : pack.diversePool;

  const emittedSummaries: string[] = [];

  for (const item of queue) {
    if (emittedSummaries.length >= 5) break;
    const delay = 800 + Math.random() * 400;
    await sleep(delay);
    const ideaId = crypto.randomUUID();
    dispatch({
      type: 'RAW_IDEA_ARRIVED',
      idea: {
        summary: item.summary,
        content: item.content,
        temperature: item.temperature,
        creativityScore: item.creativityScore,
        ideaId,
      },
    });
    emittedSummaries.push(item.summary);

    // Tick the memory chip
    const newGenerated = memory.generated + emittedSummaries.length;
    dispatch({
      type: 'MEMORY_UPDATED',
      counts: {
        projectId: pid,
        seeded: memory.seeded,
        generated: newGenerated,
        total: memory.seeded + newGenerated,
      },
    });
  }

  // Persist new exclusions to memory
  saveMemory(pid, {
    seeded: memory.seeded,
    generated: memory.generated + emittedSummaries.length,
    seenSummaries: [...memory.seenSummaries, ...emittedSummaries],
  });

  dispatch({ type: 'DIVERSE_POOL_DONE', count: emittedSummaries.length });

  const progressSteps = [0.33, 0.66, 1.0];
  for (const p of progressSteps) {
    await sleep(220);
    dispatch({ type: 'SYNTHESIS_PROGRESS', progress: p });
  }

  // Build the final list from whatever we actually emitted (so synthesis
  // pulls from a non-empty pool even after the pack is exhausted).
  const usedItems = pack.diversePool.filter((it) => emittedSummaries.includes(it.summary));
  const synthesized = usedItems.map((item) => ({
    summary: item.summary,
    content: item.content,
    trajectory_id: crypto.randomUUID(),
    idea_id: crypto.randomUUID(),
  }));

  dispatch({ type: 'SYNTHESIS_COMPLETE', ideas: synthesized });
  return synthesized;
}
