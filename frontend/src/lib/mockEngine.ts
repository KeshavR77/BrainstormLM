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

export async function mockGeneratePhased(
  prompt: string,
  _context: string | undefined,
  dispatch: (event: BrainstormEvent) => void,
): Promise<{ summary: string; content: string; trajectory_id: string }[]> {
  const packIdx = simpleHash(prompt) % packs.length;
  const pack = packs[packIdx];

  for (const item of pack.diversePool) {
    const delay = 150 + Math.random() * 100;
    await sleep(delay);
    dispatch({
      type: 'RAW_IDEA_ARRIVED',
      idea: {
        summary: item.summary,
        content: item.content,
        temperature: item.temperature,
        creativityScore: item.creativityScore,
      },
    });
  }

  dispatch({ type: 'DIVERSE_POOL_DONE', count: pack.diversePool.length });

  const progressSteps = [0.33, 0.66, 1.0];
  for (const p of progressSteps) {
    await sleep(300);
    dispatch({ type: 'SYNTHESIS_PROGRESS', progress: p });
  }

  const synthesized = pack.synthesizedIndices.map((idx) => {
    const item = pack.diversePool[idx];
    return {
      summary: item.summary,
      content: item.content,
      trajectory_id: crypto.randomUUID(),
    };
  });

  dispatch({ type: 'SYNTHESIS_COMPLETE', ideas: synthesized });
  return synthesized;
}
