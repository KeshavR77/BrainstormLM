import type { BrainstormEvent } from '../types';
import { mockGeneratePhased } from './mockEngine';

const BASE = import.meta.env.VITE_API_URL ?? '';
const USE_MOCK = import.meta.env.VITE_USE_MOCK_DATA === 'true';

export async function generateIdeas(
  prompt: string,
  userPrompt?: string,
  nIdeas = 4,
): Promise<{ summary: string; content: string; trajectory_id: string }[]> {
  const res = await fetch(`${BASE}/brainstorm`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      prompt,
      context: userPrompt ?? null,
      n_ideas: nIdeas,
      use_memory: true,
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
): Promise<{ summary: string; content: string; trajectory_id: string }[]> {
  if (USE_MOCK) {
    return mockGeneratePhased(prompt, context, dispatch);
  }
  return liveGeneratePhased(prompt, context, dispatch);
}

async function liveGeneratePhased(
  prompt: string,
  context: string | undefined,
  dispatch: (event: BrainstormEvent) => void,
): Promise<{ summary: string; content: string; trajectory_id: string }[]> {
  try {
    const res = await fetch(`${BASE}/brainstorm/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt, context: context ?? null, n_ideas: 4, use_memory: true }),
    });

    if (!res.ok) throw new Error(`Stream API error ${res.status}`);

    const reader = res.body!.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    const finalIdeas: { summary: string; content: string; trajectory_id: string }[] = [];

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
              idea: { summary: parsed.summary, content: parsed.content },
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
    const ideas = await generateIdeas(prompt, context);
    dispatch({ type: 'SYNTHESIS_COMPLETE', ideas });
    return ideas;
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
