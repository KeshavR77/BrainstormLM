const BASE = import.meta.env.VITE_API_URL ?? '';

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

export async function submitFeedback(
  trajectoryId: string,
  thumbs: 'up' | 'down',
  rating?: number,
  critique?: string,
): Promise<void> {
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
