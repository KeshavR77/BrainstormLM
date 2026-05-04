const RUNPOD_BASE = import.meta.env.VITE_RUNPOD_API_URL ?? '';
const RUNPOD_KEY = import.meta.env.VITE_RUNPOD_API_KEY ?? '';

export interface SteeredRequest {
  prompt: string;
  alpha: number;
  layer: 'mid' | 'late';
  layer_index: number;
  temperature?: number;
  max_tokens?: number;
}

export interface SteeredResponse {
  text: string;
  metrics: {
    vendi_score: number;
    prism_coherence: number;
    novelty: number;
    feasibility: number;
  };
  config: {
    alpha: number;
    layer: string;
    layer_index: number;
    temperature: number;
  };
}

export async function fetchSteeredGeneration(
  req: SteeredRequest,
): Promise<SteeredResponse> {
  const res = await fetch(`${RUNPOD_BASE}/generate`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...(RUNPOD_KEY ? { Authorization: `Bearer ${RUNPOD_KEY}` } : {}),
    },
    body: JSON.stringify({
      prompt: req.prompt,
      alpha: req.alpha,
      layer: req.layer,
      layer_index: req.layer_index,
      temperature: req.temperature ?? 0.9,
      max_tokens: req.max_tokens ?? 512,
    }),
  });

  if (!res.ok) throw new Error(`Steered API error ${res.status}`);
  return res.json();
}
