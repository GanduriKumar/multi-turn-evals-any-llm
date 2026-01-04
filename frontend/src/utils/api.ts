export type TurnFeedback = {
  dataset_id: string;
  conversation_id: string;
  model_name: string;
  turn_id: number | string;
  rating?: number | null;
  notes?: string | null;
  override_pass?: boolean | null;
  override_score?: number | null;
};

export async function submitFeedback(run_id: string, feedback: TurnFeedback[]) {
  // Post to run-scoped endpoint while preserving legacy body shape expected by some tests (includes run_id)
  const res = await fetch(`/api/v1/runs/${encodeURIComponent(run_id)}/feedback`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ run_id, feedback }),
  });
  if (!res.ok) {
    try {
      const body = await res.json();
      const msg = body?.detail?.errors ? body.detail.errors.join("; ") : JSON.stringify(body);
      throw new Error(`Feedback submission failed: ${res.status} ${msg}`);
    } catch {
      const text = await res.text();
      throw new Error(`Feedback submission failed: ${res.status} ${text}`);
    }
  }
  return res.json() as Promise<{ run_id: string; stored_path: string; total_records: number }>;
}

// Run-scoped feedback submission to match backend endpoint
export async function submitRunFeedback(run_id: string, feedback: TurnFeedback[]) {
  const res = await fetch(`/api/v1/runs/${encodeURIComponent(run_id)}/feedback`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ feedback }),
  });
  if (!res.ok) {
    try {
      const body = await res.json();
      const msg = body?.detail?.errors ? body.detail.errors.join("; ") : JSON.stringify(body);
      throw new Error(`Feedback submission failed: ${res.status} ${msg}`);
    } catch {
      const text = await res.text();
      throw new Error(`Feedback submission failed: ${res.status} ${text}`);
    }
  }
  return res.json() as Promise<{ run_id: string; stored_path: string; total_records: number }>;
}

// Compare two runs: returns summary, per_dataset, metrics_by_dataset, per_conversation
export async function compareRuns(baseline: string, current: string) {
  const params = new URLSearchParams({ baseline, current });
  const res = await fetch(`/api/v1/runs/compare?${params.toString()}`);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Compare failed: ${res.status} ${text}`);
  }
  return res.json() as Promise<{
    baseline_run_id: string;
    current_run_id: string;
    summary: { overall: { baseline: number; current: number; delta: number }; counts: any };
    per_dataset: Array<{ dataset_id: string; baseline: number; current: number; delta: number }>;
    metrics_by_dataset: Array<{ dataset_id: string; metric: string; baseline: number; current: number; delta: number }>;
    per_conversation: Array<{ dataset_id: string; conversation_id: string; model_name: string; baseline: number; current: number; delta: number }>;
  }>;
}
