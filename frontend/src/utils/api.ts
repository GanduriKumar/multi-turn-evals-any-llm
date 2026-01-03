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
  const res = await fetch("/api/feedback", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ run_id, feedback }),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Feedback submission failed: ${res.status} ${text}`);
  }
  return res.json() as Promise<{ run_id: string; stored_path: string; total_records: number }>;
}
