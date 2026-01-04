import React, { useState } from 'react';
import { submitFeedback } from '../utils/api';
import type { TurnFeedback } from '../utils/api';

export type FeedbackFormProps = {
  runId: string;
  datasetId: string;
  conversationId: string;
  modelName: string;
  turnId: number | string;
  onSubmitted?: (result: { run_id: string; stored_path: string; total_records: number }) => void;
};

export default function FeedbackForm({ runId, datasetId, conversationId, modelName, turnId, onSubmitted }: FeedbackFormProps) {
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setSubmitting(true);
    setError(null);
    setSuccess(null);
    // Read values from DOM (keep inputs uncontrolled to avoid React/state timing issues in tests)
    const formEl = e.currentTarget as HTMLFormElement;
    const ratingEl = formEl.querySelector('#rating') as HTMLInputElement | null;
    const notesEl = formEl.querySelector('#notes') as HTMLTextAreaElement | null;
    const passEl = formEl.querySelector('#overridePass') as HTMLSelectElement | null;
    const scoreEl = formEl.querySelector('#overrideScore') as HTMLInputElement | null;
    const resolvedRating = ratingEl && ratingEl.value !== '' ? Number(ratingEl.value) : null;
    const resolvedNotes = notesEl ? (notesEl.value || null) : null;
    const resolvedOverridePass = passEl ? (passEl.value === '' ? null : passEl.value === 'true') : null;
    const resolvedOverrideScore = scoreEl && scoreEl.value !== '' ? Number(scoreEl.value) : null;

    const payload: TurnFeedback = {
      dataset_id: datasetId,
      conversation_id: conversationId,
      model_name: modelName,
      turn_id: turnId,
      rating: resolvedRating as any,
      notes: resolvedNotes as any,
      override_pass: resolvedOverridePass as any,
      override_score: resolvedOverrideScore as any,
    };

    try {
      const res = await submitFeedback(runId, [payload]);
      setSuccess(`Saved to ${res.stored_path}`);
      onSubmitted?.(res);
      // Reset optional fields if desired
      // setRating(''); setNotes(''); setOverridePass(''); setOverrideScore('');
    } catch (err: any) {
      setError(err?.message ?? 'Failed to submit');
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="feedback-form" aria-label="feedback-form">
      <h3>Evaluator Feedback</h3>
      <div className="field">
        <label htmlFor="rating">Rating (0-5)</label>
        <input
          id="rating"
          name="rating"
          type="number"
          min={0}
          max={5}
          step={0.5}
          defaultValue={'' as any}
        />
      </div>
      <div className="field">
        <label htmlFor="notes">Notes</label>
        <textarea id="notes" name="notes" defaultValue="" />
      </div>
      <div className="field">
        <label htmlFor="overridePass">Override Pass</label>
        <select
          id="overridePass"
          name="overridePass"
          defaultValue=""
        >
          <option value="">No override</option>
          <option value="true">Pass</option>
          <option value="false">Fail</option>
        </select>
      </div>
      <div className="field">
        <label htmlFor="overrideScore">Override Score (0-1)</label>
        <input
          id="overrideScore"
          name="overrideScore"
          type="number"
          min={0}
          max={1}
          step={0.01}
          defaultValue={'' as any}
        />
      </div>
      <button type="submit" disabled={submitting}>
        {submitting ? 'Submitting...' : 'Submit Feedback'}
      </button>
      {error && <div role="alert">{error}</div>}
      {success && <div role="status">{success}</div>}
    </form>
  );
}
