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
  const [rating, setRating] = useState<number | ''>('');
  const [notes, setNotes] = useState('');
  const [overridePass, setOverridePass] = useState<boolean | ''>('');
  const [overrideScore, setOverrideScore] = useState<number | ''>('');
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setSubmitting(true);
    setError(null);
    setSuccess(null);

    const payload: TurnFeedback = {
      dataset_id: datasetId,
      conversation_id: conversationId,
      model_name: modelName,
      turn_id: turnId,
      rating: rating === '' ? null : Number(rating),
      notes: notes || null,
      override_pass: overridePass === '' ? null : Boolean(overridePass),
      override_score: overrideScore === '' ? null : Number(overrideScore),
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
          value={rating}
          onChange={(e) => setRating(e.target.value === '' ? '' : Number(e.target.value))}
        />
      </div>
      <div className="field">
        <label htmlFor="notes">Notes</label>
        <textarea id="notes" name="notes" value={notes} onChange={(e) => setNotes(e.target.value)} />
      </div>
      <div className="field">
        <label htmlFor="overridePass">Override Pass</label>
        <select
          id="overridePass"
          name="overridePass"
          value={overridePass === '' ? '' : overridePass ? 'true' : 'false'}
          onChange={(e) => {
            const v = e.target.value;
            setOverridePass(v === '' ? '' : v === 'true');
          }}
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
          value={overrideScore}
          onChange={(e) => setOverrideScore(e.target.value === '' ? '' : Number(e.target.value))}
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
