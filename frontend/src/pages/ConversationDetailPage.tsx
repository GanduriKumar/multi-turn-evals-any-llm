import React, { useEffect, useMemo, useState } from 'react';
import { Link, useParams, useSearchParams } from 'react-router-dom';
import { submitRunFeedback, type TurnFeedback } from '../utils/api';

// Lightweight diff utility to highlight expected vs actual differences line-by-line
function diffLines(a: string, b: string): Array<{ type: 'equal' | 'add' | 'del'; text: string }> {
  const aLines = (a || '').split(/\r?\n/);
  const bLines = (b || '').split(/\r?\n/);
  const max = Math.max(aLines.length, bLines.length);
  const out: Array<{ type: 'equal' | 'add' | 'del'; text: string }> = [];
  for (let i = 0; i < max; i++) {
    const al = aLines[i] ?? '';
    const bl = bLines[i] ?? '';
    if (al === bl) {
      out.push({ type: 'equal', text: bl });
    } else if (al === '') {
      // only actual has a line: show deletion of actual
      out.push({ type: 'del', text: bl });
    } else if (bl === '') {
      // only expected has a line: show addition of expected
      out.push({ type: 'add', text: al });
    } else {
      // both different: show deletion of actual, then addition of expected
      out.push({ type: 'del', text: bl });
      out.push({ type: 'add', text: al });
    }
  }
  return out;
}

// Virtualized list for performance (simple windowing by index range)
function useWindowed<T>(items: T[], windowSize = 20) {
  const [start, setStart] = useState(0);
  const end = Math.min(items.length, start + windowSize);
  const window = useMemo(() => items.slice(start, end), [items, start, end]);
  return { window, start, end, setStart };
}

export default function ConversationDetailPage() {
  const { runId = '', conversationId = '' } = useParams();
  const [searchParams] = useSearchParams();
  const modelName = searchParams.get('model');

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [payload, setPayload] = useState<any>(null);

  useEffect(() => {
    let active = true;
    async function load() {
      setLoading(true);
      setError(null);
      try {
        const url = modelName
          ? `/api/v1/runs/${runId}/conversations/${conversationId}?model_name=${encodeURIComponent(modelName)}`
          : `/api/v1/runs/${runId}/conversations/${conversationId}`;
        const res = await fetch(url);
        const body = await res.json();
        if (!res.ok) throw new Error(body?.error?.message || JSON.stringify(body));
        if (active) setPayload(body);
      } catch (e: any) {
        if (active) setError(e.message);
      } finally {
        if (active) setLoading(false);
      }
    }
    load();
    return () => {
      active = false;
    };
  }, [runId, conversationId, modelName]);

  // Flatten per-model results for rendering; each item contains the turns array
  const entries = (payload?.results || []) as Array<any>;

  // Virtualization over turns for the first entry with many turns; in multi-model scenarios, this can be adapted per-entry
  const turns = (entries[0]?.turns || []) as Array<any>;
  const { window, start, end, setStart } = useWindowed(turns, 30);

  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <div>
          <h2 className="text-xl font-semibold">Conversation</h2>
          <div className="text-sm text-gray-600">Run: <code>{runId}</code> · Conversation: <code>{conversationId}</code> {modelName ? <>· Model: <code>{modelName}</code></> : null}</div>
        </div>
        <div className="flex gap-2">
          <Link className="px-3 py-2 rounded border" to={`/dashboard/${runId}`}>Back to Dashboard</Link>
          <Link className="px-3 py-2 rounded border" to={`/viewer`}>Viewer</Link>
        </div>
      </div>

      {loading && <div>Loading…</div>}
      {error && <div className="text-red-600">{error}</div>}

      {entries.map((entry, idx) => (
        <div key={idx} className="mb-6 p-4 bg-white border rounded">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <div className="text-sm text-gray-600">Dataset: {entry.dataset_id} · Model: {entry.model_name}</div>
              <div className="text-sm">Aggregate Score: <strong>{Number(entry.aggregate?.score ?? 0).toFixed(4)}</strong> · {entry.aggregate?.passed ? '✅ Passed' : '❌ Failed'}</div>
            </div>
            <div className="text-xs text-gray-600">
              Thresholds: turn {entry.thresholds?.turn_pass ?? '—'} · conv {entry.thresholds?.conversation_pass ?? '—'}
            </div>
          </div>

          {/* Window controls for big transcripts */}
          {idx === 0 && turns.length > 30 && (
            <div className="mt-3 flex items-center gap-2 text-sm">
              <button className="px-2 py-1 border rounded" disabled={start === 0} onClick={() => setStart(Math.max(0, start - 30))}>Prev 30</button>
              <button className="px-2 py-1 border rounded" disabled={end >= turns.length} onClick={() => setStart(Math.min(turns.length - 1, start + 30))}>Next 30</button>
              <div className="text-gray-600">Showing {start + 1}–{end} of {turns.length} turns</div>
            </div>
          )}

          <div className="mt-4 divide-y">
            {(idx === 0 ? window : (entry.turns || [])).map((t: any) => (
              <TurnRow key={String(t.turn_id)} turn={t} datasetId={entry.dataset_id} modelName={entry.model_name} />
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}

function CodeBlock({ children, className = '' }: { children: React.ReactNode; className?: string }) {
  return (
    <pre className={`whitespace-pre-wrap text-sm p-3 rounded border bg-gray-50 ${className}`}>{children}</pre>
  );
}

function DiffView({ expected, actual }: { expected: string; actual: string }) {
  const parts = useMemo(() => diffLines(expected || '', actual || ''), [expected, actual]);
  return (
    <div className="text-sm font-mono">
      {parts.map((p, i) => (
        <div key={i} className={p.type === 'equal' ? '' : p.type === 'add' ? 'bg-green-100' : 'bg-red-100'}>
          {p.type === 'add' ? '+ ' : p.type === 'del' ? '- ' : '  '}{p.text}
        </div>
      ))}
    </div>
  );
}

function Accordion({ title, children, defaultOpen = false }: { title: React.ReactNode; children: React.ReactNode; defaultOpen?: boolean }) {
  const [open, setOpen] = useState(!!defaultOpen);
  return (
    <div className="py-3">
      <button className="w-full text-left flex justify-between items-center" onClick={() => setOpen(!open)}>
        <span className="font-medium">{title}</span>
        <span className="text-xs text-gray-500">{open ? 'Hide' : 'Show'}</span>
      </button>
      {open && <div className="mt-2">{children}</div>}
    </div>
  );
}

function TurnRow({ turn, datasetId, modelName }: { turn: any; datasetId: string; modelName: string }) {
  // Determine expected text from canonical if present; backend provides canonical with text/structured
  const expectedText = String(turn?.canonical?.text || '');
  const actualText = String(turn?.response || '');

  const hasDiff = useMemo(() => expectedText !== '' && expectedText !== actualText, [expectedText, actualText]);

  // Local unsaved feedback key (persist across navigation)
  const { runId, conversationId } = useParams();
  const storageKey = `feedback:${runId}:${conversationId}:${String(turn.turn_id)}`;
  const initial = (() => {
    try {
      const raw = localStorage.getItem(storageKey);
      return raw ? JSON.parse(raw) : {};
    } catch { return {}; }
  })();
  const [rating, setRating] = useState<string>(initial.rating ?? '');
  const [notes, setNotes] = useState<string>(initial.notes ?? '');
  const [overridePass, setOverridePass] = useState<string>(initial.overridePass ?? '');
  const [overrideScore, setOverrideScore] = useState<string>(initial.overrideScore ?? '');
  const [submitting, setSubmitting] = useState(false);
  const [status, setStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Persist unsaved feedback
  useEffect(() => {
    const snapshot = { rating, notes, overridePass, overrideScore };
    try { localStorage.setItem(storageKey, JSON.stringify(snapshot)); } catch {}
  }, [rating, notes, overridePass, overrideScore, storageKey]);

  const requiredFilled = rating !== '' && notes.trim() !== '' && overridePass !== '';

  async function onSubmit() {
    if (!runId) return;
    setSubmitting(true);
    setStatus(null);
    setError(null);
    const payload: TurnFeedback = {
      dataset_id: String(datasetId || ''),
      conversation_id: String(conversationId || ''),
      model_name: String(modelName || ''),
      turn_id: turn.turn_id,
      rating: rating === '' ? null : Number(rating),
      notes: notes || null,
      override_pass: overridePass === '' ? null : overridePass === 'true',
      override_score: overrideScore === '' ? null : Number(overrideScore),
    };
    try {
      await submitRunFeedback(runId, [payload]);
      setStatus('Feedback saved');
      // Clear local cache for this turn after successful save
      try { localStorage.removeItem(storageKey); } catch {}
    } catch (e: any) {
      setError(e?.message || 'Failed to submit');
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="py-3">
      <div className="flex items-start justify-between gap-3">
        <div className="text-sm text-gray-600">Turn {String(turn.turn_id)}</div>
        <div className="text-sm">
          Score: <strong>{Number(turn.weighted_score ?? 0).toFixed(4)}</strong> · {turn.passed ? '✅' : '❌'}
        </div>
      </div>

      <Accordion title="User Prompt">
        <CodeBlock>{turn.prompt}</CodeBlock>
      </Accordion>

      <Accordion title="Assistant Response">
        <CodeBlock className={hasDiff ? 'border-red-300' : ''}>{actualText}</CodeBlock>
      </Accordion>

      <Accordion title="Expected (Canonical)">
        <CodeBlock className={hasDiff ? 'border-green-300' : ''}>{expectedText || '—'}</CodeBlock>
      </Accordion>

      {hasDiff && (
        <Accordion title="Diff (expected → actual)" defaultOpen>
          <DiffView expected={expectedText} actual={actualText} />
        </Accordion>
      )}

      {turn.metrics && (
        <Accordion title="Metrics">
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-2 text-sm">
            {Object.entries(turn.metrics || {}).map(([k, v]) => (
              <div key={k} className="px-2 py-1 rounded border">
                <div className="text-gray-600 text-xs">{k}</div>
                <div className="font-medium">{Number(v as any).toFixed(4)}</div>
              </div>
            ))}
          </div>
        </Accordion>
      )}

      {/* Evaluation Feedback */}
      <Accordion title="Evaluator Feedback" defaultOpen>
        <div className="grid gap-3 md:grid-cols-2">
          <label className="block text-sm">
            <span className="text-gray-700">Rating (0-5)</span>
            <input className="mt-1 w-full border rounded p-2" type="number" min={0} max={5} step={0.5}
              value={rating} onChange={(e) => setRating(e.target.value)} aria-label={`rating-${turn.turn_id}`} />
          </label>
          <label className="block text-sm">
            <span className="text-gray-700">Override Pass/Fail</span>
            <select className="mt-1 w-full border rounded p-2" value={overridePass} onChange={(e) => setOverridePass(e.target.value)} aria-label={`override-${turn.turn_id}`}>
              <option value="">Select…</option>
              <option value="true">Pass</option>
              <option value="false">Fail</option>
            </select>
          </label>
          <label className="block text-sm md:col-span-2">
            <span className="text-gray-700">Notes</span>
            <textarea className="mt-1 w-full border rounded p-2" rows={3} value={notes} onChange={(e) => setNotes(e.target.value)} aria-label={`notes-${turn.turn_id}`} />
          </label>
          <label className="block text-sm md:col-span-2">
            <span className="text-gray-700">Override Score (0-1, optional)</span>
            <input className="mt-1 w-full border rounded p-2" type="number" min={0} max={1} step={0.01}
              value={overrideScore} onChange={(e) => setOverrideScore(e.target.value)} aria-label={`ovscore-${turn.turn_id}`} />
          </label>
        </div>
        <div className="mt-3 flex items-center gap-3">
          <button className="px-3 py-2 rounded border" onClick={onSubmit} disabled={!requiredFilled || submitting} aria-label={`submitfb-${turn.turn_id}`}>
            {submitting ? 'Submitting…' : 'Submit Feedback'}
          </button>
          {status && <span role="status" className="text-green-700">{status}</span>}
          {error && <span role="alert" className="text-red-700">{error}</span>}
          {!requiredFilled && <span className="text-xs text-gray-500">Fill rating, notes, and override to enable submit.</span>}
        </div>
      </Accordion>
    </div>
  );
}
