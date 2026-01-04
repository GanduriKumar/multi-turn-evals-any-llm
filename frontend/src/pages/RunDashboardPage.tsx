import { useEffect, useMemo, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import { useArtifactDownloader } from '../hooks/useArtifactDownloader';

type ConversationProgress = {
  dataset_id: string;
  conversation_id: string;
  model: string;
  status: string;
  progress: number; // 0..1
};

type RunProgress = {
  run_id: string;
  overall_status: string;
  conversations: ConversationProgress[];
  events: Record<string, any>[];
};

export default function RunDashboardPage() {
  const { runId = '' } = useParams();
  const [data, setData] = useState<RunProgress | null>(null);
  const [error, setError] = useState<string | null>(null);
  const { downloading, progress, error: dlError, filename, download, cancel } = useArtifactDownloader(runId);

  // simple polling
  useEffect(() => {
    let active = true;
    let timer: ReturnType<typeof setTimeout> | null = null;
    const ac = new AbortController();
    async function poll() {
      try {
        const res = await fetch(`/api/v1/runs/${runId}/progress`, { signal: ac.signal });
        if (!res.ok) throw new Error(`${res.status}`);
        const body = (await res.json()) as RunProgress;
        if (active) setData(body);
      } catch (e: any) {
        if (active) setError(e.message);
      } finally {
        if (active) timer = setTimeout(poll, 2000);
      }
    }
    poll();
    return () => {
      active = false;
      try { ac.abort(); } catch {}
      if (timer) clearTimeout(timer);
    };
  }, [runId]);

  const percent = useMemo(() => {
    if (!data) return 0;
    if (!data.conversations?.length) return 0;
    const vals = data.conversations.map((c) => Math.max(0, Math.min(1, c.progress || 0)));
    return Math.round((vals.reduce((a, b) => a + b, 0) / vals.length) * 100);
  }, [data]);

  async function cancelRun() {
    // We need job_id to cancel; try to derive from latest events if present
    const jobEvent = data?.events?.find((e) => e?.event === 'job_started' && e?.job_id);
    const jobId = jobEvent?.job_id;
    if (!jobId) {
      alert('Job ID not found yet. Try again shortly.');
      return;
    }
    const res = await fetch(`/api/v1/jobs/${jobId}/cancel`, { method: 'POST' });
    if (!res.ok) {
      const t = await res.text();
      alert(`Cancel failed: ${res.status} ${t}`);
    }
  }

  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-semibold">Run Dashboard</h2>
        <div className="text-sm text-gray-600">Run ID: <code>{runId}</code></div>
      </div>

      {error && <div className="text-red-600 mb-3">{error}</div>}

      <div className="mb-4 p-4 bg-white border rounded">
        <div className="flex items-center justify-between">
          <div>
            <div className="text-sm text-gray-600">Overall Status</div>
            <div className="text-lg font-medium capitalize">{data?.overall_status || 'unknown'}</div>
          </div>
          <div className="flex items-center gap-3">
            <button className="px-3 py-2 rounded bg-red-600 text-white disabled:opacity-50" disabled={!data || data.overall_status === 'completed'} onClick={cancelRun}>
              Cancel Run
            </button>
            <Link className="px-3 py-2 rounded bg-blue-600 text-white" to={`/viewer`}>Open Viewer</Link>
          </div>
        </div>
        <div className="mt-3">
          <div className="h-3 bg-gray-200 rounded">
            <div className="h-3 bg-green-500 rounded" style={{ width: `${percent}%` }} />
          </div>
          <div className="text-xs text-gray-600 mt-1">{percent}% complete</div>
        </div>
      </div>

      <div className="mb-4 p-4 bg-white border rounded">
        <h3 className="font-medium mb-2">Artifacts</h3>
        <div className="flex flex-wrap gap-2 text-sm items-center">
          {['results','summary','csv','html','markdown','raw','normalized','turn_scores','progress'].map((a) => (
            <button
              key={a}
              className="px-3 py-2 rounded border hover:bg-gray-50 disabled:opacity-50"
              onClick={() => download([a])}
              disabled={downloading}
              data-testid={`dl-${a}`}
            >
              Download {a}
            </button>
          ))}
          <button
            className="px-3 py-2 rounded border hover:bg-gray-50 disabled:opacity-50"
            onClick={() => download(['results','csv'])}
            disabled={downloading}
            data-testid="dl-zip"
          >
            Download results+csv (zip)
          </button>
          {downloading && (
            <div className="ml-2 text-xs text-gray-600" aria-label="download-progress">
              {progress !== null ? `Downloading… ${progress}%` : 'Downloading…'}
              <button className="ml-2 underline" onClick={cancel}>Cancel</button>
            </div>
          )}
          {filename && !downloading && (
            <div className="ml-2 text-xs text-green-700">Saved {filename}</div>
          )}
          {dlError && (
            <div className="ml-2 text-xs text-red-600" role="alert">{dlError}</div>
          )}
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {(data?.conversations || []).map((c) => (
          <div key={`${c.dataset_id}|${c.conversation_id}|${c.model}`} className="p-3 bg-white border rounded">
            <div className="text-sm text-gray-600">{c.dataset_id} · {c.model}</div>
            <div className="text-lg font-medium">{c.conversation_id}</div>
            <div className="mt-2 h-2 bg-gray-200 rounded">
              <div className="h-2 bg-blue-500 rounded" style={{ width: `${Math.round((c.progress || 0) * 100)}%` }} />
            </div>
            <div className="text-xs text-gray-600 mt-1">{c.status} · {Math.round((c.progress || 0) * 100)}%</div>
          </div>
        ))}
      </div>
    </div>
  );
}
