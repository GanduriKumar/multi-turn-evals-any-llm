import { useEffect, useMemo, useRef, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import { useArtifactDownloader } from '../hooks/useArtifactDownloader';

// Render a simple bar chart using canvas for performance and no extra deps
function BarChart({
  data,
  width = 800,
  height = 240,
  threshold,
}: {
  data: Array<{ label: string; value: number }>;
  width?: number;
  height?: number;
  threshold?: number | null | undefined;
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    canvas.style.width = width + 'px';
    canvas.style.height = height + 'px';
    const ctx = canvas.getContext('2d');
    if (!ctx) {
      // In test/jsdom environments, canvas 2D context may be unavailable
      return;
    }
    ctx.scale(dpr, dpr);

    // background
    (ctx as CanvasRenderingContext2D).fillStyle = '#fff';
    (ctx as CanvasRenderingContext2D).fillRect(0, 0, width, height);

    const padding = 32;
    const chartW = width - padding * 2;
    const chartH = height - padding * 2;

    // axes
    (ctx as CanvasRenderingContext2D).strokeStyle = '#ddd';
    (ctx as CanvasRenderingContext2D).beginPath();
    (ctx as CanvasRenderingContext2D).moveTo(padding, padding);
    (ctx as CanvasRenderingContext2D).lineTo(padding, padding + chartH);
    (ctx as CanvasRenderingContext2D).lineTo(padding + chartW, padding + chartH);
    (ctx as CanvasRenderingContext2D).stroke();

    const n = data.length;
    const barGap = 8;
    const barW = Math.max(1, (chartW - barGap * (n - 1)) / Math.max(1, n));
    const maxV = 1; // scores normalized 0..1

    // threshold line
    if (typeof threshold === 'number') {
      const y = padding + chartH - (threshold / maxV) * chartH;
      (ctx as CanvasRenderingContext2D).strokeStyle = '#f59e0b';
      (ctx as CanvasRenderingContext2D).setLineDash([4, 4]);
      (ctx as CanvasRenderingContext2D).beginPath();
      (ctx as CanvasRenderingContext2D).moveTo(padding, y);
      (ctx as CanvasRenderingContext2D).lineTo(padding + chartW, y);
      (ctx as CanvasRenderingContext2D).stroke();
      (ctx as CanvasRenderingContext2D).setLineDash([]);
    }

    // bars
    data.forEach((d, i) => {
      const x = padding + i * (barW + barGap);
      const h = (d.value / maxV) * chartH;
      const y = padding + chartH - h;
      const below = typeof threshold === 'number' && d.value < threshold;
      (ctx as CanvasRenderingContext2D).fillStyle = below ? '#fecaca' : '#93c5fd';
      (ctx as CanvasRenderingContext2D).fillRect(x, y, barW, h);

      // label
      (ctx as CanvasRenderingContext2D).fillStyle = '#333';
      (ctx as CanvasRenderingContext2D).font = '10px sans-serif';
      const txt = d.label;
      (ctx as CanvasRenderingContext2D).save();
      (ctx as CanvasRenderingContext2D).translate(x + barW / 2, padding + chartH + 12);
      (ctx as CanvasRenderingContext2D).rotate(-Math.PI / 4);
      (ctx as CanvasRenderingContext2D).fillText(txt, 0, 0);
      (ctx as CanvasRenderingContext2D).restore();
    });
  }, [data, width, height, threshold]);

  return <canvas ref={canvasRef} role="img" aria-label="metrics-barchart" />;
}

export default function MetricsBreakdownPage() {
  const { runId = '' } = useParams();
  const [results, setResults] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [metric, setMetric] = useState<string>('correctness');
  const [conversationFilter, setConversationFilter] = useState<string>('');
  const { downloading, progress, error: dlError, filename, download, cancel } = useArtifactDownloader(runId || '');

  useEffect(() => {
    let active = true;
    async function load() {
      try {
        const res = await fetch(`/api/v1/results/${runId}/results`);
        const body = await res.json();
        if (!res.ok) throw new Error(body?.error?.message || JSON.stringify(body));
        if (active) setResults(body);
      } catch (e: any) {
        if (active) setError(e.message);
      }
    }
    load();
    return () => { active = false };
  }, [runId]);

  const thresholds = results?.run?.thresholds || results?.summary?.thresholds || {};
  const turnPass = typeof thresholds.turn_pass === 'number' ? thresholds.turn_pass : undefined;

  // Flatten metrics across conversations/turns
  const rows = useMemo(() => {
    const out: Array<{ key: string; dataset_id: string; conversation_id: string; model_name: string; turn_id: string | number; value: number }>
      = [];
    for (const group of results?.results || []) {
      if (conversationFilter && String(group.conversation_id) !== conversationFilter) continue;
      for (const t of group.turns || []) {
        const v = Number((t.metrics || {})[metric] ?? NaN);
        if (!Number.isFinite(v)) continue;
        out.push({
          key: `${group.dataset_id}|${group.conversation_id}|${group.model_name}|${t.turn_id}`,
          dataset_id: group.dataset_id,
          conversation_id: group.conversation_id,
          model_name: group.model_name,
          turn_id: t.turn_id,
          value: v,
        });
      }
    }
    return out;
  }, [results, metric, conversationFilter]);

  const chartData = rows.map((r) => ({ label: `${r.conversation_id}#${r.turn_id}`, value: r.value }));

  // CSV export of current view
  function exportCSV() {
    const header = 'dataset_id,conversation_id,model_name,turn_id,metric,value\n';
    const lines = rows.map((r) => `${r.dataset_id},${r.conversation_id},${r.model_name},${r.turn_id},${metric},${r.value}`);
    const blob = new Blob([header + lines.join('\n')], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `metrics_${metric}_${runId}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  }

  // Image export of canvas
  function exportImage() {
    const canvas = document.querySelector('canvas[aria-label="metrics-barchart"]') as HTMLCanvasElement | null;
    if (!canvas) return;
    const url = canvas.toDataURL('image/png');
    const a = document.createElement('a');
    a.href = url;
    a.download = `metrics_${metric}_${runId}.png`;
    a.click();
  }

  const conversations = useMemo(() => {
    const s = new Set<string>();
    for (const g of results?.results || []) s.add(String(g.conversation_id));
    return [''].concat(Array.from(s));
  }, [results]);

  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <div>
          <h2 className="text-xl font-semibold">Metrics Breakdown</h2>
          <div className="text-sm text-gray-600">Run: <code>{runId}</code></div>
        </div>
        <div className="flex gap-2">
          <Link className="px-3 py-2 rounded border" to={`/dashboard/${runId}`}>Back to Dashboard</Link>
        </div>
      </div>

      {error && <div className="text-red-600 mb-3">{error}</div>}

      <div className="mb-4 p-4 bg-white border rounded flex flex-wrap gap-4 items-end">
        <div>
          <label htmlFor="metricSel" className="block text-sm font-medium">Metric</label>
          <select id="metricSel" className="border rounded p-2" value={metric} onChange={(e) => setMetric(e.target.value)}>
            {['correctness','structured','constraints','adherence','consistency','hallucination','safety'].map((m) => (
              <option key={m} value={m}>{m}</option>
            ))}
          </select>
        </div>
        <div>
          <label htmlFor="convSel" className="block text-sm font-medium">Conversation</label>
          <select id="convSel" className="border rounded p-2" value={conversationFilter} onChange={(e) => setConversationFilter(e.target.value)}>
            {conversations.map((c) => <option key={c} value={c}>{c || 'All'}</option>)}
          </select>
        </div>
        <div className="text-sm text-gray-600">Threshold: <strong>{turnPass ?? '—'}</strong></div>
        <div className="ml-auto flex gap-2">
          <button className="px-3 py-2 rounded border" onClick={exportCSV}>Export CSV</button>
          <button className="px-3 py-2 rounded border" onClick={exportImage}>Export PNG</button>
          <button className="px-3 py-2 rounded border disabled:opacity-50" disabled={downloading} onClick={() => download(['csv'])} data-testid="dl-csv">Download CSV</button>
          <button className="px-3 py-2 rounded border disabled:opacity-50" disabled={downloading} onClick={() => download(['html'])} data-testid="dl-html">Download HTML</button>
          <button className="px-3 py-2 rounded border disabled:opacity-50" disabled={downloading} onClick={() => download(['markdown'])} data-testid="dl-md">Download Markdown</button>
        </div>
        {downloading && (
          <div className="w-full text-xs text-gray-600" aria-label="download-progress">
            {progress !== null ? `Downloading… ${progress}%` : 'Downloading…'}
            <button className="ml-2 underline" onClick={cancel}>Cancel</button>
          </div>
        )}
        {filename && !downloading && (
          <div className="w-full text-xs text-green-700">Saved {filename}</div>
        )}
        {dlError && (
          <div className="w-full text-xs text-red-600" role="alert">{dlError}</div>
        )}
      </div>

      <div className="mb-4 p-4 bg-white border rounded overflow-auto">
        <BarChart data={chartData} threshold={turnPass} />
      </div>

      <div className="p-4 bg-white border rounded overflow-auto">
        <table className="min-w-full text-sm">
          <thead>
            <tr className="text-left border-b">
              <th className="p-2">Dataset</th>
              <th className="p-2">Conversation</th>
              <th className="p-2">Model</th>
              <th className="p-2">Turn</th>
              <th className="p-2">{metric}</th>
              <th className="p-2">Below Threshold</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.key} className="border-b">
                <td className="p-2">{r.dataset_id}</td>
                <td className="p-2">{r.conversation_id}</td>
                <td className="p-2">{r.model_name}</td>
                <td className="p-2">{String(r.turn_id)}</td>
                <td className="p-2">{r.value.toFixed(4)}</td>
                <td className={`p-2 ${turnPass !== undefined && r.value < turnPass ? 'text-red-600 font-semibold' : 'text-gray-500'}`}>
                  {turnPass !== undefined && r.value < turnPass ? 'Yes' : 'No'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
