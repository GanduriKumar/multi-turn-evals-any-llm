import { useEffect, useMemo, useState } from 'react';
import { Link, useLocation } from 'react-router-dom';

import type { DatasetInfo } from './DatasetsPage';

async function fetchDatasets(): Promise<DatasetInfo[]> {
  const res = await fetch('/api/v1/datasets/?page=1&page_size=1000');
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Failed to load datasets: ${res.status} ${text}`);
  }
  return res.json();
}

function useQuery() {
  const { search } = useLocation();
  return useMemo(() => new URLSearchParams(search), [search]);
}

export default function RunSetupPage() {
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const q = useQuery();
  const preselect = useMemo(() => (q.get('datasets') || '').split(',').filter(Boolean), [q]);

  const [selectedDatasets, setSelectedDatasets] = useState<Record<string, boolean>>({});

  const [provider, setProvider] = useState('dummy');
  const [model, setModel] = useState('dummy');
  const [modelName, setModelName] = useState('dummy');
  const [metricBundles, setMetricBundles] = useState<string[]>([]);
  const [truncStrategy, setTruncStrategy] = useState<'none' | 'tokens'>('none');
  const [truncTokens, setTruncTokens] = useState<number | ''>('');
  const [maxWorkers, setMaxWorkers] = useState<number>(1);
  const [seed, setSeed] = useState<number | ''>('');
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');

  const [submitting, setSubmitting] = useState(false);
  const [runId, setRunId] = useState<string | null>(null);

  

  useEffect(() => {
    setLoading(true);
    fetchDatasets()
      .then((list) => {
        setDatasets(list);
        // initialize selection with preselect
        setSelectedDatasets((prev) => {
          const next = { ...prev };
          preselect.forEach((id) => (next[id] = true));
          return next;
        });
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [preselect]);

  const payload = useMemo(() => {
    const selectedIds = Object.entries(selectedDatasets).filter(([_, v]) => v).map(([k]) => k);
    const ds = datasets.filter((d) => selectedIds.includes(d.id)).map((d) => ({
      id: d.id,
      conversation: d.conversation,
      golden: d.golden,
    }));
    const trunc = truncStrategy === 'none' ? undefined : { strategy: truncStrategy, max_input_tokens: truncTokens || undefined };
    return {
      version: '1.0.0',
      datasets: ds,
      models: [{ name: modelName || model, provider, model }],
      name: name || undefined,
      description: description || undefined,
      random_seed: seed === '' ? undefined : seed,
      metric_bundles: metricBundles.length ? metricBundles : undefined,
      truncation: trunc,
      concurrency: { max_workers: maxWorkers },
    };
  }, [datasets, selectedDatasets, provider, model, modelName, metricBundles, truncStrategy, truncTokens, maxWorkers, seed, name, description]);

  async function submit() {
    setSubmitting(true);
    try {
      const res = await fetch('/api/v1/runs/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const body = await res.json();
      if (!res.ok) {
        throw new Error(body?.error?.message || JSON.stringify(body));
      }
      setRunId(body.run_id);
    } catch (e: any) {
      alert(`Failed to start run: ${e.message}`);
    } finally {
      setSubmitting(false);
    }
  }

  const selectedCount = Object.values(selectedDatasets).filter(Boolean).length;

  return (
    <div>
      {runId && (
        <div className="mb-4 p-3 rounded bg-green-50 text-green-700 border border-green-200">
          Run started successfully. Run ID: <strong>{runId}</strong>.{' '}
          <Link className="underline" to={`/dashboard/${runId}`}>Open Dashboard</Link>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <section className="lg:col-span-2 space-y-6">
          <div className="border rounded p-4 bg-white">
            <h2 className="font-semibold mb-3">Datasets</h2>
            {loading && <div>Loading…</div>}
            {error && <div className="text-red-600">{error}</div>}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3 max-h-64 overflow-auto pr-2">
              {datasets.map((d) => (
                <label key={d.id} className="flex items-start gap-3 border p-3 rounded hover:bg-gray-50">
                  <input
                    type="checkbox"
                    checked={!!selectedDatasets[d.id]}
                    onChange={() => setSelectedDatasets((p) => ({ ...p, [d.id]: !p[d.id] }))}
                  />
                  <div className="text-sm">
                    <div className="font-medium">{d.name}</div>
                    <div className="text-gray-600">{d.id}</div>
                  </div>
                </label>
              ))}
            </div>
            <div className="text-sm text-gray-600 mt-2">Selected: {selectedCount}</div>
          </div>

          <div className="border rounded p-4 bg-white">
            <h2 className="font-semibold mb-3">Model</h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
              <div>
                <label htmlFor="provider" className="block text-sm font-medium">Provider</label>
                <select id="provider" className="border rounded p-2 w-full" value={provider} onChange={(e) => setProvider(e.target.value)}>
                  <option value="dummy">Dummy</option>
                  <option value="openai">OpenAI</option>
                  <option value="azure_openai">Azure OpenAI</option>
                  <option value="anthropic">Anthropic</option>
                </select>
              </div>
              <div>
                <label htmlFor="model" className="block text-sm font-medium">Model</label>
                <input id="model" className="border rounded p-2 w-full" value={model} onChange={(e) => setModel(e.target.value)} placeholder="e.g. gpt-4o" />
              </div>
              <div>
                <label htmlFor="modelAlias" className="block text-sm font-medium">Model Name (alias)</label>
                <input id="modelAlias" className="border rounded p-2 w-full" value={modelName} onChange={(e) => setModelName(e.target.value)} placeholder="Optional alias used in results" />
              </div>
            </div>
          </div>

          <div className="border rounded p-4 bg-white">
            <h2 className="font-semibold mb-3">Metrics</h2>
            <div className="flex flex-wrap gap-3 text-sm">
              {['core', 'safety', 'reasoning', 'hallucination'].map((m) => (
                <label key={m} className="flex items-center gap-2 border px-3 py-1 rounded">
                  <input
                    type="checkbox"
                    checked={metricBundles.includes(m)}
                    onChange={() =>
                      setMetricBundles((prev) =>
                        prev.includes(m) ? prev.filter((x) => x !== m) : [...prev, m]
                      )
                    }
                  />
                  {m}
                </label>
              ))}
            </div>
          </div>

          <div className="border rounded p-4 bg-white">
            <h2 className="font-semibold mb-3">Truncation & Concurrency</h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
              <div>
                <label htmlFor="trunc" className="block text-sm font-medium">Truncation</label>
                <select id="trunc" className="border rounded p-2 w-full" value={truncStrategy} onChange={(e) => setTruncStrategy(e.target.value as any)}>
                  <option value="none">None</option>
                  <option value="tokens">Max tokens</option>
                </select>
              </div>
              <div>
                <label htmlFor="maxTokens" className="block text-sm font-medium">Max input tokens</label>
                <input id="maxTokens"
                  className="border rounded p-2 w-full"
                  type="number"
                  value={truncTokens}
                  onChange={(e) => setTruncTokens(e.target.value === '' ? '' : Number(e.target.value))}
                  disabled={truncStrategy === 'none'}
                />
              </div>
              <div>
                <label htmlFor="workers" className="block text-sm font-medium">Max workers</label>
                <input id="workers"
                  className="border rounded p-2 w-full"
                  type="number"
                  min={1}
                  value={maxWorkers}
                  onChange={(e) => setMaxWorkers(Number(e.target.value))}
                />
              </div>
            </div>
          </div>
        </section>

        <aside className="space-y-4">
          <div className="border rounded p-4 bg-white">
            <h2 className="font-semibold mb-3">Summary</h2>
            <div className="text-sm space-y-1">
              <div><strong>Datasets:</strong> {Object.values(selectedDatasets).filter(Boolean).length}</div>
              <div><strong>Provider:</strong> {provider}</div>
              <div><strong>Model:</strong> {model}</div>
              <div><strong>Bundles:</strong> {metricBundles.join(', ') || '—'}</div>
              <div><strong>Truncation:</strong> {truncStrategy}{truncStrategy !== 'none' ? ` (${truncTokens || '—'})` : ''}</div>
              <div><strong>Max workers:</strong> {maxWorkers}</div>
            </div>
            <button
              className="mt-4 w-full bg-blue-600 text-white px-4 py-2 rounded disabled:opacity-50"
              disabled={submitting || Object.values(selectedDatasets).every((v) => !v)}
              onClick={submit}
            >
              {submitting ? 'Starting…' : 'Start Run'}
            </button>
          </div>

          <div className="border rounded p-4 bg-white">
            <h2 className="font-semibold mb-3">Run metadata</h2>
            <div className="space-y-2">
              <div>
                <label htmlFor="runName" className="block text-sm font-medium">Name</label>
                <input id="runName" className="border rounded p-2 w-full" value={name} onChange={(e) => setName(e.target.value)} />
              </div>
              <div>
                <label htmlFor="runDesc" className="block text-sm font-medium">Description</label>
                <textarea id="runDesc" className="border rounded p-2 w-full" value={description} onChange={(e) => setDescription(e.target.value)} />
              </div>
              <div>
                <label htmlFor="runSeed" className="block text-sm font-medium">Random seed</label>
                <input id="runSeed" className="border rounded p-2 w-full" type="number" value={seed} onChange={(e) => setSeed(e.target.value === '' ? '' : Number(e.target.value))} />
              </div>
            </div>
          </div>
        </aside>
      </div>
    </div>
  );
}
