import { useEffect, useMemo, useState } from 'react';
import { Link } from 'react-router-dom';

export type DatasetInfo = {
  id: string;
  name: string;
  conversation: string;
  golden: string;
  conversation_version?: string | null;
  golden_version?: string | null;
  tags: string[];
  difficulty?: string | null;
};

async function fetchDatasets(): Promise<DatasetInfo[]> {
  const res = await fetch('/api/v1/datasets/?page=1&page_size=1000');
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Failed to load datasets: ${res.status} ${text}`);
  }
  return res.json();
}

export default function DatasetsPage() {
  const [items, setItems] = useState<DatasetInfo[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [domain, setDomain] = useState('');
  const [difficulty, setDifficulty] = useState('');
  const [selected, setSelected] = useState<Record<string, boolean>>({});

  useEffect(() => {
    setLoading(true);
    fetchDatasets()
      .then(setItems)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  const domains = useMemo(() => {
    const s = new Set<string>();
    items.forEach((d) => (d.tags || []).forEach((t) => s.add(t)));
    return [''].concat(Array.from(s).sort());
  }, [items]);
  const difficulties = useMemo(() => {
    const s = new Set<string>();
    items.forEach((d) => d.difficulty && s.add(d.difficulty));
    return [''].concat(Array.from(s).sort());
  }, [items]);

  const filtered = useMemo(() => {
    return items.filter((d) => {
      const domainOk = !domain || (d.tags || []).includes(domain);
      const diffOk = !difficulty || d.difficulty === difficulty;
      return domainOk && diffOk;
    });
  }, [items, domain, difficulty]);

  const toggle = (id: string) => setSelected((p) => ({ ...p, [id]: !p[id] }));
  const selectedIds = Object.entries(selected).filter(([_, v]) => v).map(([k]) => k);

  return (
    <div>
      <div className="mb-4 flex gap-4 items-end">
        <div>
          <label htmlFor="domain" className="block text-sm font-medium">Domain</label>
          <select id="domain" className="border rounded p-2" value={domain} onChange={(e) => setDomain(e.target.value)}>
            {domains.map((d) => (
              <option key={d} value={d}>{d || 'All'}</option>
            ))}
          </select>
        </div>
        <div>
          <label htmlFor="difficulty" className="block text-sm font-medium">Difficulty</label>
          <select id="difficulty" className="border rounded p-2" value={difficulty} onChange={(e) => setDifficulty(e.target.value)}>
            {difficulties.map((d) => (
              <option key={d} value={d}>{d || 'All'}</option>
            ))}
          </select>
        </div>
        <div className="ml-auto">
          <Link
            to={{ pathname: '/run-setup', search: selectedIds.length ? `?datasets=${selectedIds.join(',')}` : '' }}
            className="bg-blue-600 text-white px-4 py-2 rounded"
          >
            Use Selected ({selectedIds.length})
          </Link>
        </div>
      </div>

      {loading && <div>Loading…</div>}
      {error && <div className="text-red-600">{error}</div>}

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {filtered.map((d) => (
          <div key={d.id} className="border rounded p-4 bg-white">
            <div className="flex items-start gap-3">
              <input type="checkbox" checked={!!selected[d.id]} onChange={() => toggle(d.id)} className="mt-1"/>
              <div className="flex-1">
                <div className="font-semibold">{d.name}</div>
                <div className="text-xs text-gray-600">ID: {d.id}</div>
                <div className="mt-2 flex flex-wrap gap-2">
                  {(d.tags || []).map((t) => (
                    <span key={t} className="text-xs bg-gray-100 px-2 py-1 rounded">{t}</span>
                  ))}
                  {d.difficulty && (
                    <span className="text-xs bg-purple-100 text-purple-700 px-2 py-1 rounded">{d.difficulty}</span>
                  )}
                </div>
              </div>
            </div>
            <div className="mt-3 text-xs text-gray-500">
              conv v{d.conversation_version || '-'} · golden v{d.golden_version || '-'}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
