import React, { useEffect, useMemo, useState } from 'react'

type RunListItem = {
  run_id: string
  dataset_id?: string
  model_spec?: string
  has_results: boolean
  created_ts?: number
}

type RunResults = any

function Card({ title, children }: { title: string, children: React.ReactNode }) {
  return (
    <div className="rounded-lg border border-gray-200 bg-white shadow-sm">
      <div className="border-b border-gray-100 px-4 py-2 font-medium text-gray-800">{title}</div>
      <div className="p-4">{children}</div>
    </div>
  )
}

export default function ReportsPage() {
  const [runs, setRuns] = useState<RunListItem[]>([])
  const [runId, setRunId] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [results, setResults] = useState<RunResults | null>(null)

  const loadRuns = async () => {
    setError(null)
    try {
      const r = await fetch('/runs')
      if (!r.ok) throw new Error(`HTTP ${r.status}`)
      const js = await r.json()
      setRuns(js)
      if (js.length && !runId) setRunId(js[js.length - 1].run_id)
    } catch (e:any) {
      setError(e.message || 'Failed to load runs')
    }
  }

  const loadResults = async (id: string) => {
    setLoading(true); setError(null)
    try {
      const r = await fetch(`/runs/${id}/results`)
      if (!r.ok) throw new Error(`HTTP ${r.status}`)
      const js = await r.json()
      setResults(js)
    } catch (e:any) {
      setError(e.message || 'Failed to load results')
      setResults(null)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { loadRuns() }, [])
  useEffect(() => { if (runId) loadResults(runId) }, [runId])

  const download = (type: 'json' | 'csv' | 'html') => {
    window.open(`/runs/${runId}/artifacts?type=${type}`, '_blank')
  }

  return (
    <div className="grid gap-4">
      <Card title="Select Run">
        <div className="flex flex-wrap gap-2 items-center text-sm">
          <select className="border rounded px-2 py-1 min-w-[240px]" value={runId} onChange={e => setRunId(e.target.value)}>
            <option value="" disabled>Select a run</option>
            {runs.map(r => (
              <option key={r.run_id} value={r.run_id}>{r.run_id} — {r.dataset_id || '?'} — {r.model_spec || '?'}</option>
            ))}
          </select>
          <button className="px-3 py-1.5 rounded border hover:bg-gray-50" onClick={loadRuns}>Refresh</button>
          <div className="grow" />
          <button disabled={!runId} onClick={() => download('json')} className="px-3 py-1.5 rounded border hover:bg-gray-50 disabled:opacity-50">Download JSON</button>
          <button disabled={!runId} onClick={() => download('csv')} className="px-3 py-1.5 rounded border hover:bg-gray-50 disabled:opacity-50">Download CSV</button>
          <button disabled={!runId} onClick={() => download('html')} className="px-3 py-1.5 rounded border hover:bg-gray-50 disabled:opacity-50">Open Report</button>
        </div>
      </Card>

      <Card title="Report Summary">
        {loading ? (
          <div className="text-sm text-gray-600">Loading…</div>
        ) : error ? (
          <div className="text-sm text-danger">{error}</div>
        ) : results ? (
          <div className="text-sm space-y-3">
            <div className="flex flex-wrap gap-4">
              <div><span className="text-gray-500">Run:</span> <span className="font-mono">{results.run_id}</span></div>
              <div><span className="text-gray-500">Dataset:</span> {results.dataset_id}</div>
              <div><span className="text-gray-500">Model:</span> {results.model_spec}</div>
              <div><span className="text-gray-500">Conversations:</span> {results.conversations?.length ?? 0}</div>
            </div>
            <div className="overflow-x-auto">
              <table className="min-w-full text-xs">
                <thead>
                  <tr className="text-left text-gray-600">
                    <th className="py-2 pr-4">Conversation</th>
                    <th className="py-2 pr-4">Pass</th>
                    <th className="py-2 pr-4">Weighted rate</th>
                    <th className="py-2 pr-4">Turns</th>
                  </tr>
                </thead>
                <tbody>
                  {(results.conversations || []).map((c:any) => (
                    <tr key={c.conversation_id} className="border-t">
                      <td className="py-2 pr-4 font-mono">{c.conversation_id}</td>
                      <td className="py-2 pr-4">{String(c.summary?.conversation_pass)}</td>
                      <td className="py-2 pr-4">{(c.summary?.weighted_pass_rate ?? 0).toFixed(2)}</td>
                      <td className="py-2 pr-4">{(c.turns || []).length}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        ) : (
          <div className="text-sm text-gray-700">Select a run to view the report.</div>
        )}
      </Card>
    </div>
  )
}
