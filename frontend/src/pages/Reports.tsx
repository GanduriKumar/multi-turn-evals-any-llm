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
  const [fbRating, setFbRating] = useState<number>(5)
  const [fbNotes, setFbNotes] = useState<string>('')
  const [fbOverrideConv, setFbOverrideConv] = useState<null | boolean>(null)
  const [fbTurnId, setFbTurnId] = useState<string>('')
  const [fbTurnPass, setFbTurnPass] = useState<boolean | null>(null)
  const [fbMsg, setFbMsg] = useState<string | null>(null)
  const [fbErr, setFbErr] = useState<string | null>(null)

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

  const submitFeedback = async () => {
    if (!runId) return
    setFbMsg(null); setFbErr(null)
    try {
      const body: any = { rating: fbRating, notes: fbNotes }
      if (fbOverrideConv !== null) body.override_conversation_pass = fbOverrideConv
      if (fbTurnId && fbTurnPass !== null) body.override_turn = { conversation_id: fbTurnId.split('#')[0], turn_index: Number(fbTurnId.split('#')[1] || 0), pass: fbTurnPass }
      const r = await fetch(`/runs/${runId}/feedback`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) })
      const js = await r.json()
      if (!r.ok) throw new Error(js?.detail || 'Failed to submit feedback')
      setFbMsg('Feedback submitted')
      setFbNotes('')
      setFbOverrideConv(null)
      setFbTurnId(''); setFbTurnPass(null)
    } catch (e:any) {
      setFbErr(e.message || 'Failed to submit feedback')
    }
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
            <div className="mt-4 border-t pt-3">
              <div className="font-medium mb-2">Human Feedback</div>
              <div className="grid gap-3 sm:grid-cols-2">
                <label className="flex items-center gap-2">
                  <span className="w-28">Rating</span>
                  <input type="number" min={1} max={5} className="border rounded px-2 py-1 w-24" value={fbRating} onChange={e => setFbRating(Number(e.target.value))} />
                </label>
                <label className="flex items-center gap-2">
                  <span className="w-28">Conv override</span>
                  <select className="border rounded px-2 py-1 w-40" value={fbOverrideConv === null ? '' : (fbOverrideConv ? 'true' : 'false')} onChange={e => setFbOverrideConv(e.target.value === '' ? null : e.target.value === 'true')}>
                    <option value="">no override</option>
                    <option value="true">force pass</option>
                    <option value="false">force fail</option>
                  </select>
                </label>
                <label className="flex items-center gap-2">
                  <span className="w-28">Turn override</span>
                  <select className="border rounded px-2 py-1 grow" value={fbTurnId} onChange={e => setFbTurnId(e.target.value)}>
                    <option value="">none</option>
                    {(results.conversations || []).flatMap((c:any) => (c.turns||[]).map((t:any) => ({ key: `${c.conversation_id}#${t.turn_index}`, label: `${c.conversation_id} turn ${t.turn_index}` }))).map(item => (
                      <option key={item.key} value={item.key}>{item.label}</option>
                    ))}
                  </select>
                </label>
                <label className="flex items-center gap-2">
                  <span className="w-28">Turn pass?</span>
                  <select className="border rounded px-2 py-1 w-40" value={fbTurnPass === null ? '' : (fbTurnPass ? 'true' : 'false')} onChange={e => setFbTurnPass(e.target.value === '' ? null : e.target.value === 'true')}>
                    <option value="">no override</option>
                    <option value="true">pass</option>
                    <option value="false">fail</option>
                  </select>
                </label>
              </div>
              <label className="block mt-3">
                <span className="sr-only">Notes</span>
                <textarea className="mt-1 w-full h-24 text-xs border rounded p-2" placeholder="Evaluator notes" value={fbNotes} onChange={e => setFbNotes(e.target.value)} />
              </label>
              <div className="mt-2 flex items-center gap-2">
                <button onClick={submitFeedback} className="px-3 py-1.5 rounded bg-primary text-white hover:opacity-90">Submit Feedback</button>
                {fbMsg && <span className="text-gray-700">{fbMsg}</span>}
                {fbErr && <span className="text-danger">{fbErr}</span>}
              </div>
            </div>
          </div>
        ) : (
          <div className="text-sm text-gray-700">Select a run to view the report.</div>
        )}
      </Card>
    </div>
  )
}
