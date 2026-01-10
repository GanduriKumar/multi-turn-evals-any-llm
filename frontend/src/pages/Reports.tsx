import React, { useEffect, useMemo, useState } from 'react'
import Card from '../components/Card'
import Button from '../components/Button'
import Badge from '../components/Badge'
import { Select, Input, Textarea } from '../components/Form'
import { useVertical } from '../context/VerticalContext'

type RunListItem = {
  run_id: string
  dataset_id?: string
  model_spec?: string
  has_results: boolean
  created_ts?: number
}

type RunResults = any

 

export default function ReportsPage() {
  const { vertical } = useVertical()
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
      const r = await fetch(`/runs?vertical=${encodeURIComponent(vertical)}`)
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
      const r = await fetch(`/runs/${id}/results?vertical=${encodeURIComponent(vertical)}`)
      if (!r.ok) {
        if (r.status === 404) {
          // Friendly message when a run has no generated report yet
          setResults(null)
          setError('No report found for this run.')
          return
        }
        throw new Error(`HTTP ${r.status}`)
      }
      const js = await r.json()
      setResults(js)
    } catch (e:any) {
      setError(e.message || 'Failed to load results')
      setResults(null)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { loadRuns() }, [vertical])
  useEffect(() => { if (runId) loadResults(runId) }, [runId, vertical])

  const download = (type: 'json' | 'csv' | 'html') => {
    window.open(`/runs/${runId}/artifacts?type=${type}&vertical=${encodeURIComponent(vertical)}`, '_blank')
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
          <Select className="min-w-[240px]" value={runId} onChange={e => setRunId(e.target.value)}>
            <option value="" disabled>Select a run</option>
            {runs.map(r => (
              <option key={r.run_id} value={r.run_id}>{r.run_id} — {r.dataset_id || '?'} — {r.model_spec || '?'}</option>
            ))}
          </Select>
          <Button variant="secondary" onClick={loadRuns}>Refresh</Button>
          <div className="grow" />
          <Button variant="secondary" disabled={!runId} onClick={() => download('json')}>Download JSON</Button>
          <Button variant="secondary" disabled={!runId} onClick={() => download('csv')}>Download CSV</Button>
          <Button variant="warning" disabled={!runId} onClick={() => download('html')}>Open Report</Button>
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
              <div><span className="text-gray-500">Model:</span> <span className="text-success font-medium">{results.model_spec}</span></div>
              <div><span className="text-gray-500">Conversations:</span> {results.conversations?.length ?? 0}</div>
            </div>
            {results.domain_description && (
              <div className="text-xs text-gray-600 max-w-3xl">{results.domain_description}</div>
            )}
            <div className="overflow-x-auto">
              <table className="min-w-full text-xs">
                <thead>
                  <tr className="text-left text-gray-600">
                    <th className="py-2 pr-4">Conversation</th>
                    <th className="py-2 pr-4">Pass</th>
                    <th className="py-2 pr-4">Weighted rate</th>
                    <th className="py-2 pr-4">Turns</th>
                    <th className="py-2 pr-4">Failed turns</th>
                    <th className="py-2 pr-4">Failed metrics</th>
                  </tr>
                </thead>
                <tbody>
                  {(results.conversations || []).map((c:any) => {
                    const title = c.conversation_title || c.conversation_slug || c.conversation_id
                    const slug = c.conversation_slug || c.conversation_id
                    const domain = c.domain
                    const behavior = c.behavior
                    const scenario = c.scenario
                    const s = c.summary || {}
                    const turnsTotal = typeof s.total_user_turns === 'number' ? s.total_user_turns : (c.turns || []).length
                    const failedTurns = s.failed_turns_count ?? 0
                    const failedMetrics = Array.isArray(s.failed_metrics) ? s.failed_metrics.join(', ') : ''
                    return (
                      <tr key={c.conversation_id} className="border-t align-top">
                        <td className="py-2 pr-4">
                          <div className="font-medium">{title}</div>
                          <div className="text-[11px] text-gray-500">{slug}</div>
                          <div className="text-[11px] text-gray-600">{[domain, behavior, scenario].filter(Boolean).join(' • ')}</div>
                          {c.conversation_description && (
                            <div className="text-[11px] text-gray-600 truncate max-w-[520px]" title={c.conversation_description}>{c.conversation_description}</div>
                          )}
                        </td>
                        <td className="py-2 pr-4">{s.conversation_pass ? <Badge variant="success">pass</Badge> : <Badge variant="danger">fail</Badge>}</td>
                        <td className="py-2 pr-4">{(s.weighted_pass_rate ?? 0).toFixed(2)}</td>
                        <td className="py-2 pr-4">{turnsTotal}</td>
                        <td className="py-2 pr-4">{failedTurns}</td>
                        <td className="py-2 pr-4 whitespace-pre-wrap break-words max-w-[360px]">{failedMetrics}</td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
            <div className="mt-4 border-t pt-3">
              <div className="font-medium mb-2">Human Feedback</div>
              <div className="grid gap-3 sm:grid-cols-2">
                <label className="flex items-center gap-2">
                  <span className="w-28">Rating</span>
                  <Input type="number" min={1} max={5} className="w-24" value={fbRating} onChange={e => setFbRating(Number(e.target.value))} />
                </label>
                <label className="flex items-center gap-2">
                  <span className="w-28">Conv override</span>
                  <Select className="w-40" value={fbOverrideConv === null ? '' : (fbOverrideConv ? 'true' : 'false')} onChange={e => setFbOverrideConv(e.target.value === '' ? null : e.target.value === 'true')}>
                    <option value="">no override</option>
                    <option value="true">force pass</option>
                    <option value="false">force fail</option>
                  </Select>
                </label>
                <label className="flex items-center gap-2">
                  <span className="w-28">Turn override</span>
                  <Select className="grow" value={fbTurnId} onChange={e => setFbTurnId(e.target.value)}>
                    <option value="">none</option>
                    {(results.conversations || []).flatMap((c:any) => {
                      const title = c.conversation_title || c.conversation_slug || c.conversation_id
                      return (c.turns || []).map((t:any) => ({ key: `${c.conversation_id}#${t.turn_index}`, label: `${title} — turn ${t.turn_index}` }))
                    }).map((item:any) => (
                      <option key={item.key} value={item.key}>{item.label}</option>
                    ))}
                  </Select>
                </label>
                <label className="flex items-center gap-2">
                  <span className="w-28">Turn pass?</span>
                  <Select className="w-40" value={fbTurnPass === null ? '' : (fbTurnPass ? 'true' : 'false')} onChange={e => setFbTurnPass(e.target.value === '' ? null : e.target.value === 'true')}>
                    <option value="">no override</option>
                    <option value="true">pass</option>
                    <option value="false">fail</option>
                  </Select>
                </label>
              </div>
              <label className="block mt-3">
                <span className="sr-only">Notes</span>
                <Textarea className="mt-1 w-full h-24 text-xs" placeholder="Evaluator notes" value={fbNotes} onChange={e => setFbNotes(e.target.value)} />
              </label>
              <div className="mt-2 flex items-center gap-2">
                <Button variant="success" onClick={submitFeedback}>Submit Feedback</Button>
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
