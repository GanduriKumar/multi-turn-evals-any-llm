import React, { useEffect, useMemo, useRef, useState } from 'react'
import Card from '../components/Card'
import Button from '../components/Button'
import CircularProgress from '../components/CircularProgress'
import Badge from '../components/Badge'
import { Input, Select, Checkbox } from '../components/Form'

type DatasetItem = {
  dataset_id: string
  version?: string
  valid?: boolean
}
type RunListItem = {
  run_id: string
  dataset_id?: string
  model_spec?: string
  has_results: boolean
  created_ts?: number
  state?: string
  progress_pct?: number
  completed_conversations?: number
  job_id?: string
  stale?: boolean
}

type VersionInfo = {
  version: string
  gemini_enabled: boolean
  ollama_host: string | null
  semantic_threshold: number
  openai_enabled?: boolean
}

type StartRunResponse = {
  job_id: string
  run_id: string
  state: string
}

type JobStatus = {
  job_id: string
  run_id: string
  state: string
  progress_pct: number
  total_conversations: number
  completed_conversations: number
  error?: string | null
}

export default function RunsPage() {
  const [datasets, setDatasets] = useState<DatasetItem[]>([])
  const [ver, setVer] = useState<VersionInfo | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const [datasetId, setDatasetId] = useState('')
  const [modelSpec, setModelSpec] = useState('openai:gpt-5.1')
  const [semanticThreshold, setSemanticThreshold] = useState(0.8)
  const [metricExact, setMetricExact] = useState(true)
  const [metricSemantic, setMetricSemantic] = useState(true)
  const [metricConsistency, setMetricConsistency] = useState(true)
  const [metricAdherence, setMetricAdherence] = useState(true)
  const [metricHallucination, setMetricHallucination] = useState(true)

  const [starting, setStarting] = useState(false)
  const [startRes, setStartRes] = useState<StartRunResponse | null>(null)
  const [status, setStatus] = useState<JobStatus | null>(null)
  const pollRef = useRef<number | null>(null)
  const [recentRuns, setRecentRuns] = useState<RunListItem[]>([])

  // Helper: sync a single run row in Recent Runs from a JobStatus snapshot
  const syncRecentWithStatus = (s: JobStatus) => {
    if (!s) return
    setRecentRuns(prev => {
      const idx = prev.findIndex(r => r.run_id === s.run_id)
      const patch: Partial<RunListItem> = {
        state: s.state,
        progress_pct: s.progress_pct,
        completed_conversations: s.completed_conversations,
        job_id: s.job_id,
        stale: false,
      }
      if (idx >= 0) {
        const updated = { ...prev[idx], ...patch }
        return [...prev.slice(0, idx), updated, ...prev.slice(idx + 1)]
      }
      // If not present, add a minimal entry to top
      const minimal: RunListItem = {
        run_id: s.run_id,
        dataset_id: undefined,
        model_spec: undefined,
        has_results: false,
        created_ts: Date.now(),
        ...patch,
      }
      return [minimal, ...prev]
    })
  }

  const refreshRuns = async () => {
    try {
      const r = await fetch('/runs')
      if (r.ok) {
        const runs = await r.json()
        setRecentRuns(runs)
      }
    } catch {}
  }

  useEffect(() => {
    const load = async () => {
      setLoading(true); setError(null)
      try {
        const [dR, vR, sR, rR] = await Promise.all([
          fetch('/datasets'),
          fetch('/version'),
          fetch('/settings'),
          fetch('/runs')
        ])
        if (!dR.ok) throw new Error(`Datasets HTTP ${dR.status}`)
        if (!vR.ok) throw new Error(`Version HTTP ${vR.status}`)
        if (!sR.ok) throw new Error(`Settings HTTP ${sR.status}`)
        if (!rR.ok) throw new Error(`Runs HTTP ${rR.status}`)
        const d = await dR.json()
        const v = await vR.json() as VersionInfo & { models?: { ollama?: string; gemini?: string; openai?: string } }
        const s = await sR.json() as any
        const runs = await rR.json() as RunListItem[]
        setDatasets(d)
        setVer(v)
        // choose default model based on available providers and configured defaults
        try {
          const oll = v.models?.ollama || 'llama3.2:latest'
          const gem = v.models?.gemini || 'gemini-2.5'
          const oai = v.models?.openai || 'gpt-5.1'
          const defaultModel = (v.openai_enabled ? `openai:${oai}` : (v.gemini_enabled ? `gemini:${gem}` : `ollama:${oll}`))
          setModelSpec(defaultModel)
        } catch {}
        setRecentRuns(runs)
        setSemanticThreshold(Number(v.semantic_threshold) || 0.8)
        // Seed metric toggles from persisted settings.metrics if available
        const cfg = (s && s.metrics && Array.isArray(s.metrics.metrics)) ? s.metrics.metrics : null
        if (cfg) {
          const byName: Record<string, boolean> = {}
          for (const m of cfg) byName[m.name] = !!m.enabled
          setMetricExact(byName['exact_match'] ?? true)
          setMetricSemantic(byName['semantic_similarity'] ?? true)
          setMetricConsistency(byName['consistency'] ?? true)
          setMetricAdherence(byName['adherence'] ?? true)
          setMetricHallucination(byName['hallucination'] ?? true)
        }
        // default dataset
        if (d && d.length) setDatasetId(d[0].dataset_id)
        // If a running/paused job exists, resume polling it
        const active = runs.find(x => x.state && ['running','paused','cancelling'].includes(String(x.state)) && !x.stale)
        if (active?.job_id) {
          // seed UI from list and start polling
          setStartRes({ job_id: active.job_id, run_id: active.run_id, state: String(active.state || 'running') })
          pollRef.current = window.setInterval(async () => {
            const pr = await fetch(`/runs/${active.job_id}/status`)
            const ps = await pr.json()
            setStatus(ps)
            syncRecentWithStatus(ps)
            if (ps.state === 'succeeded' || ps.state === 'failed' || ps.state === 'cancelled') {
              if (pollRef.current) { window.clearInterval(pollRef.current); pollRef.current = null }
              // One-time refresh to ensure list reflects final state
              refreshRuns()
            }
          }, 1500)
        }
      } catch (e:any) {
        setError(e.message || 'Failed to load')
      } finally {
        setLoading(false)
      }
    }
    load()
    return () => { if (pollRef.current) window.clearInterval(pollRef.current) }
  }, [])

  const availableModels = useMemo(() => {
    const arr: { id: string; label: string }[] = []
    const m = (ver as any)?.models || {}
    const ollamaModel = m.ollama || 'llama3.2:latest'
    const geminiModel = m.gemini || 'gemini-2.5'
    const openaiModel = m.openai || 'gpt-5.1'
    arr.push({ id: `ollama:${ollamaModel}`, label: `Ollama — ${ollamaModel}` })
    if (ver?.gemini_enabled) arr.push({ id: `gemini:${geminiModel}`, label: `Gemini — ${geminiModel}` })
    if (ver?.openai_enabled) arr.push({ id: `openai:${openaiModel}`, label: `OpenAI — ${openaiModel}` })
    return arr
  }, [ver])

  const startRun = async () => {
    setStartRes(null); setStatus(null); setStarting(true); setError(null)
    try {
      const metrics: string[] = []
      if (metricExact) metrics.push('exact_match')
      if (metricSemantic) metrics.push('semantic_similarity')
      if (metricConsistency) metrics.push('consistency')
      if (metricAdherence) metrics.push('adherence')
      if (metricHallucination) metrics.push('hallucination')
      const payload = {
        dataset_id: datasetId,
        model_spec: modelSpec,
        metrics,
        thresholds: { semantic_threshold: semanticThreshold },
      }
      const r = await fetch('/runs', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(payload) })
      const body = await r.json()
      if (!r.ok) throw new Error(body?.detail || 'Failed to start run')
      setStartRes(body)
      // Seed recent runs with this new job if missing
      setRecentRuns(prev => {
        if (prev.some(r => r.run_id === body.run_id)) return prev
        return [
          { run_id: body.run_id, dataset_id: datasetId, model_spec: modelSpec, has_results: false, state: body.state, progress_pct: 0, job_id: body.job_id, stale: false },
          ...prev,
        ]
      })
      // Begin polling
      pollRef.current = window.setInterval(async () => {
        const pr = await fetch(`/runs/${body.job_id}/status`)
        const ps = await pr.json()
        setStatus(ps)
        syncRecentWithStatus(ps)
        if (ps.state === 'succeeded' || ps.state === 'failed' || ps.state === 'cancelled') {
          if (pollRef.current) { window.clearInterval(pollRef.current); pollRef.current = null }
          refreshRuns()
        }
      }, 1500)
    } catch (e:any) {
      setError(e.message || 'Failed to start run')
    } finally {
      setStarting(false)
    }
  }

  const control = async (action: 'pause'|'resume'|'cancel', jobIdOverride?: string) => {
    try {
      const jobId = jobIdOverride || startRes?.job_id || status?.job_id
      if (!jobId) return
      const resp = await fetch(`/runs/${jobId}/control`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ action }) })
      if (!resp.ok) {
        const body = await resp.json().catch(() => ({}))
        throw new Error(body?.detail || 'Control failed')
      }
      const body = await resp.json()
      setStatus(body)
      syncRecentWithStatus(body)
      if (action === 'cancel' || body.state === 'cancelled' || body.state === 'failed' || body.state === 'succeeded') {
        // Stop polling and refresh list to align both views
        if (pollRef.current) { window.clearInterval(pollRef.current); pollRef.current = null }
        refreshRuns()
      }
    } catch (e) {
      // ignore for now; poller will reconcile
    }
  }

  return (
    <div className="grid gap-4">
      <Card title="Configure Run">
        {loading ? (
          <div className="text-sm text-gray-600">Loading…</div>
        ) : error ? (
          <div className="text-sm text-danger">{error}</div>
        ) : (
          <div className="space-y-4 text-sm">
            <div className="grid sm:grid-cols-2 gap-4">
              <label className="flex items-center gap-2">
                <span className="w-28">Dataset</span>
                <Select className="grow" value={datasetId} onChange={e => setDatasetId(e.target.value)}>
                  {datasets.map(d => (
                    <option key={d.dataset_id} value={d.dataset_id}>{d.dataset_id} {d.version ? `(${d.version})` : ''}</option>
                  ))}
                </Select>
              </label>
              <label className="flex items-center gap-2">
                <span className="w-28">Model</span>
                <Select className="grow" value={modelSpec} onChange={e => setModelSpec(e.target.value)}>
                  {availableModels.map(m => (<option key={m.id} value={m.id}>{m.label}</option>))}
                </Select>
              </label>
              <label className="flex items-center gap-2">
                <span className="w-28">Semantic thr.</span>
                <Input type="number" step="0.01" min={0} max={1} className="w-28" value={semanticThreshold} onChange={e => setSemanticThreshold(Number(e.target.value))} />
              </label>
            </div>

            <div className="flex flex-wrap gap-4">
              <label className="inline-flex items-center gap-2"><Checkbox checked={metricExact} onChange={e => setMetricExact((e.target as HTMLInputElement).checked)} /> exact</label>
              <label className="inline-flex items-center gap-2"><Checkbox checked={metricSemantic} onChange={e => setMetricSemantic((e.target as HTMLInputElement).checked)} /> semantic</label>
              <label className="inline-flex items-center gap-2"><Checkbox checked={metricConsistency} onChange={e => setMetricConsistency((e.target as HTMLInputElement).checked)} /> consistency</label>
              <label className="inline-flex items-center gap-2"><Checkbox checked={metricAdherence} onChange={e => setMetricAdherence((e.target as HTMLInputElement).checked)} /> adherence</label>
              <label className="inline-flex items-center gap-2"><Checkbox checked={metricHallucination} onChange={e => setMetricHallucination((e.target as HTMLInputElement).checked)} /> hallucination</label>
            </div>

            <Button variant="primary" onClick={startRun} disabled={starting || !datasetId}>
              {starting ? 'Starting…' : 'Start Run'}
            </Button>
          </div>
        )}
      </Card>

      {startRes && (
        <Card title="Run Status">
          <div className="text-sm space-y-2">
            <div>Job: <span className="font-mono">{startRes.job_id}</span></div>
            <div>Run: <span className="font-mono">{startRes.run_id}</span></div>
            <div className="flex items-center gap-2">
              <span>State:</span>
              <span className="font-medium">
                {(() => {
                  const s = status?.state || startRes.state
                  if (s === 'succeeded') return <Badge variant="success">succeeded</Badge>
                  if (s === 'failed' || s === 'cancelled') return <Badge variant="danger">{s}</Badge>
                  return <Badge variant="warning">{s}</Badge>
                })()}
              </span>
            </div>
            {status && (
              <div className="flex items-center gap-8 flex-wrap">
                {(() => {
                  const rawPct = typeof status.progress_pct === 'number' && isFinite(status.progress_pct) ? status.progress_pct : 0
                  const pct = Math.max(0, Math.min(100, Math.round(rawPct)))
                  return (
                    <>
                      <div className="flex flex-col items-center gap-3">
                        <CircularProgress value={pct} size={224} strokeWidth={12} bezel />
                        {status.state !== 'succeeded' && status.state !== 'failed' && status.state !== 'cancelled' && (
                          <div className="flex gap-2">
                            {status.state !== 'paused' ? (
                              <Button onClick={() => control('pause')}>Pause</Button>
                            ) : (
                              <Button variant="success" onClick={() => control('resume')}>Resume</Button>
                            )}
                            <Button variant="danger" onClick={() => control('cancel')}>Abort</Button>
                          </div>
                        )}
                      </div>
                      <div className="text-sm text-gray-700 min-w-[220px]">
                        <div>{status.completed_conversations} / {status.total_conversations}</div>
                        <div className="text-gray-500">{pct}% complete</div>
                        {status.error && <div className="text-danger">{status.error}</div>}
                      </div>
                    </>
                  )
                })()}
              </div>
            )}
          </div>
        </Card>
      )}

      {/* Recent runs (lets user reselect or view status when returning) */}
      <Card title="Recent Runs">
        <div className="text-xs text-gray-700">
          {recentRuns.length === 0 && <div>No runs yet.</div>}
          {recentRuns.length > 0 && (
            <div className="overflow-x-auto">
              <table className="min-w-full text-left">
                <thead>
                  <tr>
                    <th className="px-2 py-1">Run</th>
                    <th className="px-2 py-1">Dataset</th>
                    <th className="px-2 py-1">Model</th>
                    <th className="px-2 py-1">State</th>
                    <th className="px-2 py-1">Progress</th>
                    <th className="px-2 py-1"></th>
                  </tr>
                </thead>
                <tbody>
                  {recentRuns.map(r => (
                    <tr key={r.run_id} className="border-t">
                      <td className="px-2 py-1 font-mono text-[11px]">{r.run_id}</td>
                      <td className="px-2 py-1">{r.dataset_id}</td>
                      <td className="px-2 py-1">{r.model_spec}</td>
                      <td className="px-2 py-1">{r.state || (r.has_results ? 'succeeded' : '')}{r.stale ? ' (stale)' : ''}</td>
                      <td className="px-2 py-1">{typeof r.progress_pct === 'number' && isFinite(r.progress_pct as number) ? `${Math.round(r.progress_pct as number)}%` : ''}</td>
                      <td className="px-2 py-1">
                        {r.job_id && !r.stale && (r.state === 'running' || r.state === 'paused') && (
                          <div className="flex gap-2">
                            {r.state === 'running' ? (
                              <Button onClick={async () => {
                                await control('pause', r.job_id!)
                                // Show this job in the status card and keep polling
                                setStartRes({ job_id: r.job_id!, run_id: r.run_id, state: 'paused' })
                                if (!pollRef.current) {
                                  pollRef.current = window.setInterval(async () => {
                                    const pr = await fetch(`/runs/${r.job_id}/status`)
                                    const ps = await pr.json()
                                    setStatus(ps)
                                    syncRecentWithStatus(ps)
                                    if (ps.state === 'succeeded' || ps.state === 'failed' || ps.state === 'cancelled') {
                                      if (pollRef.current) { window.clearInterval(pollRef.current); pollRef.current = null }
                                      refreshRuns()
                                    }
                                  }, 1500)
                                }
                              }}>Pause</Button>
                            ) : (
                              <Button variant="success" onClick={async () => {
                                await control('resume', r.job_id!)
                                setStartRes({ job_id: r.job_id!, run_id: r.run_id, state: 'running' })
                                if (pollRef.current) window.clearInterval(pollRef.current)
                                pollRef.current = window.setInterval(async () => {
                                  const pr = await fetch(`/runs/${r.job_id}/status`)
                                  const ps = await pr.json()
                                  setStatus(ps)
                                  syncRecentWithStatus(ps)
                                  if (ps.state === 'succeeded' || ps.state === 'failed' || ps.state === 'cancelled') {
                                    if (pollRef.current) { window.clearInterval(pollRef.current); pollRef.current = null }
                                    refreshRuns()
                                  }
                                }, 1500)
                              }}>Resume</Button>
                            )}
                            <Button variant="danger" onClick={async () => {
                              try {
                                const resp = await fetch(`/runs/${r.job_id}/control`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ action: 'cancel' }) })
                                if (resp.ok) {
                                  const s = await resp.json()
                                  setStatus(s)
                                  syncRecentWithStatus(s)
                                  if (pollRef.current) { window.clearInterval(pollRef.current); pollRef.current = null }
                                  refreshRuns()
                                }
                              } catch {}
                            }}>Abort</Button>
                          </div>
                        )}
                        {r.job_id && !r.stale && r.state === 'cancelling' && (
                          <div className="flex gap-2">
                            <Button variant="danger" onClick={async () => {
                              try {
                                const resp = await fetch(`/runs/${r.job_id}/control`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ action: 'cancel' }) })
                                if (resp.ok) {
                                  const s = await resp.json()
                                  setStatus(s)
                                  syncRecentWithStatus(s)
                                  if (pollRef.current) { window.clearInterval(pollRef.current); pollRef.current = null }
                                  refreshRuns()
                                }
                              } catch {}
                            }}>Abort</Button>
                          </div>
                        )}
                        {r.stale && (["stale","running","paused","cancelling"].includes(String(r.state))) && (
                          <div className="flex gap-2">
                            <Button onClick={async () => {
                              // Mark a stale, previously-active job as cancelled (does not affect current active job)
                              try {
                                const resp = await fetch(`/runs/${r.job_id}/control`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ action: 'cancel' }) })
                                if (resp.ok) {
                                  // Refresh list to reflect persisted change
                                  await refreshRuns()
                                }
                              } catch {}
                            }}>Mark as cancelled</Button>
                          </div>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </Card>
    </div>
  )
}
