import React, { useEffect, useMemo, useRef, useState } from 'react'
import Card from '../components/Card'
import Button from '../components/Button'
import Badge from '../components/Badge'
import { Input, Select, Checkbox } from '../components/Form'

type DatasetItem = {
  dataset_id: string
  version?: string
  valid?: boolean
}

type VersionInfo = {
  version: string
  gemini_enabled: boolean
  ollama_host: string | null
  semantic_threshold: number
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
  const [modelSpec, setModelSpec] = useState('ollama:llama3.2:latest')
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

  useEffect(() => {
    const load = async () => {
      setLoading(true); setError(null)
      try {
        const [dR, vR] = await Promise.all([
          fetch('/datasets'),
          fetch('/version')
        ])
        if (!dR.ok) throw new Error(`Datasets HTTP ${dR.status}`)
        if (!vR.ok) throw new Error(`Version HTTP ${vR.status}`)
        const d = await dR.json()
        const v = await vR.json() as VersionInfo
        setDatasets(d)
        setVer(v)
        setSemanticThreshold(Number(v.semantic_threshold) || 0.8)
        // default dataset
        if (d && d.length) setDatasetId(d[0].dataset_id)
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
    const arr = [{ id: 'ollama:llama3.2:latest', label: 'Ollama — llama3.2:latest' }]
    if (ver?.gemini_enabled) arr.push({ id: 'gemini:gemini-2.5', label: 'Gemini — gemini-2.5' })
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
      // Begin polling
      pollRef.current = window.setInterval(async () => {
        const pr = await fetch(`/runs/${body.job_id}/status`)
        const ps = await pr.json()
        setStatus(ps)
        if (ps.state === 'succeeded' || ps.state === 'failed' || ps.state === 'cancelled') {
          if (pollRef.current) { window.clearInterval(pollRef.current); pollRef.current = null }
        }
      }, 1500)
    } catch (e:any) {
      setError(e.message || 'Failed to start run')
    } finally {
      setStarting(false)
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

            <Button onClick={startRun} disabled={starting || !datasetId}>
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
              <>
                <div className="h-2 w-full bg-primary/10 rounded">
                  <div className="h-2 rounded bg-primary" style={{ width: `${Math.round(status.progress_pct * 100)}%` }} />
                </div>
                <div className="text-xs text-gray-600">{status.completed_conversations} / {status.total_conversations} ({Math.round(status.progress_pct * 100)}%)</div>
                {status.error && <div className="text-danger">{status.error}</div>}
              </>
            )}
          </div>
        </Card>
      )}
    </div>
  )
}
