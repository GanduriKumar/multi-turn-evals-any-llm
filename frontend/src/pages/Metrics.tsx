import React, { useEffect, useState } from 'react'
import Card from '../components/Card'
import Button from '../components/Button'
import { Input, Checkbox } from '../components/Form'

 

type MetricCfg = { name: string, enabled: boolean, weight?: number, threshold?: number }

type MetricsCfg = { metrics: MetricCfg[] }

export default function MetricsPage() {
  const [cfg, setCfg] = useState<MetricsCfg | null>(null)
  const [saving, setSaving] = useState(false)
  const [msg, setMsg] = useState<string | null>(null)
  const [err, setErr] = useState<string | null>(null)

  const load = async () => {
    setErr(null)
    try {
      const r = await fetch('/metrics-config')
      if (!r.ok) throw new Error(`HTTP ${r.status}`)
      setCfg(await r.json())
    } catch (e:any) {
      setErr(e.message || 'Failed to load')
    }
  }

  useEffect(() => { load() }, [])

  const save = async () => {
    if (!cfg) return
    setSaving(true); setErr(null); setMsg(null)
    try {
      const r = await fetch('/metrics-config', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(cfg) })
      const js = await r.json()
      if (!r.ok) throw new Error(js?.detail || 'Save failed')
      setMsg('Saved')
    } catch (e:any) {
      setErr(e.message || 'Save failed')
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="grid gap-4">
      <Card title="Metrics Configuration">
        {cfg ? (
          <div className="space-y-3 text-sm">
            {cfg.metrics.map((m, i) => (
              <div key={m.name} className="grid sm:grid-cols-3 gap-3 items-center">
                <label className="inline-flex items-center gap-2">
                  <Checkbox checked={m.enabled} onChange={e => setCfg(x => { const y = {...x!}; y.metrics = y.metrics.map((mm, idx) => idx===i ? {...mm, enabled: (e.target as HTMLInputElement).checked} : mm); return y })} />
                  <span className="font-medium">{m.name}</span>
                </label>
                <label className="flex items-center gap-2">
                  <span className="w-24">Weight</span>
                  <Input type="number" step="0.1" className="w-28" value={m.weight ?? 1} onChange={e => setCfg(x => { const y = {...x!}; y.metrics = y.metrics.map((mm, idx) => idx===i ? {...mm, weight: Number(e.target.value)} : mm); return y })} />
                </label>
                {'threshold' in m ? (
                  <label className="flex items-center gap-2">
                    <span className="w-24">Threshold</span>
                    <Input type="number" step="0.01" min={0} max={1} className="w-28" value={m.threshold ?? 0} onChange={e => setCfg(x => { const y = {...x!}; y.metrics = y.metrics.map((mm, idx) => idx===i ? {...mm, threshold: Number(e.target.value)} : mm); return y })} />
                  </label>
                ) : <div />}
              </div>
            ))}
            <div className="flex items-center gap-2">
              <Button onClick={save} disabled={saving}>{saving ? 'Saving…' : 'Save'}</Button>
              {msg && <span className="text-gray-700">{msg}</span>}
              {err && <span className="text-danger">{err}</span>}
            </div>
          </div>
        ) : err ? (
          <div className="text-sm text-danger">{err}</div>
        ) : (
          <div className="text-sm text-gray-600">Loading…</div>
        )}
      </Card>
    </div>
  )
}
