import React, { useEffect, useState } from 'react'

function Card({ title, children }: { title: string, children: React.ReactNode }) {
  return (
    <div className="rounded-lg border border-gray-200 bg-white shadow-sm">
      <div className="border-b border-gray-100 px-4 py-2 font-medium text-gray-800">{title}</div>
      <div className="p-4">{children}</div>
    </div>
  )
}

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
                  <input type="checkbox" checked={m.enabled} onChange={e => setCfg(x => { const y = {...x!}; y.metrics = y.metrics.map((mm, idx) => idx===i ? {...mm, enabled: e.target.checked} : mm); return y })} />
                  <span className="font-medium">{m.name}</span>
                </label>
                <label className="flex items-center gap-2">
                  <span className="w-24">Weight</span>
                  <input type="number" step="0.1" className="border rounded px-2 py-1 w-28" value={m.weight ?? 1} onChange={e => setCfg(x => { const y = {...x!}; y.metrics = y.metrics.map((mm, idx) => idx===i ? {...mm, weight: Number(e.target.value)} : mm); return y })} />
                </label>
                {'threshold' in m ? (
                  <label className="flex items-center gap-2">
                    <span className="w-24">Threshold</span>
                    <input type="number" step="0.01" min={0} max={1} className="border rounded px-2 py-1 w-28" value={m.threshold ?? 0} onChange={e => setCfg(x => { const y = {...x!}; y.metrics = y.metrics.map((mm, idx) => idx===i ? {...mm, threshold: Number(e.target.value)} : mm); return y })} />
                  </label>
                ) : <div />}
              </div>
            ))}
            <div className="flex items-center gap-2">
              <button onClick={save} disabled={saving} className="px-3 py-1.5 rounded bg-primary text-white hover:opacity-90 disabled:opacity-50">{saving ? 'Saving…' : 'Save'}</button>
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
