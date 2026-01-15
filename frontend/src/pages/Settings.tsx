import React, { useEffect, useState } from 'react'
import Card from '../components/Card'
import Button from '../components/Button'
import { Input } from '../components/Form'
import { useVertical } from '../context/VerticalContext'

type Settings = {
  ollama_host: string | null
  gemini_enabled: boolean
  openai_enabled?: boolean
  semantic_threshold: number
  hallucination_threshold?: number
  models?: { ollama?: string; gemini?: string; openai?: string }
  embed_model?: string
  industry_vertical?: string
  supported_verticals?: string[]
}

export default function SettingsPage() {
  const { vertical, supported, setVertical } = useVertical()
  const [settings, setSettings] = useState<Settings | null>(null)
  const [ollama, setOllama] = useState('')
  const [apiKeyGemini, setApiKeyGemini] = useState('')
  const [apiKeyOpenAI, setApiKeyOpenAI] = useState('')
  const [modelOllama, setModelOllama] = useState('llama3.2:latest')
  const [modelGemini, setModelGemini] = useState('gemini-2.5')
  const [modelOpenAI, setModelOpenAI] = useState('gpt-5.1')
  const [embedModel, setEmbedModel] = useState('nomic-embed-text')
  const [semThr, setSemThr] = useState(0.8)
  const [hallThr, setHallThr] = useState(0.8)
  const [saving, setSaving] = useState(false)
  const [msg, setMsg] = useState<string | null>(null)
  const [err, setErr] = useState<string | null>(null)

  const load = async () => {
    setErr(null)
    try {
      const [settingsRes, metricsRes] = await Promise.all([
        fetch('/settings'),
        fetch('/metrics-config').catch(() => null),
      ])
      if (!settingsRes.ok) throw new Error(`HTTP ${settingsRes.status}`)
      const js = await settingsRes.json()
      const metricsCfg = metricsRes && metricsRes.ok ? await metricsRes.json() : null
      setSettings(js)
      setOllama(js.ollama_host || '')
      const semFromSettings = Number(js.semantic_threshold)
      const hallFromSettings = typeof js.hallucination_threshold === 'number' ? Number(js.hallucination_threshold) : undefined
      const semFromMetrics = (() => {
        if (!metricsCfg || !Array.isArray(metricsCfg?.metrics)) return undefined
        const mm = metricsCfg.metrics.find((m: any) => String(m?.name) === 'semantic_similarity')
        const t = (mm && typeof mm.threshold === 'number') ? Number(mm.threshold) : undefined
        return (typeof t === 'number' && isFinite(t)) ? t : undefined
      })()
      const hallFromMetrics = (() => {
        if (!metricsCfg || !Array.isArray(metricsCfg?.metrics)) return undefined
        const mm = metricsCfg.metrics.find((m: any) => String(m?.name) === 'hallucination')
        const t = (mm && typeof mm.threshold === 'number') ? Number(mm.threshold) : undefined
        return (typeof t === 'number' && isFinite(t)) ? t : undefined
      })()
      setSemThr(typeof semFromMetrics === 'number' ? semFromMetrics : (isFinite(semFromSettings) ? semFromSettings : 0.8))
      setHallThr(typeof hallFromMetrics === 'number' ? hallFromMetrics : (typeof hallFromSettings === 'number' ? hallFromSettings : 0.8))
      const m = js.models || {}
      if (m.ollama) setModelOllama(m.ollama)
      if (m.gemini) setModelGemini(m.gemini)
      if (m.openai) setModelOpenAI(m.openai)
      if (js.embed_model) setEmbedModel(js.embed_model)
    } catch (e:any) {
      setErr(e.message || 'Failed to load settings')
    }
  }

  useEffect(() => { load() }, [])

  const save = async () => {
    setSaving(true); setMsg(null); setErr(null)
    try {
      const r = await fetch('/settings', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ ollama_host: ollama, google_api_key: apiKeyGemini || undefined, openai_api_key: apiKeyOpenAI || undefined, semantic_threshold: semThr, hallucination_threshold: hallThr, ollama_model: modelOllama, gemini_model: modelGemini, openai_model: modelOpenAI, embed_model: embedModel }) })
      const js = await r.json()
      if (!r.ok) throw new Error(js?.detail || 'Save failed')
      // Keep metrics-config thresholds in sync with Provider Settings values
      try {
        const mr = await fetch('/metrics-config')
        if (mr.ok) {
          const cfg = await mr.json()
          if (cfg && Array.isArray(cfg.metrics)) {
            const updated = { ...cfg }
            updated.metrics = cfg.metrics.map((m: any) => {
              const name = String(m?.name || '')
              if (name === 'semantic_similarity') return { ...m, threshold: Number(semThr) }
              if (name === 'hallucination') return { ...m, threshold: Number(hallThr) }
              return m
            })
            await fetch('/metrics-config', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(updated) })
          }
        }
      } catch {}
      setMsg('Saved.')
      await load()
    } catch (e:any) {
      setErr(e.message || 'Save failed')
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="grid gap-4">
      <Card title="Vertical">
        <div className="flex items-center gap-2 text-sm">
          <span className="w-40">Industry Vertical</span>
          <select className="border rounded px-2 py-1" value={vertical} onChange={e => setVertical(e.target.value)}>
            {supported.map(v => <option key={v} value={v}>{v}</option>)}
          </select>
          <span className="text-xs text-gray-600">Affects dataset and run storage locations</span>
        </div>
      </Card>
      <Card title="Provider Settings (.env dev-only)">
        {settings ? (
          <div className="space-y-3 text-sm">
            <label className="flex items-center gap-2">
              <span className="w-40">OLLAMA_HOST</span>
              <Input className="grow" value={ollama} onChange={e => setOllama(e.target.value)} placeholder="http://localhost:11434" />
            </label>
            <label className="flex items-center gap-2">
              <span className="w-40">EMBED_MODEL</span>
              <Input className="grow" value={embedModel} onChange={e => setEmbedModel(e.target.value)} placeholder="nomic-embed-text" />
            </label>
            <label className="flex items-center gap-2">
              <span className="w-40">OLLAMA_MODEL</span>
              <Input className="grow" value={modelOllama} onChange={e => setModelOllama(e.target.value)} placeholder="llama3.2:latest" />
            </label>
            <label className="flex items-center gap-2">
              <span className="w-40">GOOGLE_API_KEY</span>
              <Input className="grow" value={apiKeyGemini} onChange={e => setApiKeyGemini(e.target.value)} placeholder="leave blank to keep" />
            </label>
            <label className="flex items-center gap-2">
              <span className="w-40">GEMINI_MODEL</span>
              <Input className="grow" value={modelGemini} onChange={e => setModelGemini(e.target.value)} placeholder="gemini-2.5" />
            </label>
            <label className="flex items-center gap-2">
              <span className="w-40">OPENAI_API_KEY</span>
              <Input className="grow" value={apiKeyOpenAI} onChange={e => setApiKeyOpenAI(e.target.value)} placeholder="leave blank to keep" />
            </label>
            <label className="flex items-center gap-2">
              <span className="w-40">OPENAI_MODEL</span>
              <Input className="grow" value={modelOpenAI} onChange={e => setModelOpenAI(e.target.value)} placeholder="gpt-5.1" />
            </label>
            <label className="flex items-center gap-2">
              <span className="w-40">Semantic threshold</span>
              <Input type="number" step="0.01" min={0} max={1} className="w-28" value={semThr} onChange={e => setSemThr(Number(e.target.value))} />
            </label>
            <label className="flex items-center gap-2">
              <span className="w-40">Hallucination threshold</span>
              <Input type="number" step="0.01" min={0} max={1} className="w-28" value={hallThr} onChange={e => setHallThr(Number(e.target.value))} />
            </label>
            <div className="flex items-center gap-2">
              <Button variant="success" onClick={save} disabled={saving}>{saving ? 'Saving…' : 'Save'}</Button>
              {msg && <span className="text-gray-700">{msg}</span>}
              {err && <span className="text-danger">{err}</span>}
            </div>
            <p className="text-xs text-gray-500">Values are stored in a local .env at repo root. Do not commit secrets.</p>
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
