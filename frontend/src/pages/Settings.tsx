import React, { useEffect, useState } from 'react'

type Settings = {
  ollama_host: string | null
  gemini_enabled: boolean
  semantic_threshold: number
}

function Card({ title, children }: { title: string, children: React.ReactNode }) {
  return (
    <div className="rounded-lg border border-gray-200 bg-white shadow-sm">
      <div className="border-b border-gray-100 px-4 py-2 font-medium text-gray-800">{title}</div>
      <div className="p-4">{children}</div>
    </div>
  )
}

export default function SettingsPage() {
  const [settings, setSettings] = useState<Settings | null>(null)
  const [ollama, setOllama] = useState('')
  const [apiKey, setApiKey] = useState('')
  const [semThr, setSemThr] = useState(0.8)
  const [saving, setSaving] = useState(false)
  const [msg, setMsg] = useState<string | null>(null)
  const [err, setErr] = useState<string | null>(null)

  const load = async () => {
    setErr(null)
    try {
      const r = await fetch('/settings')
      if (!r.ok) throw new Error(`HTTP ${r.status}`)
      const js = await r.json()
      setSettings(js)
      setOllama(js.ollama_host || '')
      setSemThr(Number(js.semantic_threshold) || 0.8)
    } catch (e:any) {
      setErr(e.message || 'Failed to load settings')
    }
  }

  useEffect(() => { load() }, [])

  const save = async () => {
    setSaving(true); setMsg(null); setErr(null)
    try {
      const r = await fetch('/settings', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ ollama_host: ollama, google_api_key: apiKey || undefined, semantic_threshold: semThr }) })
      const js = await r.json()
      if (!r.ok) throw new Error(js?.detail || 'Save failed')
      setMsg('Saved. Restart backend to apply to providers.')
      await load()
    } catch (e:any) {
      setErr(e.message || 'Save failed')
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="grid gap-4">
      <Card title="Provider Settings (.env dev-only)">
        {settings ? (
          <div className="space-y-3 text-sm">
            <label className="flex items-center gap-2">
              <span className="w-40">OLLAMA_HOST</span>
              <input className="border rounded px-2 py-1 grow" value={ollama} onChange={e => setOllama(e.target.value)} placeholder="http://localhost:11434" />
            </label>
            <label className="flex items-center gap-2">
              <span className="w-40">GOOGLE_API_KEY</span>
              <input className="border rounded px-2 py-1 grow" value={apiKey} onChange={e => setApiKey(e.target.value)} placeholder="leave blank to keep" />
            </label>
            <label className="flex items-center gap-2">
              <span className="w-40">Semantic threshold</span>
              <input type="number" step="0.01" min={0} max={1} className="border rounded px-2 py-1 w-28" value={semThr} onChange={e => setSemThr(Number(e.target.value))} />
            </label>
            <div className="flex items-center gap-2">
              <button onClick={save} disabled={saving} className="px-3 py-1.5 rounded bg-primary text-white hover:opacity-90 disabled:opacity-50">{saving ? 'Saving…' : 'Save'}</button>
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
