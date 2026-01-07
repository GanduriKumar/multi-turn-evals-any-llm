import React, { useEffect, useState } from 'react'

function Card({ title, children }: { title: string, children: React.ReactNode }) {
  return (
    <div className="rounded-lg border border-gray-200 bg-white shadow-sm">
      <div className="border-b border-gray-100 px-4 py-2 font-medium text-gray-800">{title}</div>
      <div className="p-4">{children}</div>
    </div>
  )
}

type DatasetDoc = any

type GoldenDoc = any

export default function GoldenEditorPage() {
  const [datasets, setDatasets] = useState<{dataset_id: string}[]>([])
  const [selected, setSelected] = useState('')
  const [dataset, setDataset] = useState<DatasetDoc | null>(null)
  const [golden, setGolden] = useState<GoldenDoc | null>(null)
  const [msg, setMsg] = useState<string | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const [saving, setSaving] = useState(false)
  const [overwrite, setOverwrite] = useState(false)
  const [bump, setBump] = useState(true)

  const loadDatasets = async () => {
    try {
      const r = await fetch('/datasets'); const js = await r.json(); setDatasets(js)
      if (js.length && !selected) setSelected(js[0].dataset_id)
    } catch {}
  }

  const load = async (id: string) => {
    setDataset(null); setGolden(null); setErr(null)
    try {
      const [dR, gR] = await Promise.all([
        fetch(`/datasets/${id}`), fetch(`/goldens/${id}`)
      ])
      if (dR.ok) setDataset(await dR.json())
      if (gR.ok) setGolden(await gR.json())
    } catch (e:any) { setErr(e.message || 'Failed to load') }
  }

  useEffect(() => { loadDatasets() }, [])
  useEffect(() => { if (selected) load(selected) }, [selected])

  const save = async () => {
    setSaving(true); setMsg(null); setErr(null)
    try {
      const r = await fetch('/datasets/save', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ dataset, golden, overwrite, bump_version: bump }) })
      const js = await r.json()
      if (!r.ok) throw new Error(js?.detail ? JSON.stringify(js.detail) : 'Save failed')
      setMsg('Saved successfully')
    } catch (e:any) { setErr(e.message || 'Save failed') } finally { setSaving(false) }
  }

  return (
    <div className="grid gap-4">
      <Card title="Select Dataset">
        <div className="flex items-center gap-2 text-sm">
          <select className="border rounded px-2 py-1 min-w-[240px]" value={selected} onChange={e => setSelected(e.target.value)}>
            {datasets.map(d => (<option key={d.dataset_id} value={d.dataset_id}>{d.dataset_id}</option>))}
          </select>
          <label className="inline-flex items-center gap-2"><input type="checkbox" checked={overwrite} onChange={e => setOverwrite(e.target.checked)} /> Overwrite</label>
          <label className="inline-flex items-center gap-2"><input type="checkbox" checked={bump} onChange={e => setBump(e.target.checked)} /> Bump patch version</label>
          <button onClick={save} disabled={saving || !dataset} className="px-3 py-1.5 rounded bg-primary text-white hover:opacity-90 disabled:opacity-50">{saving ? 'Saving…' : 'Save'}</button>
          {msg && <span className="text-gray-700">{msg}</span>}
          {err && <span className="text-danger">{err}</span>}
        </div>
      </Card>

      <Card title="Dataset JSON">
        {dataset ? (
          <textarea className="w-full h-72 text-xs font-mono border rounded p-2" value={JSON.stringify(dataset, null, 2)} onChange={e => setDataset(JSON.parse(e.target.value || '{}'))} />
        ) : <div className="text-sm text-gray-600">No dataset loaded</div>}
      </Card>

      <Card title="Golden JSON">
        {golden ? (
          <textarea className="w-full h-72 text-xs font-mono border rounded p-2" value={JSON.stringify(golden, null, 2)} onChange={e => setGolden(JSON.parse(e.target.value || '{}'))} />
        ) : <div className="text-sm text-gray-600">No golden loaded</div>}
      </Card>
    </div>
  )
}
