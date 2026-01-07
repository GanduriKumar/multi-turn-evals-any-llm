import React, { useEffect, useState } from 'react'
import Card from '../components/Card'
import Button from '../components/Button'
import { Select, Checkbox, Textarea } from '../components/Form'

 

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
          <Select className="min-w-[240px]" value={selected} onChange={e => setSelected(e.target.value)}>
            {datasets.map(d => (<option key={d.dataset_id} value={d.dataset_id}>{d.dataset_id}</option>))}
          </Select>
          <label className="inline-flex items-center gap-2"><Checkbox checked={overwrite} onChange={e => setOverwrite((e.target as HTMLInputElement).checked)} /> Overwrite</label>
          <label className="inline-flex items-center gap-2"><Checkbox checked={bump} onChange={e => setBump((e.target as HTMLInputElement).checked)} /> Bump patch version</label>
          <Button onClick={save} disabled={saving || !dataset}>{saving ? 'Saving…' : 'Save'}</Button>
          {msg && <span className="text-gray-700">{msg}</span>}
          {err && <span className="text-danger">{err}</span>}
        </div>
      </Card>

      <Card title="Dataset JSON">
        {dataset ? (
          <Textarea className="w-full h-72 text-xs font-mono" value={JSON.stringify(dataset, null, 2)} onChange={e => setDataset(JSON.parse(e.target.value || '{}'))} />
        ) : <div className="text-sm text-gray-600">No dataset loaded</div>}
      </Card>

      <Card title="Golden JSON">
        {golden ? (
          <Textarea className="w-full h-72 text-xs font-mono" value={JSON.stringify(golden, null, 2)} onChange={e => setGolden(JSON.parse(e.target.value || '{}'))} />
        ) : <div className="text-sm text-gray-600">No golden loaded</div>}
      </Card>
    </div>
  )
}
