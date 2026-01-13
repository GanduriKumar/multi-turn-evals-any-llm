import React, { useEffect, useMemo, useState } from 'react'
import Card from '../components/Card'
import Button from '../components/Button'
import Badge from '../components/Badge'
import { Input, Select, Checkbox } from '../components/Form'
import { useVertical } from '../context/VerticalContext'

type DatasetItem = {
  dataset_id: string
  version?: string
  domain?: string
  difficulty?: string
  conversations?: number
  turns_per_conversation?: number | null
  has_golden?: boolean
  valid?: boolean
  errors?: string[]
}

export default function DatasetsPage() {
  const { vertical } = useVertical()
  const [list, setList] = useState<DatasetItem[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const fetchList = async () => {
    setLoading(true); setError(null)
    try {
      const r = await fetch(`/datasets?vertical=${encodeURIComponent(vertical)}`)
      if (!r.ok) throw new Error(`HTTP ${r.status}`)
      const js = await r.json()
      setList(js)
    } catch (e:any) {
      setError(e.message || 'Failed to load')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { fetchList() }, [vertical])

  const [fileDataset, setFileDataset] = useState<File | null>(null)
  const [fileGolden, setFileGolden] = useState<File | null>(null)
  const [uploadMsg, setUploadMsg] = useState<string | null>(null)
  const [uploading, setUploading] = useState(false)
  const [overwrite, setOverwrite] = useState(false)
  const [reportMsg, setReportMsg] = useState<string | null>(null)
  const [reportErr, setReportErr] = useState<string | null>(null)
  const [reportBusyId, setReportBusyId] = useState<string | null>(null)

  const handleUpload = async () => {
    setUploadMsg(null)
    if (!fileDataset) { setUploadMsg('Choose a dataset JSON'); return }
    setUploading(true)
    try {
      const fd = new FormData()
      fd.append('dataset', fileDataset)
      if (fileGolden) fd.append('golden', fileGolden)
      const r = await fetch(`/datasets/upload?overwrite=${overwrite ? 'true' : 'false'}&vertical=${encodeURIComponent(vertical)}` , { method: 'POST', body: fd })
      const ct = r.headers.get('content-type') || ''
      const body = ct.includes('application/json') ? await r.json() : await r.text()
      if (!r.ok) {
        const msg = typeof body === 'string' ? body : (body?.detail ? JSON.stringify(body.detail) : 'Upload failed')
        throw new Error(msg)
      }
      setUploadMsg('Uploaded successfully')
      setFileDataset(null)
      setFileGolden(null)
      await fetchList()
    } catch (e:any) {
      setUploadMsg(e.message || 'Upload failed')
    } finally {
      setUploading(false)
    }
  }

  const downloadPerTurn = async (datasetId: string) => {
    setReportMsg(null); setReportErr(null); setReportBusyId(datasetId)
    try {
      // fetch dataset and golden
      const [dsRes, gdRes] = await Promise.all([
        fetch(`/datasets/${encodeURIComponent(datasetId)}?vertical=${encodeURIComponent(vertical)}`),
        fetch(`/goldens/${encodeURIComponent(datasetId)}?vertical=${encodeURIComponent(vertical)}`)
      ])
      if (!dsRes.ok) throw new Error(`Dataset fetch failed (${dsRes.status})`)
      if (!gdRes.ok) throw new Error(`Golden fetch failed (${gdRes.status})`)
      const dataset = await dsRes.json()
      const golden = await gdRes.json()
      const r = await fetch('/coverage/per-turn.csv', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ dataset, golden })
      })
      const text = await r.text()
      if (!r.ok) throw new Error(text || 'Failed to generate report')
      const blob = new Blob([text], { type: 'text/csv' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `${datasetId}-per-turn.csv`
      a.click()
      URL.revokeObjectURL(url)
      setReportMsg(`Downloaded ${datasetId}-per-turn.csv`)
    } catch (e:any) {
      setReportErr(e.message || 'Failed to generate per-turn report')
    } finally {
      setReportBusyId(null)
    }
  }

  return (
    <div className="grid gap-4">
      <Card title="Upload JSON">
        <div className="space-y-3 text-sm">
          <div className="flex items-center gap-2">
            <label className="w-40">Dataset (.dataset.json)</label>
            <Input type="file" accept="application/json" onChange={e => setFileDataset(e.target.files?.[0] || null)} />
          </div>
          <div className="flex items-center gap-2">
            <label className="w-40">Golden (.golden.json)</label>
            <Input type="file" accept="application/json" onChange={e => setFileGolden(e.target.files?.[0] || null)} />
          </div>
          <label className="flex items-center gap-2">
            <Checkbox checked={overwrite} onChange={e => setOverwrite((e.target as HTMLInputElement).checked)} />
            <span>Overwrite existing files</span>
          </label>
          <div className="flex items-center gap-2">
            <Button onClick={handleUpload} disabled={uploading}>{uploading ? 'Uploading…' : 'Upload'}</Button>
            {uploadMsg && <span className="text-sm text-gray-700 break-all">{uploadMsg}</span>}
          </div>
          <p className="text-xs text-gray-500">Files are validated server-side and saved into the backend datasets/ folder.</p>
        </div>
      </Card>

      <Card title="Datasets">
        <div className="mb-2 flex items-center gap-2">
          <Button variant="secondary" onClick={fetchList}>Refresh</Button>
          {loading && <span className="text-sm text-gray-600">Loading…</span>}
          {error && <span className="text-sm text-danger">{error}</span>}
        </div>
        <div className="overflow-x-auto">
          <table className="min-w-full text-sm">
            <thead>
              <tr className="text-left text-gray-600">
                <th className="py-2 pr-4">ID</th>
                <th className="py-2 pr-4">Version</th>
                <th className="py-2 pr-4">Domain</th>
                <th className="py-2 pr-4">Difficulty</th>
                <th className="py-2 pr-4">Conversations</th>
                <th className="py-2 pr-4">Turns/Conversation</th>
                <th className="py-2 pr-4">Golden</th>
                <th className="py-2 pr-4">Valid</th>
                <th className="py-2 pr-4">Reports</th>
              </tr>
            </thead>
            <tbody>
              {list.map(row => (
                <tr key={row.dataset_id} className="border-t">
                  <td className="py-2 pr-4 font-medium">{row.dataset_id}</td>
                  <td className="py-2 pr-4">{row.version || '-'}</td>
                  <td className="py-2 pr-4">{row.domain || '-'}</td>
                  <td className="py-2 pr-4">{row.difficulty || '-'}</td>
                  <td className="py-2 pr-4">{row.conversations ?? '-'}</td>
                  <td className="py-2 pr-4">{row.turns_per_conversation ?? '-'}</td>
                  <td className="py-2 pr-4">{row.has_golden ? <Badge variant="success">Yes</Badge> : <Badge variant="warning">No</Badge>}</td>
                  <td className="py-2 pr-4">{row.valid ? <Badge variant="success">Valid</Badge> : <Badge variant="danger">Invalid</Badge>}</td>
                  <td className="py-2 pr-4">
                    <div className="flex items-center gap-2">
                      <Button
                        variant="secondary"
                        disabled={!row.has_golden || reportBusyId === row.dataset_id}
                        onClick={() => downloadPerTurn(row.dataset_id)}
                      >
                        {reportBusyId === row.dataset_id ? 'Generating…' : 'Per-turn CSV'}
                      </Button>
                    </div>
                  </td>
                </tr>
              ))}
              {list.length === 0 && !loading && (
                <tr><td className="py-3 text-gray-500" colSpan={7}>No datasets</td></tr>
              )}
            </tbody>
          </table>
          <div className="mt-2 text-sm">
            {reportMsg && <span className="text-success">{reportMsg}</span>}
            {reportErr && <span className="text-danger">{reportErr}</span>}
          </div>
        </div>
      </Card>
    </div>
  )
}
