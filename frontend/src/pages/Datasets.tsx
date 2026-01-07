import React, { useEffect, useMemo, useState } from 'react'

type DatasetItem = {
  dataset_id: string
  version?: string
  domain?: string
  difficulty?: string
  conversations?: number
  has_golden?: boolean
  valid?: boolean
  errors?: string[]
}

function Card({ title, children }: { title: string, children: React.ReactNode }) {
  return (
    <div className="rounded-lg border border-gray-200 bg-white shadow-sm">
      <div className="border-b border-gray-100 px-4 py-2 font-medium text-gray-800">{title}</div>
      <div className="p-4">{children}</div>
    </div>
  )
}

export default function DatasetsPage() {
  const [list, setList] = useState<DatasetItem[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const fetchList = async () => {
    setLoading(true); setError(null)
    try {
      const r = await fetch('/datasets')
      if (!r.ok) throw new Error(`HTTP ${r.status}`)
      const js = await r.json()
      setList(js)
    } catch (e:any) {
      setError(e.message || 'Failed to load')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { fetchList() }, [])

  const [fileDataset, setFileDataset] = useState<File | null>(null)
  const [fileGolden, setFileGolden] = useState<File | null>(null)
  const [uploadMsg, setUploadMsg] = useState<string | null>(null)
  const [uploading, setUploading] = useState(false)
  const [overwrite, setOverwrite] = useState(false)

  const handleUpload = async () => {
    setUploadMsg(null)
    if (!fileDataset) { setUploadMsg('Choose a dataset JSON'); return }
    setUploading(true)
    try {
      const fd = new FormData()
      fd.append('dataset', fileDataset)
      if (fileGolden) fd.append('golden', fileGolden)
      const r = await fetch(`/datasets/upload?overwrite=${overwrite ? 'true' : 'false'}` , { method: 'POST', body: fd })
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

  return (
    <div className="grid gap-4">
      <Card title="Upload JSON">
        <div className="space-y-3 text-sm">
          <div className="flex items-center gap-2">
            <label className="w-40">Dataset (.dataset.json)</label>
            <input type="file" accept="application/json" onChange={e => setFileDataset(e.target.files?.[0] || null)} />
          </div>
          <div className="flex items-center gap-2">
            <label className="w-40">Golden (.golden.json)</label>
            <input type="file" accept="application/json" onChange={e => setFileGolden(e.target.files?.[0] || null)} />
          </div>
          <label className="flex items-center gap-2">
            <input type="checkbox" checked={overwrite} onChange={e => setOverwrite(e.target.checked)} />
            <span>Overwrite existing files</span>
          </label>
          <div className="flex items-center gap-2">
            <button onClick={handleUpload} disabled={uploading} className="px-3 py-1.5 rounded bg-primary text-white hover:opacity-90 disabled:opacity-50">{uploading ? 'Uploading…' : 'Upload'}</button>
            {uploadMsg && <span className="text-sm text-gray-700 break-all">{uploadMsg}</span>}
          </div>
          <p className="text-xs text-gray-500">Files are validated server-side and saved into the backend datasets/ folder.</p>
        </div>
      </Card>

      <Card title="Datasets">
        <div className="mb-2 flex items-center gap-2">
          <button onClick={fetchList} className="px-3 py-1.5 rounded border hover:bg-gray-50">Refresh</button>
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
                <th className="py-2 pr-4">Golden</th>
                <th className="py-2 pr-4">Valid</th>
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
                  <td className="py-2 pr-4">{row.has_golden ? 'Yes' : 'No'}</td>
                  <td className="py-2 pr-4">{row.valid ? 'Yes' : 'No'}</td>
                </tr>
              ))}
              {list.length === 0 && !loading && (
                <tr><td className="py-3 text-gray-500" colSpan={7}>No datasets</td></tr>
              )}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}
