import React, { useEffect, useMemo, useRef, useState } from 'react'
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
  const [chatOpen, setChatOpen] = useState(false)
  const [chatDataset, setChatDataset] = useState<string | null>(null)
  const [chatConversation, setChatConversation] = useState<string | null>(null)
  const [chatConvs, setChatConvs] = useState<any[]>([])
  const [chatHistory, setChatHistory] = useState<{ role: 'user'|'assistant'; content: string }[]>([])
  const [chatInput, setChatInput] = useState('')
  const [chatModel, setChatModel] = useState('')
  const [chatModels, setChatModels] = useState<string[]>([])
  const [chatBusy, setChatBusy] = useState(false)
  const chatRef = useRef<HTMLDivElement | null>(null)
  const chatAbortRef = useRef<AbortController | null>(null)
  const chatPollRef = useRef<boolean>(false)

  // Auto-scroll chat to bottom on new messages
  useEffect(() => {
    const el = chatRef.current
    if (el) {
      el.scrollTop = el.scrollHeight
    }
  }, [chatHistory, chatOpen])

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
  const openChat = async (datasetId: string) => {
    setChatDataset(datasetId)
    setChatConversation(null)
    setChatConvs([])
    setChatHistory([])
    setChatInput('')
    setChatOpen(true)
    // Load dataset to populate conversation dropdown
    try {
      const r = await fetch(`/datasets/${encodeURIComponent(datasetId)}?vertical=${encodeURIComponent(vertical)}`)
      if (r.ok) {
        const ds = await r.json()
        const convs = Array.isArray(ds?.conversations) ? ds.conversations : []
        setChatConvs(convs)
        if (convs.length > 0) setChatConversation(convs[0].conversation_id)
      }
    } catch {
      // ignore and keep manual entry
    }
    // Load model options based on /version
    try {
      const v = await fetch('/version')
      if (v.ok) {
        const s = await v.json()
        const def = s?.models || {}
        const opts: string[] = []
        if (s?.openai_enabled && def.openai) opts.push(`openai:${def.openai}`)
        if (s?.gemini_enabled && def.gemini) opts.push(`gemini:${def.gemini}`)
        if (def.ollama) opts.push(`ollama:${def.ollama}`)
        setChatModels(opts.length ? opts : ['openai:gpt-5.1','gemini:gemini-2.5','ollama:llama3.2:latest'])
        if (!chatHistory.length) setChatModel(opts[0] || 'openai:gpt-5.1')
      } else {
        setChatModels(['openai:gpt-5.1','gemini:gemini-2.5','ollama:llama3.2:latest'])
        if (!chatHistory.length) setChatModel('openai:gpt-5.1')
      }
    } catch {
      setChatModels(['openai:gpt-5.1','gemini:gemini-2.5','ollama:llama3.2:latest'])
      if (!chatHistory.length) setChatModel('openai:gpt-5.1')
    }
  }

  const sendChat = async () => {
    if (!chatDataset || !chatInput.trim()) return
    const [provider, model] = chatModel.includes(':') ? chatModel.split(':', 1) && [chatModel.split(':')[0], chatModel.split(':').slice(1).join(':')] : ['openai', chatModel]
    const body = { dataset_id: chatDataset, conversation_id: chatConversation || undefined, message: chatInput, model: chatModel, history: chatHistory }
    setChatBusy(true)
    try {
      const newHist = [...chatHistory, { role: 'user', content: chatInput }]
      setChatHistory(newHist)
      setChatInput('')
      // Submit background job and poll
      const submit = await fetch(`/chat/dataset/submit?vertical=${encodeURIComponent(vertical)}`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) })
      const js = await submit.json().catch(() => ({} as any))
      if (!submit.ok || !js?.job_id) {
        const msg = (js?.detail || 'Failed to submit job') as string
        setChatHistory(h => [...h, { role: 'assistant', content: `Error: ${msg}` }])
        return
      }
      const jobId = js.job_id as string
      // persist job
      const convKey = chatConversation || 'None'
      const storageKey = `datasetChatJob:${vertical}:${chatDataset}:${convKey}`
      try {
        localStorage.setItem(storageKey, jobId)
        // persist the pending user message so we can show it immediately on resume
        localStorage.setItem(`${storageKey}:userContent`, body.message)
      } catch {}
      let done = false
      chatPollRef.current = true
      setChatBusy(true)
      while (!done && chatPollRef.current) {
        await new Promise(res => setTimeout(res, 1200))
        try {
          const st = await fetch(`/chat/dataset/job/${encodeURIComponent(jobId)}?dataset_id=${encodeURIComponent(chatDataset)}&vertical=${encodeURIComponent(vertical)}`)
          if (!st.ok) break
          const sj = await st.json()
          if (sj?.status === 'completed') {
            setChatHistory(h => [...h, { role: 'assistant', content: String(sj.content || '') }])
            try {
              localStorage.removeItem(storageKey)
              localStorage.removeItem(`${storageKey}:userContent`)
            } catch {}
            done = true
          } else if (sj?.status === 'failed') {
            setChatHistory(h => [...h, { role: 'assistant', content: `Error: ${String(sj.error || 'failed')}` }])
            try {
              localStorage.removeItem(storageKey)
              localStorage.removeItem(`${storageKey}:userContent`)
            } catch {}
            done = true
          }
        } catch { /* ignore transient */ }
      }
      setChatBusy(false)
    } catch (e:any) {
      const aborted = e && (e.name === 'AbortError' || e.message === 'The user aborted a request.')
      const newHist = [...chatHistory, { role: 'user', content: chatInput }, { role: 'assistant', content: aborted ? 'Cancelled.' : `Error: ${e.message || 'failed to chat'}` }]
      setChatHistory(newHist)
      setChatInput('')
    } finally {
      setChatBusy(false)
      if (chatAbortRef.current) chatAbortRef.current = null
    }
  }

  const cancelChat = () => {
    if (chatBusy && chatAbortRef.current) {
      chatAbortRef.current.abort()
    }
    chatPollRef.current = false
    setChatBusy(false)
  }

  // Restore dataset chat history on reopen
  useEffect(() => {
    if (!chatOpen || !chatDataset) return
    ;(async () => {
      try {
        const url = `/chat/dataset/log?dataset_id=${encodeURIComponent(chatDataset)}&vertical=${encodeURIComponent(vertical)}${chatConversation ? `&conversation_id=${encodeURIComponent(chatConversation)}` : ''}`
        const r = await fetch(url)
        if (!r.ok) return
        const js = await r.json()
        const events = Array.isArray(js?.events) ? js.events : []
        const hist: { role:'user'|'assistant'; content:string }[] = []
        for (const e of events) {
          if (e?.user) hist.push({ role: 'user', content: String(e.user) })
          if (e?.assistant) hist.push({ role: 'assistant', content: String(e.assistant) })
        }
        // If a job is in-flight, ensure the pending user message is visible in UI even if file write raced
        try {
          const convKey = chatConversation || 'None'
          const storageKey = `datasetChatJob:${vertical}:${chatDataset}:${convKey}`
          const pending = localStorage.getItem(storageKey)
          if (pending) {
            const pendingMsg = localStorage.getItem(`${storageKey}:userContent`)
            if (pendingMsg) {
              const already = hist.some(m => m.role === 'user' && m.content === pendingMsg)
              if (!already) hist.push({ role: 'user', content: pendingMsg })
            }
          }
        } catch { /* ignore */ }
        if (hist.length) setChatHistory(hist)
      } catch { /* ignore */ }
    })()
  }, [chatOpen, chatDataset, chatConversation, vertical])

  // Resume polling if a dataset chat job exists for this dataset when returning
  useEffect(() => {
    if (!chatDataset) return
    try {
      const prefix = `datasetChatJob:${vertical}:${chatDataset}:`
      for (let i = 0; i < localStorage.length; i++) {
        const key = localStorage.key(i) || ''
        if (key.startsWith(prefix)) {
          const convKey = key.substring(prefix.length)
          const jobId = localStorage.getItem(key) || ''
          if (!jobId) continue
          if (!chatOpen) setChatOpen(true)
          setChatConversation(convKey === 'None' ? null : convKey)
          chatPollRef.current = true
          setChatBusy(true)
          // show pending user message immediately from localStorage if available
          try {
            const pendingMsg = localStorage.getItem(`${prefix}${convKey}:userContent`)
            if (pendingMsg) setChatHistory(h => [...h, { role: 'user', content: pendingMsg }])
          } catch { /* ignore */ }
          ;(async () => {
            let done = false
            while (!done && chatPollRef.current) {
              await new Promise(res => setTimeout(res, 1200))
              try {
                const st = await fetch(`/chat/dataset/job/${encodeURIComponent(jobId)}?dataset_id=${encodeURIComponent(chatDataset)}&vertical=${encodeURIComponent(vertical)}`)
                if (!st.ok) break
                const sj = await st.json()
                if (sj?.status === 'completed') {
                  setChatHistory(h => [...h, { role: 'assistant', content: String(sj.content || '') }])
                  try {
                    localStorage.removeItem(key)
                    localStorage.removeItem(`${key}:userContent`)
                  } catch {}
                  done = true
                } else if (sj?.status === 'failed') {
                  setChatHistory(h => [...h, { role: 'assistant', content: `Error: ${String(sj.error || 'failed')}` }])
                  try {
                    localStorage.removeItem(key)
                    localStorage.removeItem(`${key}:userContent`)
                  } catch {}
                  done = true
                }
              } catch { /* ignore */ }
            }
            setChatBusy(false)
          })()
          break
        }
      }
    } catch { /* ignore */ }
  }, [chatDataset, vertical])

  useEffect(() => { fetchList() }, [vertical])

  // Auto-open dataset chat if a background job exists for any dataset in this vertical
  useEffect(() => {
    try {
      let found: { key: string, datasetId: string, convKey: string } | null = null
      for (let i = 0; i < localStorage.length; i++) {
        const key = localStorage.key(i) || ''
        const prefix = `datasetChatJob:${vertical}:`
        if (key.startsWith(prefix)) {
          const rest = key.substring(prefix.length) // <datasetId>:<convKey>
          const sep = rest.indexOf(':')
          if (sep > 0) {
            const dsId = rest.substring(0, sep)
            const cKey = rest.substring(sep + 1)
            found = { key, datasetId: dsId, convKey: cKey }
            break
          }
        }
      }
      if (found) {
        if (!chatOpen) setChatOpen(true)
        setChatDataset(found.datasetId)
        setChatConversation(found.convKey === 'None' ? null : found.convKey)
        setChatBusy(true) // reflect in-flight until resume polling kicks in
      }
    } catch { /* ignore */ }
  }, [vertical])

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
                <th className="py-2 pr-4">Chat</th>
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
                  <td className="py-2 pr-4">
                    <Button onClick={() => openChat(row.dataset_id)}>Chat</Button>
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

      {chatOpen && (
        <Card title="Chat with dataset">
          <div className="text-xs mb-2">Dataset: <span className="font-mono">{chatDataset}</span></div>
          <div className="flex items-center gap-2 mb-3">
            <select className="select select-bordered select-sm" value={chatModel} onChange={e => setChatModel(e.target.value)}>
              {chatModels.map(m => <option key={m} value={m}>{m}</option>)}
            </select>
            <Button variant="warning" onClick={() => setChatOpen(false)}>Close</Button>
          </div>
          <div className="space-y-3">
            <label className="text-xs">Conversation</label>
            {chatConvs.length > 0 ? (
              <select className="select select-bordered w-full select-sm" value={chatConversation || ''} onChange={e => setChatConversation(e.target.value)}>
                <option value="">All conversations</option>
                {chatConvs.map((c:any) => {
                  const title = c.conversation_title || c.conversation_slug || c.conversation_id
                  return <option key={c.conversation_id} value={c.conversation_id}>{title} — {c.conversation_id}</option>
                })}
              </select>
            ) : (
              <input className="input input-bordered w-full input-sm" placeholder="e.g. conv-001" value={chatConversation || ''} onChange={e => setChatConversation(e.target.value)} />
            )}
            <div ref={chatRef} className="border rounded p-2 h-72 overflow-auto bg-base-200 text-base-content">
              {chatHistory.length === 0 ? (
                <div className="text-[12px] text-gray-800">Ask a question about the selected conversation. The assistant will use the conversation metadata and user turns as context.</div>
              ) : (
                <div className="space-y-2">
                  {chatHistory.map((m,i) => (
                    <div key={i} className={`p-2 rounded ${m.role === 'user' ? 'bg-base-100' : 'bg-base-300'}`}>
                      <div className="text-base font-semibold opacity-70">{m.role}</div>
                      <div className="text-sm whitespace-pre-wrap break-words">{m.content}</div>
                    </div>
                  ))}
                </div>
              )}
            </div>
            <div className="flex items-center gap-2">
              <input className="input input-bordered w-full input-sm" placeholder="Type your question" value={chatInput} onChange={e => setChatInput(e.target.value)} onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendChat() } }} />
              {chatBusy ? (
                <Button onClick={cancelChat} aria-label="Cancel sending" title="Cancel sending">
                  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" className="w-5 h-5">
                    <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="2" />
                    <rect x="9" y="9" width="6" height="6" fill="currentColor" className="animate-pulse" />
                  </svg>
                </Button>
              ) : (
                <Button onClick={sendChat} aria-label="Send" title="Send">
                  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="w-5 h-5">
                    <path d="M12 3c.3 0 .6.12.8.34l7 7a1.14 1.14 0 0 1 .2 1.26A1 1 0 0 1 19 12h-5v8a1 1 0 1 1-2 0v-8H7a1 1 0 0 1-1-.4 1.14 1.14 0 0 1 .2-1.26l7-7c.2-.22.5-.34.8-.34z" />
                  </svg>
                </Button>
              )}
            </div>
          </div>
        </Card>
      )}
    </div>
  )
}
