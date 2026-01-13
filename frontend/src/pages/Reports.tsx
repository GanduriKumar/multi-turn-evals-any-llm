import React, { useEffect, useMemo, useRef, useState } from 'react'
import Card from '../components/Card'
import Button from '../components/Button'
import Badge from '../components/Badge'
import { Select, Input, Textarea } from '../components/Form'
import { useVertical } from '../context/VerticalContext'
import CircularProgress from '../components/CircularProgress'

type RunListItem = {
  run_id: string
  dataset_id?: string
  model_spec?: string
  has_results: boolean
  created_ts?: number
}

type RunResults = any

 

export default function ReportsPage() {
  const { vertical } = useVertical()
  const [runs, setRuns] = useState<RunListItem[]>([])
  const [runId, setRunId] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [results, setResults] = useState<RunResults | null>(null)
  const [fbRating, setFbRating] = useState<number>(5)
  const [fbNotes, setFbNotes] = useState<string>('')
  const [fbOverrideConv, setFbOverrideConv] = useState<null | boolean>(null)
  const [fbTurnId, setFbTurnId] = useState<string>('')
  const [fbTurnPass, setFbTurnPass] = useState<boolean | null>(null)
  const [fbMsg, setFbMsg] = useState<string | null>(null)
  const [fbErr, setFbErr] = useState<string | null>(null)

  // Compare state
  const [runA, setRunA] = useState('')
  const [runB, setRunB] = useState('')
  const [diff, setDiff] = useState<any | null>(null)
  const [heatmap, setHeatmap] = useState(false)
  const [expandedKey, setExpandedKey] = useState<string | null>(null)
  const [diffLoading, setDiffLoading] = useState(false)
  const [diffError, setDiffError] = useState<string | null>(null)
  const [resultsA, setResultsA] = useState<any | null>(null)
  const [resultsB, setResultsB] = useState<any | null>(null)
  const metricsList = useMemo(() => ['exact','semantic','consistency','adherence','hallucination'], [])
  const metricLabel = (m: string) => (m === 'hallucination' ? 'hallucination (risk↓)' : m)
  const [filterMetric, setFilterMetric] = useState<string | null>(null)
  // Full-page side-by-side viewer state
  const [sbsOpen, setSbsOpen] = useState(false)
  const [sbsKey, setSbsKey] = useState<string | null>(null)
  const [sbsIdx, setSbsIdx] = useState(0)
  const [sbsChangedOnly, setSbsChangedOnly] = useState(false)

  // Report chat state
  const [rcOpen, setRcOpen] = useState(false)
  const [rcConversation, setRcConversation] = useState<string>('') // empty => all conversations
  const [rcHistory, setRcHistory] = useState<{ role: 'user'|'assistant'; content: string }[]>([])
  const [rcInput, setRcInput] = useState('')
  const [rcModel, setRcModel] = useState('')
  const [rcModels, setRcModels] = useState<string[]>([])
  const [rcBusy, setRcBusy] = useState(false)
  const rcChatRef = useRef<HTMLDivElement | null>(null)
  const rcAbortRef = useRef<AbortController | null>(null)
  const rcPollRef = useRef<boolean>(false)

  // Auto-scroll chat to bottom on new messages
  useEffect(() => {
    const el = rcChatRef.current
    if (el) {
      el.scrollTop = el.scrollHeight
    }
  }, [rcHistory, rcOpen])

  const sbsData = useMemo(() => {
    if (!sbsKey || !resultsA || !resultsB || !diff) return null
    const findConv = (res:any) => {
      const convs = (res?.conversations || []) as any[]
      return convs.find(c => (c.conversation_slug || c.conversation_title || c.conversation_id) === sbsKey)
    }
    const ca = findConv(resultsA)
    const cb = findConv(resultsB)
    if (!ca && !cb) return null
    const turnsA = (ca?.turns || []) as any[]
    const turnsB = (cb?.turns || []) as any[]
    const maxLen = Math.max(turnsA.length, turnsB.length)
    const perTurn = (diff.per_turn || []).filter((t:any) => t.key === sbsKey)
    const ptMap: Record<number, any> = {}
    for (const t of perTurn) ptMap[Number(t.turn_index)] = t
    const rows = Array.from({ length: maxLen }).map((_, i) => {
      const ta = turnsA.find(x => Number(x.turn_index) === i)
      const tb = turnsB.find(x => Number(x.turn_index) === i)
      const pt = ptMap[i] || {}
      const user = ta?.user_prompt_snippet || tb?.user_prompt_snippet || ''
      const a = ta ? { pass: ta.turn_pass, out: ta.assistant_output_snippet, metrics: (ta.metrics || {}) } : null
      const b = tb ? { pass: tb.turn_pass, out: tb.assistant_output_snippet, metrics: (tb.metrics || {}) } : null
      const changed = Boolean(pt?.turn_pass?.changed) || Object.values(pt?.metrics || {}).some((m:any) => m?.changed)
      return { idx: i, user, a, b, pt, changed }
    })
    const indices = rows.map((r, i) => i)
    const changedIdx = rows.map((r,i) => (r.changed ? i : -1)).filter(v => v >= 0)
    return { ca, cb, rows, indices, changedIdx }
  }, [sbsKey, resultsA, resultsB, diff])

  const openSideBySide = (key: string) => {
    setSbsKey(key)
    setSbsIdx(0)
    setSbsChangedOnly(false)
    setSbsOpen(true)
  }

  const navSbs = (dir: 1 | -1) => {
    if (!sbsData) return
    const list = sbsChangedOnly ? sbsData.changedIdx : sbsData.indices
    if (!list.length) return
    const pos = list.findIndex(i => i === sbsIdx)
    const nextPos = pos < 0 ? 0 : (pos + (dir === 1 ? 1 : -1) + list.length) % list.length
    setSbsIdx(list[nextPos])
  }
  const convMetricHeat = useMemo(() => {
    if (!diff) return null
    const counts: Record<string, Record<string, number>> = {}
    let maxCount = 0
    for (const row of (diff.per_turn || [])) {
      const key = row.key as string
      const mets = (row.metrics || {}) as any
      for (const m of metricsList) {
        const info = mets[m]
        if (info && info.changed) {
          counts[key] = counts[key] || {}
          counts[key][m] = (counts[key][m] || 0) + 1
          if (counts[key][m] > maxCount) maxCount = counts[key][m]
        }
      }
    }
    const rows = Object.entries(counts).map(([key, m]) => ({ key, counts: m as Record<string, number>, total: Object.values(m as any).reduce((a:number,b:any)=>a+Number(b||0),0) }))
    rows.sort((a,b) => b.total - a.total)
    const top = rows.slice(0, 12)
    return { rows: top, max: maxCount }
  }, [diff, metricsList])

  const loadRuns = async () => {
    setError(null)
    try {
      const r = await fetch(`/runs?vertical=${encodeURIComponent(vertical)}`)
      if (!r.ok) throw new Error(`HTTP ${r.status}`)
      const js = await r.json()
      setRuns(js)
      if (js.length && !runId) setRunId(js[js.length - 1].run_id)
      // Preselect two most recent for comparison
      if (js.length >= 2) {
        const sorted = [...js].sort((a:any,b:any) => (b.created_ts || 0) - (a.created_ts || 0))
        setRunA(sorted[0].run_id)
        setRunB(sorted[1].run_id)
      } else if (js.length === 1) {
        setRunA(js[0].run_id)
        setRunB('')
      }
    } catch (e:any) {
      setError(e.message || 'Failed to load runs')
    }
  }

  const loadResults = async (id: string) => {
    setLoading(true); setError(null)
    try {
      const r = await fetch(`/runs/${id}/results?vertical=${encodeURIComponent(vertical)}`)
      if (!r.ok) {
        if (r.status === 404) {
          // Friendly message when a run has no generated report yet
          setResults(null)
          setError('No report found for this run.')
          return
        }
        throw new Error(`HTTP ${r.status}`)
      }
      const js = await r.json()
      setResults(js)
    } catch (e:any) {
      setError(e.message || 'Failed to load results')
      setResults(null)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { loadRuns() }, [vertical])
  useEffect(() => { if (runId) loadResults(runId) }, [runId, vertical])

  // Auto-open chat if a background job exists for any run in this vertical
  useEffect(() => {
    try {
      let found: { key: string, runId: string, convKey: string } | null = null
      for (let i = 0; i < localStorage.length; i++) {
        const key = localStorage.key(i) || ''
        const prefix = `reportChatJob:${vertical}:`
        if (key.startsWith(prefix)) {
          const rest = key.substring(prefix.length) // <runId>:<convKey>
          const sep = rest.indexOf(':')
          if (sep > 0) {
            const rId = rest.substring(0, sep)
            const cKey = rest.substring(sep + 1)
            found = { key, runId: rId, convKey: cKey }
            break
          }
        }
      }
      if (found) {
        if (!rcOpen) setRcOpen(true)
        if (runId !== found.runId) setRunId(found.runId)
        setRcConversation(found.convKey === 'all' ? '' : found.convKey)
        setRcBusy(true) // reflect in-flight immediately until polling starts
      }
    } catch { /* ignore */ }
  }, [vertical])

  const compareNow = async () => {
    if (!runA || !runB || runA === runB) { setDiff(null); setDiffError('Pick two different runs'); return }
    setDiffError(null); setDiff(null); setDiffLoading(true)
    try {
      const url = `/reports/compare?runA=${encodeURIComponent(runA)}&runB=${encodeURIComponent(runB)}&vertical=${encodeURIComponent(vertical)}`
      const r = await fetch(url)
      if (!r.ok) {
        const js = await r.json().catch(() => ({}))
        throw new Error(js?.detail || `HTTP ${r.status}`)
      }
      const js = await r.json()
      setDiff(js)
      // Prefetch full results to enable side-by-side turn diffs
      try {
        const [ra, rb] = await Promise.all([
          fetch(`/runs/${encodeURIComponent(runA)}/results?vertical=${encodeURIComponent(vertical)}`),
          fetch(`/runs/${encodeURIComponent(runB)}/results?vertical=${encodeURIComponent(vertical)}`)
        ])
        setResultsA(ra.ok ? await ra.json() : null)
        setResultsB(rb.ok ? await rb.json() : null)
      } catch {
        setResultsA(null); setResultsB(null)
      }
    } catch (e:any) {
      setDiffError(e.message || 'Failed to compare')
    } finally {
      setDiffLoading(false)
    }
  }

  const download = (type: 'json' | 'csv' | 'html') => {
    window.open(`/runs/${runId}/artifacts?type=${type}&vertical=${encodeURIComponent(vertical)}`, '_blank')
  }

  const submitFeedback = async () => {
    if (!runId) return
    setFbMsg(null); setFbErr(null)
    try {
      const body: any = { rating: fbRating, notes: fbNotes }
      if (fbOverrideConv !== null) body.override_conversation_pass = fbOverrideConv
      if (fbTurnId && fbTurnPass !== null) body.override_turn = { conversation_id: fbTurnId.split('#')[0], turn_index: Number(fbTurnId.split('#')[1] || 0), pass: fbTurnPass }
      const r = await fetch(`/runs/${runId}/feedback`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) })
      const js = await r.json()
      if (!r.ok) throw new Error(js?.detail || 'Failed to submit feedback')
      setFbMsg('Feedback submitted')
      setFbNotes('')
      setFbOverrideConv(null)
      setFbTurnId(''); setFbTurnPass(null)
    } catch (e:any) {
      setFbErr(e.message || 'Failed to submit feedback')
    }
  }

  const openReportChat = () => {
    if (!runId) return
    setRcOpen(true)
    setRcHistory([])
    setRcInput('')
    setRcConversation('')
  }

  // Configure chat models list and default selection using run model_spec or backend defaults
  useEffect(() => {
    if (!rcOpen) return
    ;(async () => {
      try {
        const r = await fetch('/version')
        if (!r.ok) return
        const s = await r.json()
        const def = s?.models || {}
        const opts: string[] = []
        if (s?.openai_enabled && def.openai) opts.push(`openai:${def.openai}`)
        if (s?.gemini_enabled && def.gemini) opts.push(`gemini:${def.gemini}`)
        if (def.ollama) opts.push(`ollama:${def.ollama}`)
        // Prefer the run's model_spec if available
        const runModel = (results as any)?.model_spec as string | undefined
        const all = [...opts]
        if (runModel && !all.includes(runModel)) all.unshift(runModel)
        setRcModels(all.length ? all : ['openai:gpt-5.1','gemini:gemini-2.5','ollama:llama3.2:latest'])
        // Only set default if user hasn't typed yet
        if (!rcHistory.length) {
          if (runModel) setRcModel(runModel)
          else if (opts.length) setRcModel(opts[0])
          else setRcModel('openai:gpt-5.1')
        }
      } catch {
        // fallback options
        setRcModels(['openai:gpt-5.1','gemini:gemini-2.5','ollama:llama3.2:latest'])
        if (!rcHistory.length && !rcModel) setRcModel('openai:gpt-5.1')
      }
    })()
  }, [rcOpen, runId, results])

  const sendReportChat = async () => {
    if (!runId || !rcInput.trim()) return
    const body: any = { run_id: runId, message: rcInput, model: rcModel, history: rcHistory }
    if (rcConversation) body.conversation_id = rcConversation
    setRcBusy(true)
    try {
      const hist = [...rcHistory, { role: 'user', content: rcInput }]
      setRcHistory(hist)
      setRcInput('')
      // Submit background job so it continues across navigation
      const submit = await fetch(`/chat/report/submit?vertical=${encodeURIComponent(vertical)}`, {
        method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body)
      })
      const js = await submit.json().catch(() => ({} as any))
      if (!submit.ok || !js?.job_id) {
        const msg = (js?.detail || 'Failed to submit job') as string
        setRcHistory([...hist, { role: 'assistant', content: `Error: ${msg}` }])
        return
      }
      const jobId = js.job_id as string
      // persist job so we can resume after navigation
      const convKey = rcConversation || 'all'
      const storageKey = `reportChatJob:${vertical}:${runId}:${convKey}`
      try {
        localStorage.setItem(storageKey, jobId)
        // persist the pending user message so we can show it immediately on resume
        localStorage.setItem(`${storageKey}:userContent`, body.message)
      } catch {}
      // Poll for job status until completed/failed; keep polling even if user navigates away
      rcPollRef.current = true
      setRcBusy(true)
      let done = false
      while (!done && rcPollRef.current) {
        await new Promise(res => setTimeout(res, 1200))
        try {
          const st = await fetch(`/chat/report/job/${encodeURIComponent(jobId)}?run_id=${encodeURIComponent(runId)}&vertical=${encodeURIComponent(vertical)}`)
          if (!st.ok) break
          const sj = await st.json()
          if (sj?.status === 'completed') {
            setRcHistory(h => [...h, { role: 'assistant', content: String(sj.content || '') }])
            try {
              localStorage.removeItem(storageKey)
              localStorage.removeItem(`${storageKey}:userContent`)
            } catch {}
            done = true
          } else if (sj?.status === 'failed') {
            setRcHistory(h => [...h, { role: 'assistant', content: `Error: ${String(sj.error || 'failed')}` }])
            try {
              localStorage.removeItem(storageKey)
              localStorage.removeItem(`${storageKey}:userContent`)
            } catch {}
            done = true
          }
        } catch { /* ignore transient */ }
      }
      setRcBusy(false)
    } catch (e:any) {
      const aborted = e && (e.name === 'AbortError' || e.message === 'The user aborted a request.')
      const hist = [...rcHistory, { role: 'user', content: rcInput }, { role: 'assistant', content: aborted ? 'Cancelled.' : `Error: ${e.message || 'failed to chat'}` }]
      setRcHistory(hist)
      setRcInput('')
    } finally {
      setRcBusy(false)
      if (rcAbortRef.current) rcAbortRef.current = null
    }
  }

  const cancelReportChat = () => {
    if (rcBusy && rcAbortRef.current) {
      rcAbortRef.current.abort()
    }
    rcPollRef.current = false
    setRcBusy(false)
  }

  // Restore chat history when user reopens the page/card
  useEffect(() => {
    if (!rcOpen || !runId) return
    ;(async () => {
      try {
        const url = `/chat/report/log?run_id=${encodeURIComponent(runId)}&vertical=${encodeURIComponent(vertical)}${rcConversation ? `&conversation_id=${encodeURIComponent(rcConversation)}` : ''}`
        const r = await fetch(url)
        if (!r.ok) return
        const js = await r.json()
        const events = Array.isArray(js?.events) ? js.events : []
        // Convert persisted events to UI history
        const hist: { role:'user'|'assistant'; content:string }[] = []
        for (const e of events) {
          if (e?.user) hist.push({ role: 'user', content: String(e.user) })
          if (e?.assistant) hist.push({ role: 'assistant', content: String(e.assistant) })
        }
        // If a job is in-flight, ensure the pending user message is visible even if the log race missed it
        try {
          const convKey = rcConversation || 'all'
          const storageKey = `reportChatJob:${vertical}:${runId}:${convKey}`
          const pending = localStorage.getItem(storageKey)
          if (pending) {
            const pendingMsg = localStorage.getItem(`${storageKey}:userContent`)
            if (pendingMsg) {
              const hasLastUser = hist.length > 0 && hist[hist.length - 1].role === 'user' && hist[hist.length - 1].content === pendingMsg
              const alreadyIncluded = hist.some(m => m.role === 'user' && m.content === pendingMsg)
              if (!alreadyIncluded) hist.push({ role: 'user', content: pendingMsg })
              else if (!hasLastUser) {
                // move last occurrence to end for clarity
                const filtered = hist.filter(m => !(m.role === 'user' && m.content === pendingMsg))
                filtered.push({ role: 'user', content: pendingMsg })
                while (hist.length) hist.pop()
                hist.push(...filtered)
              }
            }
          }
        } catch { /* ignore */ }
        if (hist.length) setRcHistory(hist)
      } catch { /* ignore */ }
    })()
  }, [rcOpen, runId, rcConversation, vertical])

  // Resume polling if a job exists for this run when returning to page
  useEffect(() => {
    if (!runId) return
    try {
      // find any job for this run/vertical
      for (let i = 0; i < localStorage.length; i++) {
        const key = localStorage.key(i) || ''
        const prefix = `reportChatJob:${vertical}:${runId}:`
        if (key.startsWith(prefix)) {
          const convKey = key.substring(prefix.length)
          const jobId = localStorage.getItem(key) || ''
          if (!jobId) continue
          if (!rcOpen) setRcOpen(true)
          setRcConversation(convKey === 'all' ? '' : convKey)
          // start polling
          rcPollRef.current = true
          setRcBusy(true)
          // show pending user message immediately from localStorage if available
          try {
            const pendingMsg = localStorage.getItem(`${prefix}${convKey}:userContent`)
            if (pendingMsg) setRcHistory(h => [...h, { role: 'user', content: pendingMsg }])
          } catch { /* ignore */ }
          ;(async () => {
            let done = false
            while (!done && rcPollRef.current) {
              await new Promise(res => setTimeout(res, 1200))
              try {
                const st = await fetch(`/chat/report/job/${encodeURIComponent(jobId)}?run_id=${encodeURIComponent(runId)}&vertical=${encodeURIComponent(vertical)}`)
                if (!st.ok) break
                const sj = await st.json()
                if (sj?.status === 'completed') {
                  setRcHistory(h => [...h, { role: 'assistant', content: String(sj.content || '') }])
                  try {
                    localStorage.removeItem(key)
                    localStorage.removeItem(`${key}:userContent`)
                  } catch {}
                  done = true
                } else if (sj?.status === 'failed') {
                  setRcHistory(h => [...h, { role: 'assistant', content: `Error: ${String(sj.error || 'failed')}` }])
                  try {
                    localStorage.removeItem(key)
                    localStorage.removeItem(`${key}:userContent`)
                  } catch {}
                  done = true
                }
              } catch { /* ignore */ }
            }
            setRcBusy(false)
          })()
          break
        }
      }
    } catch { /* ignore */ }
  }, [runId, vertical])

  return (
    <div className="grid gap-4">
      <Card title="Compare Reports">
        <div className="flex flex-wrap items-center gap-2 text-sm">
          <Select className="min-w-[220px]" value={runA} onChange={e => setRunA(e.target.value)}>
            <option value="" disabled>Select Report A</option>
            {runs.map(r => (<option key={r.run_id} value={r.run_id}>{r.run_id} — {r.dataset_id || '?'} — {r.model_spec || '?'}</option>))}
          </Select>
          <span className="px-1">vs</span>
          <Select className="min-w-[220px]" value={runB} onChange={e => setRunB(e.target.value)}>
            <option value="" disabled>Select Report B</option>
            {runs.map(r => (<option key={r.run_id} value={r.run_id}>{r.run_id} — {r.dataset_id || '?'} — {r.model_spec || '?'}</option>))}
          </Select>
          <Button variant="primary" onClick={compareNow} disabled={!runA || !runB || runA === runB}>Compare</Button>
          {diffError && <span className="text-danger">{diffError}</span>}
        </div>
        {diffLoading && <div className="text-sm text-gray-600 mt-2">Computing diff…</div>}
        {diff && (
          <div className="mt-4 space-y-4">
            {diff.alignment && (diff.alignment.unmatched_a?.length || diff.alignment.unmatched_b?.length) ? (
              <div className="text-xs text-warning font-medium">{diff.alignment.note || 'Note: datasets differ; comparison is directional.'}</div>
            ) : null}
            <div className="grid gap-4 sm:grid-cols-2">
              <Card title="Conversations Pass Rate">
                <div className="flex items-center gap-6">
                  <div className="flex flex-col items-center gap-1">
                    <CircularProgress value={Math.round(diff.runA.summary.conv.pass_rate || 0)} size={128} bezel variant="primary" />
                    <div className="text-xs text-gray-600">A: {diff.runA.run_id}</div>
                  </div>
                  <div className="flex flex-col items-center gap-1">
                    <CircularProgress value={Math.round(diff.runB.summary.conv.pass_rate || 0)} size={128} bezel variant="success" />
                    <div className="text-xs text-gray-600">B: {diff.runB.run_id}</div>
                  </div>
                </div>
              </Card>
              <Card title="Turns Pass Rate">
                <div className="flex items-center gap-6">
                  <div className="flex flex-col items-center gap-1">
                    <CircularProgress value={Math.round(diff.runA.summary.turn.pass_rate || 0)} size={128} bezel variant="primary" />
                    <div className="text-xs text-gray-600">A Turns</div>
                  </div>
                  <div className="flex flex-col items-center gap-1">
                    <CircularProgress value={Math.round(diff.runB.summary.turn.pass_rate || 0)} size={128} bezel variant="success" />
                    <div className="text-xs text-gray-600">B Turns</div>
                  </div>
                </div>
              </Card>
            </div>
            <div className="flex items-center justify-end gap-2">
              <Button variant="secondary" onClick={() => window.open(`/reports/compare?runA=${encodeURIComponent(runA)}&runB=${encodeURIComponent(runB)}&vertical=${encodeURIComponent(vertical)}&type=csv`, '_blank')}>Download Diff CSV</Button>
              <Button variant="warning" onClick={() => window.open(`/reports/compare?runA=${encodeURIComponent(runA)}&runB=${encodeURIComponent(runB)}&vertical=${encodeURIComponent(vertical)}&type=pdf`, '_blank')}>Download Diff PDF</Button>
            </div>
            <Card title="Per-metric Pass Rate Delta">
              <div className="overflow-x-auto">
                <table className="min-w-full text-xs">
                  <thead>
                    <tr className="text-left text-gray-600">
                      <th className="py-1 pr-3">Metric</th>
                      <th className="py-1 pr-3">A</th>
                      <th className="py-1 pr-3">B</th>
                      <th className="py-1 pr-3">Δ (B−A)</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(diff.metrics_delta || {}).map(([m, row]: any) => (
                      <tr key={m} className="border-t">
                        <td className="py-1 pr-3 font-medium">{metricLabel(String(m))}</td>
                        <td className="py-1 pr-3">{(row.a_pass_rate || 0).toFixed(2)}%</td>
                        <td className="py-1 pr-3">{(row.b_pass_rate || 0).toFixed(2)}%</td>
                        <td className="py-1 pr-3">
                          <div className="w-48 h-2 bg-gray-200 rounded overflow-hidden">
                            {(() => {
                              const v = Math.max(-100, Math.min(100, row.delta || 0))
                              const pct = Math.abs(v)
                              const color = v >= 0 ? 'bg-success' : 'bg-danger'
                              return <div className={`h-2 ${color}`} style={{ width: `${pct}%` }} />
                            })()}
                          </div>
                          <span className="text-[11px] text-gray-600 ml-2">{(row.delta || 0).toFixed(2)}%</span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>
            <Card title="Metrics Heatmap">
              <div className="flex flex-wrap gap-2 text-[11px]">
                {Object.entries(diff.metrics_delta || {}).map(([m, row]: any) => {
                  const d = Number(row.delta || 0)
                  const pos = d >= 0
                  const intensity = Math.min(100, Math.abs(d))
                  const bg = pos ? `rgba(16,185,129,${0.15 + 0.003*intensity})` : `rgba(239,68,68,${0.15 + 0.003*intensity})`
                  const bd = pos ? 'border-emerald-400' : 'border-rose-400'
                  return (
                    <div key={m} className={`px-2 py-1 rounded border ${bd}`} style={{ background: bg }}>
                      <span className="font-medium mr-2">{metricLabel(String(m))}</span>
                      <span>{(row.a_pass_rate || 0).toFixed(1)}%</span>
                      <span className="mx-1">→</span>
                      <span>{(row.b_pass_rate || 0).toFixed(1)}%</span>
                      <span className="ml-2">Δ {(d).toFixed(1)}%</span>
                    </div>
                  )
                })}
              </div>
            </Card>
            <Card title="Domain/Behavior Delta">
              <div className="overflow-x-auto">
                <table className="min-w-full text-xs">
                  <thead>
                    <tr className="text-left text-gray-600">
                      <th className="py-1 pr-3">Domain</th>
                      <th className="py-1 pr-3">Behavior</th>
                      <th className="py-1 pr-3">A pass%</th>
                      <th className="py-1 pr-3">B pass%</th>
                      <th className="py-1 pr-3">Δ</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(diff.domain_behavior_delta || []).map((r:any, i:number) => (
                      <tr key={`${r.domain}|${r.behavior}|${i}`} className="border-t">
                        <td className="py-1 pr-3">{r.domain}</td>
                        <td className="py-1 pr-3">{r.behavior}</td>
                        <td className="py-1 pr-3">{(r.a.pass_rate || 0).toFixed(2)}%</td>
                        <td className="py-1 pr-3">{(r.b.pass_rate || 0).toFixed(2)}%</td>
                        <td className="py-1 pr-3">{(r.delta.pass_rate || 0).toFixed(2)}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>
            {convMetricHeat && convMetricHeat.rows.length > 0 && (
              <Card title="Conversation × Metric Change Heatmap (Top 12)">
                <div className="overflow-x-auto">
                  <table className="min-w-full text-xs">
                    <thead>
                      <tr className="text-left text-gray-600">
                        <th className="py-1 pr-3">Conversation</th>
                        {metricsList.map(m => (
                          <th key={m} className="py-1 px-2">{metricLabel(m)}</th>
                        ))}
                        <th className="py-1 px-2">Total</th>
                      </tr>
                    </thead>
                    <tbody>
                      {convMetricHeat.rows.map((r:any) => (
                        <tr key={r.key} className="border-t">
                          <td className="py-1 pr-3 truncate max-w-[260px]" title={r.key}>{r.key}</td>
                          {metricsList.map(m => {
                            const c = Number((r.counts || {})[m] || 0)
                            const alpha = convMetricHeat.max ? Math.min(1, 0.12 + 0.88 * (c / convMetricHeat.max)) : 0
                            const bg = `rgba(59,130,246,${alpha})` // blue heat
                            const isClickable = c > 0
                            return (
                              <td
                                key={`${r.key}-${m}`}
                                className={`py-1 px-2 text-center ${isClickable ? 'cursor-pointer underline-offset-2 hover:underline' : ''}`}
                                style={{ background: c ? bg : undefined }}
                                title={`${m}: ${c}${isClickable ? ' (click to drill down)' : ''}`}
                                onClick={() => { if (isClickable) { setExpandedKey(r.key); setFilterMetric(m) } }}
                              >
                                {c || ''}
                              </td>
                            )
                          })}
                          <td className="py-1 px-2 text-center font-medium">{r.total}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </Card>
            )}
            <Card title="Per-conversation Changes (Top 20)">
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs text-gray-500">Click a row to expand turn-level diffs</span>
                <label className="flex items-center gap-2 text-xs">
                  <input type="checkbox" checked={heatmap} onChange={(e) => setHeatmap(e.target.checked)} /> Heatmap
                </label>
              </div>
              <div className="overflow-x-auto">
                <table className="min-w-full text-xs">
                  <thead>
                    <tr className="text-left text-gray-600">
                      <th className="py-1 pr-3">Conversation key</th>
                      <th className="py-1 pr-3">Pass A→B</th>
                      <th className="py-1 pr-3">Failed turns Δ</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(diff.per_conversation || []).slice(0,20).map((row:any) => {
                      const isOpen = expandedKey === row.key
                      return (
                        <React.Fragment key={row.key}>
                          <tr className="border-t cursor-pointer" onClick={() => setExpandedKey(isOpen ? null : row.key)}>
                            <td className="py-1 pr-3">{row.key}</td>
                            <td className="py-1 pr-3">{String(row.a.conversation_pass)} → {String(row.b.conversation_pass)}</td>
                            <td className="py-1 pr-3">{row.delta.failed_turns_delta}</td>
                          </tr>
                          {isOpen && (
                            <tr>
                              <td className="py-2 pr-3 bg-base-200" colSpan={3}>
                                <div className="text-[11px] font-medium mb-1 flex items-center gap-2">
                                  <span>Turn-level differences</span>
                                  {filterMetric && (
                                    <span className="badge badge-outline text-[10px]">
                                      Filter: {filterMetric}
                                      <button className="ml-2 text-[10px] text-primary" onClick={() => setFilterMetric(null)}>clear</button>
                                    </span>
                                  )}
                                  <Button variant="secondary" onClick={(e) => { e.stopPropagation(); openSideBySide(row.key) }}>Open side-by-side</Button>
                                </div>
                                <div className="overflow-x-auto">
                                  <table className="table-compact w-full text-[11px]">
                                    <thead>
                                      <tr>
                                        <th className="py-1 pr-2 text-left">#</th>
                                        <th className="py-1 pr-2 text-left">Pass A→B</th>
                                        <th className="py-1 pr-2 text-left">Changed metrics</th>
                                      </tr>
                                    </thead>
                                    <tbody>
                                      {(diff.per_turn || [])
                                        .filter((t:any) => t.key === row.key)
                                        .filter((t:any) => !filterMetric || ((t.metrics || {})[filterMetric]?.changed))
                                        .map((t:any) => {
                                        const changed = Object.entries(t.metrics || {}).filter(([,v]: any) => (v as any)?.changed).map(([k]) => k).join(', ')
                                        const cls = t.turn_pass?.a === false && t.turn_pass?.b === true ? 'text-success' : t.turn_pass?.a === true && t.turn_pass?.b === false ? 'text-danger' : ''
                                        return (
                                          <tr key={`${t.key}-${t.turn_index}`} className={cls}>
                                            <td className="py-1 pr-2">{(t.turn_index ?? 0) + 1}</td>
                                            <td className="py-1 pr-2">{String(t.turn_pass?.a)} → {String(t.turn_pass?.b)}</td>
                                            <td className="py-1 pr-2 truncate max-w-[480px]" title={changed || '—'}>{changed || '—'}</td>
                                          </tr>
                                        )
                                      })}
                                    </tbody>
                                  </table>
                                </div>
                                {/* Side-by-side snippets for turns with changed pass */}
                                <div className="mt-2 grid md:grid-cols-2 gap-2">
                                  {(diff.per_turn || [])
                                    .filter((t:any) => t.key === row.key && t.turn_pass?.changed)
                                    .filter((t:any) => !filterMetric || ((t.metrics || {})[filterMetric]?.changed))
                                    .slice(0,6)
                                    .map((t:any) => {
                                    const findConv = (res:any) => {
                                      const convs = (res?.conversations || []) as any[]
                                      return convs.find(c => (c.conversation_slug || c.conversation_title || c.conversation_id) === row.key)
                                    }
                                    const ca = resultsA ? findConv(resultsA) : null
                                    const cb = resultsB ? findConv(resultsB) : null
                                    const ta = (ca?.turns || []).find((x:any) => Number(x.turn_index) === Number(t.turn_index))
                                    const tb = (cb?.turns || []).find((x:any) => Number(x.turn_index) === Number(t.turn_index))
                                    const up = ta?.user_prompt_snippet || tb?.user_prompt_snippet || ''
                                    return (
                                      <div key={`sbs-${t.turn_index}`} className="bg-base-100 rounded p-2 border">
                                        <div className="text-[11px] mb-1 font-medium">Turn {(t.turn_index ?? 0) + 1} — Pass {String(t.turn_pass?.a)} → {String(t.turn_pass?.b)}</div>
                                        <div className="text-[11px] text-gray-600 mb-1">User: {up || '—'}</div>
                                        <div className="grid grid-cols-2 gap-2 text-[11px]">
                                          <div>
                                            <div className="text-[10px] text-gray-500 mb-1">A</div>
                                            <div className="whitespace-pre-wrap break-words max-h-28 overflow-auto border rounded p-2">{ta?.assistant_output_snippet || '—'}</div>
                                          </div>
                                          <div>
                                            <div className="text-[10px] text-gray-500 mb-1">B</div>
                                            <div className="whitespace-pre-wrap break-words max-h-28 overflow-auto border rounded p-2">{tb?.assistant_output_snippet || '—'}</div>
                                          </div>
                                        </div>
                                      </div>
                                    )
                                  })}
                                </div>
                                {heatmap && (
                                  <div className="mt-2 grid grid-cols-12 gap-1">
                                    {Array.from({ length: (diff.summary?.turns?.b?.total || 0) }).map((_, idx) => {
                                      const t = (diff.per_turn || []).find((x:any) => x.key === row.key && x.turn_index === idx)
                                      const a = t?.turn_pass?.a; const b = t?.turn_pass?.b
                                      const color = a === b ? (a ? 'bg-success' : 'bg-danger') : 'bg-warning'
                                      return <div key={idx} className={`h-2 ${color}`} title={`Turn ${idx+1}: A=${String(a)} B=${String(b)}`}></div>
                                    })}
                                  </div>
                                )}
                              </td>
                            </tr>
                          )}
                        </React.Fragment>
                      )
                    })}
                  </tbody>
                </table>
              </div>
            </Card>
          </div>
        )}
      </Card>
      {sbsOpen && sbsData && (
        <div className="fixed inset-0 z-50 bg-black/90 flex items-center justify-center p-4" onClick={() => setSbsOpen(false)}>
          <div className="bg-base-100 w-full max-w-6xl max-h-[90vh] overflow-auto rounded shadow-lg" onClick={e => e.stopPropagation()}>
            <div className="p-3 border-b flex items-center justify-between">
              <div className="text-sm">
                <div className="font-medium">Side-by-side diff</div>
                <div className="text-gray-600">Conversation: <span className="font-mono">{sbsKey}</span></div>
              </div>
              <div className="flex items-center gap-2">
                <label className="text-xs flex items-center gap-2">
                  <input type="checkbox" checked={sbsChangedOnly} onChange={e => { setSbsChangedOnly(e.target.checked); setSbsIdx(0) }} />
                  Changed turns only
                </label>
                <Button variant="secondary" onClick={() => navSbs(-1)}>Prev</Button>
                <div className="text-xs">Turn {sbsIdx + 1} / {sbsData.rows.length}</div>
                <Button variant="secondary" onClick={() => navSbs(1)}>Next</Button>
                <Button variant="warning" onClick={() => setSbsOpen(false)}>Close</Button>
              </div>
            </div>
            {(() => {
              const r = sbsData.rows[sbsIdx] || {}
              const pt = r.pt || {}
              const metrics = Object.keys(pt.metrics || {})
              const changedMetrics = metrics.filter(m => (pt.metrics[m] || {}).changed)
              return (
                <div className="p-3 space-y-3">
                  <div className="text-[12px] text-gray-600">User: {r.user || '—'}</div>
                  <div className="grid md:grid-cols-2 gap-3">
                    <div>
                      <div className="text-[11px] text-gray-500 mb-1">A — {diff.runA.run_id}</div>
                      <div className="border rounded p-2 whitespace-pre-wrap break-words min-h-[120px]">{r.a?.out || '—'}</div>
                    </div>
                    <div>
                      <div className="text-[11px] text-gray-500 mb-1">B — {diff.runB.run_id}</div>
                      <div className="border rounded p-2 whitespace-pre-wrap break-words min-h-[120px]">{r.b?.out || '—'}</div>
                    </div>
                  </div>
                  <div>
                    <div className="text-[11px] font-medium mb-1">Metrics</div>
                    <div className="flex flex-wrap gap-2 text-[11px]">
                      {metrics.length === 0 ? <span className="text-gray-500">No metrics</span> : metrics.map(m => {
                        const info:any = (pt.metrics || {})[m] || {}
                        const cls = info.changed ? 'badge-warning' : (info.b ? 'badge-success' : 'badge-ghost')
                        return (
                          <span key={m} className={`badge ${cls}`}>
                            {m}: {String(info.a)} → {String(info.b)}
                          </span>
                        )
                      })}
                    </div>
                  </div>
                  {changedMetrics.length > 0 && (
                    <div className="text-[11px] text-gray-600">Changed: {changedMetrics.join(', ')}</div>
                  )}
                </div>
              )
            })()}
          </div>
        </div>
      )}
      <Card title="Select Run">
        <div className="flex flex-wrap gap-2 items-center text-sm">
          <Select className="min-w-[240px]" value={runId} onChange={e => setRunId(e.target.value)}>
            <option value="" disabled>Select a run</option>
            {runs.map(r => (
              <option key={r.run_id} value={r.run_id}>{r.run_id} — {r.dataset_id || '?'} — {r.model_spec || '?'}</option>
            ))}
          </Select>
          <Button variant="secondary" onClick={loadRuns}>Refresh</Button>
          <div className="grow" />
          <Button variant="secondary" disabled={!runId} onClick={() => download('json')}>Download JSON</Button>
          <Button variant="secondary" disabled={!runId} onClick={() => download('csv')}>Download CSV</Button>
          <Button variant="warning" disabled={!runId} onClick={() => download('html')}>Open Report</Button>
          <Button disabled={!runId || !results} onClick={openReportChat} title={!results ? 'No report found for this run' : ''}>Chat</Button>
        </div>
      </Card>

      {rcOpen && (
        <Card title="Chat with report">
          <div className="text-xs mb-2">Run: <span className="font-mono">{runId}</span></div>
          <div className="flex items-center gap-2 mb-3">
            <select className="select select-bordered select-sm" value={rcModel} onChange={e => setRcModel(e.target.value)}>
              {rcModels.map(m => <option key={m} value={m}>{m}</option>)}
            </select>
            <Button variant="warning" onClick={() => setRcOpen(false)}>Close</Button>
          </div>
          {!results && (
            <div className="alert alert-warning text-xs mb-2">No report found for this run. Open a run with a generated report first.</div>
          )}
          <label className="text-xs">Conversation (optional focus)</label>
          <select className="select select-bordered w-full select-sm" value={rcConversation} onChange={e => setRcConversation(e.target.value)}>
            <option value="">All conversations</option>
            {(results?.conversations || []).map((c:any) => {
              const title = c.conversation_title || c.conversation_slug || c.conversation_id
              const id = c.conversation_id || c.conversation_slug
              return <option key={id} value={id}>{title} — {id}</option>
            })}
          </select>
          <div ref={rcChatRef} className="border rounded p-2 h-72 overflow-auto bg-base-200 text-base-content mt-3">
            {rcHistory.length === 0 ? (
              <div className="text-[12px] text-gray-800">Ask about failures, deltas, or a specific conversation. The assistant has summary context and optionally a conversation transcript snippet.</div>
            ) : (
              <div className="space-y-2">
                {rcHistory.map((m,i) => (
                  <div key={i} className={`p-2 rounded ${m.role === 'user' ? 'bg-base-100' : 'bg-base-300'}`}>
                    <div className="text-base font-semibold opacity-70">{m.role}</div>
                    <div className="text-sm whitespace-pre-wrap break-words">{m.content}</div>
                  </div>
                ))}
              </div>
            )}
          </div>
          <div className="flex items-center gap-2 mt-2">
            <input className="input input-bordered w-full input-sm" placeholder="Type your question" value={rcInput} onChange={e => setRcInput(e.target.value)} onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendReportChat() } }} />
            {rcBusy ? (
              <Button onClick={cancelReportChat} aria-label="Cancel sending" title="Cancel sending">
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" className="w-5 h-5">
                  <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="2" />
                  <rect x="9" y="9" width="6" height="6" fill="currentColor" className="animate-pulse" />
                </svg>
              </Button>
            ) : (
              <Button onClick={sendReportChat} aria-label="Send" title="Send">
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="w-5 h-5">
                  <path d="M12 3c.3 0 .6.12.8.34l7 7a1.14 1.14 0 0 1 .2 1.26A1 1 0 0 1 19 12h-5v8a1 1 0 1 1-2 0v-8H7a1 1 0 0 1-1-.4 1.14 1.14 0 0 1 .2-1.26l7-7c.2-.22.5-.34.8-.34z" />
                </svg>
              </Button>
            )}
          </div>
        </Card>
      )}

      <Card title="Report Summary">
        {loading ? (
          <div className="text-sm text-gray-600">Loading…</div>
        ) : error ? (
          <div className="text-sm text-danger">{error}</div>
        ) : results ? (
          <div className="text-sm space-y-3">
            <div className="flex flex-wrap gap-4">
              <div><span className="text-gray-500">Run:</span> <span className="font-mono">{results.run_id}</span></div>
              <div><span className="text-gray-500">Dataset:</span> {results.dataset_id}</div>
              <div><span className="text-gray-500">Model:</span> <span className="text-success font-medium">{results.model_spec}</span></div>
              <div><span className="text-gray-500">Conversations:</span> {results.conversations?.length ?? 0}</div>
            </div>
            {results.domain_description && (
              <div className="text-xs text-gray-600 max-w-3xl">{results.domain_description}</div>
            )}
            <div className="overflow-x-auto">
              <table className="min-w-full text-xs">
                <thead>
                  <tr className="text-left text-gray-600">
                    <th className="py-2 pr-4">Conversation</th>
                    <th className="py-2 pr-4">Pass</th>
                    <th className="py-2 pr-4">Weighted rate</th>
                    <th className="py-2 pr-4">Turns</th>
                    <th className="py-2 pr-4">Failed turns</th>
                    <th className="py-2 pr-4">Failed metrics</th>
                  </tr>
                </thead>
                <tbody>
                  {(results.conversations || []).map((c:any) => {
                    const title = c.conversation_title || c.conversation_slug || c.conversation_id
                    const slug = c.conversation_slug || c.conversation_id
                    const domain = c.domain
                    const behavior = c.behavior
                    const scenario = c.scenario
                    const s = c.summary || {}
                    const turnsTotal = typeof s.total_user_turns === 'number' ? s.total_user_turns : (c.turns || []).length
                    const failedTurns = s.failed_turns_count ?? 0
                    const failedMetrics = Array.isArray(s.failed_metrics) ? s.failed_metrics.join(', ') : ''
                    return (
                      <tr key={c.conversation_id} className="border-t align-top">
                        <td className="py-2 pr-4">
                          <div className="font-medium">{title}</div>
                          <div className="text-[11px] text-gray-500">{slug}</div>
                          <div className="text-[11px] text-gray-600">{[domain, behavior, scenario].filter(Boolean).join(' • ')}</div>
                          {c.conversation_description && (
                            <div className="text-[11px] text-gray-600 truncate max-w-[520px]" title={c.conversation_description}>{c.conversation_description}</div>
                          )}
                        </td>
                        <td className="py-2 pr-4">{s.conversation_pass ? <Badge variant="success">pass</Badge> : <Badge variant="danger">fail</Badge>}</td>
                        <td className="py-2 pr-4">{(s.weighted_pass_rate ?? 0).toFixed(2)}</td>
                        <td className="py-2 pr-4">{turnsTotal}</td>
                        <td className="py-2 pr-4">{failedTurns}</td>
                        <td className="py-2 pr-4 whitespace-pre-wrap break-words max-w-[360px]">{failedMetrics}</td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
            <div className="mt-4 border-t pt-3">
              <div className="font-medium mb-2">Human Feedback</div>
              <div className="grid gap-3 sm:grid-cols-2">
                <label className="flex items-center gap-2">
                  <span className="w-28">Rating</span>
                  <Input type="number" min={1} max={5} className="w-24" value={fbRating} onChange={e => setFbRating(Number(e.target.value))} />
                </label>
                <label className="flex items-center gap-2">
                  <span className="w-28">Conv override</span>
                  <Select className="w-40" value={fbOverrideConv === null ? '' : (fbOverrideConv ? 'true' : 'false')} onChange={e => setFbOverrideConv(e.target.value === '' ? null : e.target.value === 'true')}>
                    <option value="">no override</option>
                    <option value="true">force pass</option>
                    <option value="false">force fail</option>
                  </Select>
                </label>
                <label className="flex items-center gap-2">
                  <span className="w-28">Turn override</span>
                  <Select className="grow" value={fbTurnId} onChange={e => setFbTurnId(e.target.value)}>
                    <option value="">none</option>
                    {(results.conversations || []).flatMap((c:any) => {
                      const title = c.conversation_title || c.conversation_slug || c.conversation_id
                      return (c.turns || []).map((t:any) => ({ key: `${c.conversation_id}#${t.turn_index}`, label: `${title} — turn ${t.turn_index}` }))
                    }).map((item:any) => (
                      <option key={item.key} value={item.key}>{item.label}</option>
                    ))}
                  </Select>
                </label>
                <label className="flex items-center gap-2">
                  <span className="w-28">Turn pass?</span>
                  <Select className="w-40" value={fbTurnPass === null ? '' : (fbTurnPass ? 'true' : 'false')} onChange={e => setFbTurnPass(e.target.value === '' ? null : e.target.value === 'true')}>
                    <option value="">no override</option>
                    <option value="true">pass</option>
                    <option value="false">fail</option>
                  </Select>
                </label>
              </div>
              <label className="block mt-3">
                <span className="sr-only">Notes</span>
                <Textarea className="mt-1 w-full h-24 text-xs" placeholder="Evaluator notes" value={fbNotes} onChange={e => setFbNotes(e.target.value)} />
              </label>
              <div className="mt-2 flex items-center gap-2">
                <Button variant="success" onClick={submitFeedback}>Submit Feedback</Button>
                {fbMsg && <span className="text-gray-700">{fbMsg}</span>}
                {fbErr && <span className="text-danger">{fbErr}</span>}
              </div>
            </div>
          </div>
        ) : (
          <div className="text-sm text-gray-700">Select a run to view the report.</div>
        )}
      </Card>

    </div>
  )
}

