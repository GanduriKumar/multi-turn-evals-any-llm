import { useEffect, useMemo, useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import { compareRuns } from '../utils/api'

// Small helper for number formatting
function fmt(n: number) {
  return Number(n ?? 0).toFixed(4)
}

export default function RunComparisonPage() {
  const [params, setParams] = useSearchParams()
  const [baseline, setBaseline] = useState(params.get('baseline') || '')
  const [current, setCurrent] = useState(params.get('current') || '')
  const [metricFilter, setMetricFilter] = useState('')
  const [datasetFilter, setDatasetFilter] = useState('')
  const [data, setData] = useState<any | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  // Update URL when ids change
  useEffect(() => {
    const p = new URLSearchParams(params)
    baseline ? p.set('baseline', baseline) : p.delete('baseline')
    current ? p.set('current', current) : p.delete('current')
    setParams(p, { replace: true })
  }, [baseline, current])

  async function onCompare() {
    setLoading(true)
    setError(null)
    setData(null)
    try {
      const res = await compareRuns(baseline, current)
      setData(res)
    } catch (e: any) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  // Build filter options from metrics_by_dataset and per_dataset
  const allMetrics = useMemo(() => {
    const s = new Set<string>()
    for (const r of data?.metrics_by_dataset || []) s.add(r.metric)
    return ['', ...Array.from(s)]
  }, [data])
  const allDatasets = useMemo(() => {
    const s = new Set<string>()
    for (const r of data?.per_dataset || []) s.add(r.dataset_id)
    for (const r of data?.metrics_by_dataset || []) s.add(r.dataset_id)
    return ['', ...Array.from(s)]
  }, [data])

  const perDatasetFiltered = useMemo(() => {
    return (data?.per_dataset || []).filter((r: any) => !datasetFilter || r.dataset_id === datasetFilter)
  }, [data, datasetFilter])
  const metricsFiltered = useMemo(() => {
    return (data?.metrics_by_dataset || []).filter((r: any) => {
      if (datasetFilter && r.dataset_id !== datasetFilter) return false
      if (metricFilter && r.metric !== metricFilter) return false
      return true
    })
  }, [data, datasetFilter, metricFilter])

  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <div>
          <h2 className="text-xl font-semibold">Run Comparison</h2>
          <div className="text-sm text-gray-600">Compare two runs and view deltas by dataset and metric.</div>
        </div>
        <div className="flex gap-2">
          {baseline && <Link className="px-3 py-2 rounded border" to={`/dashboard/${baseline}`}>Baseline Dashboard</Link>}
          {current && <Link className="px-3 py-2 rounded border" to={`/dashboard/${current}`}>Current Dashboard</Link>}
        </div>
      </div>

      <div className="p-4 bg-white border rounded mb-4 grid gap-3 md:grid-cols-2">
        <label className="text-sm">
          <span className="text-gray-700">Baseline Run ID</span>
          <input className="mt-1 w-full border rounded p-2" value={baseline} onChange={(e) => setBaseline(e.target.value)} placeholder="run_a" />
        </label>
        <label className="text-sm">
          <span className="text-gray-700">Current Run ID</span>
          <input className="mt-1 w-full border rounded p-2" value={current} onChange={(e) => setCurrent(e.target.value)} placeholder="run_b" />
        </label>
        <div className="md:col-span-2 flex items-center gap-3">
          <button className="px-3 py-2 rounded border" onClick={onCompare} disabled={!baseline || !current || loading}>
            {loading ? 'Comparing…' : 'Compare'}
          </button>
          <label className="text-sm">
            <span className="text-gray-700">Filter Dataset</span>
            <select className="mt-1 border rounded p-2 ml-2" value={datasetFilter} onChange={(e) => setDatasetFilter(e.target.value)} aria-label="dataset-filter">
              {allDatasets.map((d) => <option key={d} value={d}>{d || 'All'}</option>)}
            </select>
          </label>
          <label className="text-sm">
            <span className="text-gray-700">Filter Metric</span>
            <select className="mt-1 border rounded p-2 ml-2" value={metricFilter} onChange={(e) => setMetricFilter(e.target.value)} aria-label="metric-filter">
              {allMetrics.map((m) => <option key={m} value={m}>{m || 'All'}</option>)}
            </select>
          </label>
        </div>
      </div>

      {error && <div className="text-red-600 mb-3">{error}</div>}

      {data && (
        <>
          <div className="p-4 bg-white border rounded mb-4">
            <div className="text-sm text-gray-700">Overall delta</div>
            <div className="text-lg">Baseline {fmt(data.summary.overall.baseline)} → Current {fmt(data.summary.overall.current)} (<span className={data.summary.overall.delta >= 0 ? 'text-green-700' : 'text-red-700'}>{fmt(data.summary.overall.delta)}</span>)</div>
            <div className="text-xs text-gray-500 mt-1">Common conversations: {data.summary.counts.conversations_common} · Baseline-only: {data.summary.counts.baseline_only} · Current-only: {data.summary.counts.current_only}</div>
          </div>

          <div className="p-4 bg-white border rounded mb-4 overflow-auto" data-testid="per-dataset-table">
            <h3 className="font-semibold mb-2">Per Dataset</h3>
            <table className="min-w-full text-sm">
              <thead>
                <tr className="text-left border-b">
                  <th className="p-2">Dataset</th>
                  <th className="p-2">Baseline</th>
                  <th className="p-2">Current</th>
                  <th className="p-2">Delta</th>
                </tr>
              </thead>
              <tbody>
                {perDatasetFiltered.map((r: any) => (
                  <tr key={r.dataset_id} className="border-b">
                    <td className="p-2">{r.dataset_id}</td>
                    <td className="p-2">{fmt(r.baseline)}</td>
                    <td className="p-2">{fmt(r.current)}</td>
                    <td className={`p-2 ${r.delta >= 0 ? 'text-green-700' : 'text-red-700'}`}>{fmt(r.delta)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="p-4 bg-white border rounded mb-4 overflow-auto" data-testid="metrics-table">
            <h3 className="font-semibold mb-2">Metrics by Dataset</h3>
            <table className="min-w-full text-sm">
              <thead>
                <tr className="text-left border-b">
                  <th className="p-2">Dataset</th>
                  <th className="p-2">Metric</th>
                  <th className="p-2">Baseline</th>
                  <th className="p-2">Current</th>
                  <th className="p-2">Delta</th>
                </tr>
              </thead>
              <tbody>
                {metricsFiltered.map((r: any, i: number) => (
                  <tr key={`${r.dataset_id}|${r.metric}|${i}`} className="border-b">
                    <td className="p-2">{r.dataset_id}</td>
                    <td className="p-2">{r.metric}</td>
                    <td className="p-2">{fmt(r.baseline)}</td>
                    <td className="p-2">{fmt(r.current)}</td>
                    <td className={`p-2 ${r.delta >= 0 ? 'text-green-700' : 'text-red-700'}`}>{fmt(r.delta)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  )
}
