import React, { useEffect, useMemo, useState } from 'react'
import Card from '../components/Card'
import Button from '../components/Button'
import { Checkbox, Select } from '../components/Form'
import { Input } from '../components/Form'
import { useVertical } from '../context/VerticalContext'

type Pair = {
  domain: string
  behavior: string
  raw_total: number
  final_total: number
  breakdown: { name: string, type: string, removed_exclude: number, removed_cap: number }[]
}

export default function CoverageGeneratorPage() {
  const { vertical } = useVertical()
  const [domains, setDomains] = useState<string[]>([])
  const [behaviors, setBehaviors] = useState<string[]>([])
  const [selectedDomains, setSelectedDomains] = useState<string[]>([])
  const [selectedBehaviors, setSelectedBehaviors] = useState<string[]>([])
  const [manifestPairs, setManifestPairs] = useState<Pair[]>([])
  const [combined, setCombined] = useState(true)
  const [save, setSave] = useState(false)
  const [overwrite, setOverwrite] = useState(false)
  const [busy, setBusy] = useState(false)
  const [msg, setMsg] = useState<string| null>(null)
  const [err, setErr] = useState<string| null>(null)
  const [regenBusy, setRegenBusy] = useState(false)
  const [regenMsg, setRegenMsg] = useState<string | null>(null)
  const [covSettings, setCovSettings] = useState<any | null>(null)
  const [saveCovBusy, setSaveCovBusy] = useState(false)
  const [saveCovMsg, setSaveCovMsg] = useState<string | null>(null)

  useEffect(() => {
    const load = async () => {
      // Use v2 taxonomy first (matches v2-only backend); fallback to legacy if needed.
      let r = await fetch('/coverage/taxonomy_v2')
      if (!r.ok) r = await fetch('/coverage/taxonomy')
      const js = await r.json()
      setDomains(js.domains || [])
      setBehaviors(js.behaviors || [])
      // Reset selections to avoid stale labels from a different taxonomy source
      setSelectedDomains([])
      setSelectedBehaviors([])
      try {
        const cs = await fetch('/coverage/settings')
        if (cs.ok) setCovSettings(await cs.json())
      } catch {}
    }
    load()
  }, [])

  const loadManifest = async () => {
    setBusy(true); setErr(null)
    try {
      const params = new URLSearchParams()
      if (selectedDomains.length) params.set('domains', selectedDomains.join(','))
      if (selectedBehaviors.length) params.set('behaviors', selectedBehaviors.join(','))
      // Use v2 manifest to match v2 taxonomy values; fallback to legacy
      let r = await fetch('/coverage/manifest_v2' + (params.toString() ? `?${params.toString()}` : ''))
      if (!r.ok) {
        r = await fetch('/coverage/manifest' + (params.toString() ? `?${params.toString()}` : ''))
      }
      const js = await r.json()
      setManifestPairs(js.pairs || [])
    } catch (e:any) {
      setErr(e.message || 'Failed to load manifest')
    } finally {
      setBusy(false)
    }
  }

  const triggerGenerate = async () => {
    setBusy(true); setErr(null); setMsg(null)
    try {
      const body = {
        combined,
        dry_run: !save,
        save,
        overwrite,
        domains: selectedDomains.length ? selectedDomains : undefined,
        behaviors: selectedBehaviors.length ? selectedBehaviors : undefined,
        version: '1.0.0',
        vertical,
      }
      const r = await fetch('/coverage/generate', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) })
      let js: any = null
      const text = await r.text()
      try { js = JSON.parse(text) } catch { /* not JSON */ }
      if (!r.ok) throw new Error(js?.detail || text || 'Generation failed')
      if (js.saved) setMsg(`Saved ${js.files?.length || 0} dataset(s) to server (vertical: ${vertical})`)
      else setMsg(`Generated ${js.outputs?.length || 0} dataset(s) (dry run)`) 
    } catch (e:any) {
      setErr(e.message || 'Generation failed')
    } finally {
      setBusy(false)
    }
  }

  // --- Regenerate (optimized) for an existing combined dataset id ---
  const [existingDatasetId, setExistingDatasetId] = useState('')
  const slugify = (s: string) => s.toLowerCase().trim().replace(/[^a-z0-9]+/g, '-').replace(/-+/g, '-').replace(/^-|-$/g, '')
  const parseCoverageCombined = (id: string) => {
    if (!id || !id.startsWith('coverage-')) return null
    const parts = id.split('-')
    if (parts.length < 4) return null
    const last = parts[parts.length - 1]
    const penultimate = parts[parts.length - 2]
    if (penultimate !== 'combined') return null
    const version = last
    const domainSlug = parts.slice(1, parts.length - 2).join('-')
    if (!domainSlug) return null
    return { domainSlug, version }
  }
  const regenerateOptimized = async () => {
    setRegenMsg(null)
    if (!existingDatasetId) { setRegenMsg('Enter an existing combined dataset id.'); return }
    const parsed = parseCoverageCombined(existingDatasetId)
    if (!parsed) { setRegenMsg('Only coverage-<domain>-combined-<version> dataset ids are supported.'); return }
    setRegenBusy(true)
    try {
      const t = await fetch('/coverage/taxonomy_v2')
      if (!t.ok) throw new Error(`Taxonomy HTTP ${t.status}`)
      const tj = await t.json()
      const domains: string[] = Array.isArray(tj.domains) ? tj.domains : []
      const match = domains.find(d => slugify(d) === parsed.domainSlug)
      if (!match) throw new Error('Domain not found in taxonomy for dataset id.')
      const body = { combined: true, dry_run: false, save: true, overwrite: true, domains: [match], version: parsed.version, vertical }
      const r = await fetch('/coverage/generate', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) })
      const text = await r.text()
      let js: any = null
      try { js = JSON.parse(text) } catch {}
      if (!r.ok) throw new Error(js?.detail || text || `Generate HTTP ${r.status}`)
      setRegenMsg('Regenerated with optimized coverage (pairwise). Files overwritten.')
    } catch (e:any) {
      setRegenMsg(e.message || 'Failed to regenerate')
    } finally {
      setRegenBusy(false)
    }
  }

  return (
    <div className="grid gap-4">
      <Card title="Dataset Generation Strategy (Server)">
        {covSettings ? (
          <div className="grid sm:grid-cols-3 gap-3 text-sm">
            <label className="flex items-center gap-2"><span className="w-32">Mode</span>
              <Select className="grow" value={covSettings.mode || 'pairwise'} onChange={e => setCovSettings((x:any)=>({...x, mode: e.target.value}))}>
                <option value="pairwise">pairwise</option>
                <option value="exhaustive">exhaustive</option>
              </Select>
            </label>
            <label className="flex items-center gap-2"><span className="w-32">t</span>
              <Input type="number" min={2} className="w-28" value={covSettings.t || 2} onChange={e => setCovSettings((x:any)=>({...x, t: Number(e.target.value)}))} />
            </label>
            <label className="flex items-center gap-2"><span className="w-32">Budget</span>
              <Input type="number" min={1} className="w-28" value={covSettings.per_behavior_budget || 120} onChange={e => setCovSettings((x:any)=>({...x, per_behavior_budget: Number(e.target.value)}))} />
            </label>
            <label className="flex items-center gap-2"><span className="w-32">Seed</span>
              <Input type="number" className="w-28" value={(covSettings.sampler?.rng_seed) ?? 42} onChange={e => setCovSettings((x:any)=>({
                ...x, sampler: { ...(x.sampler||{}), rng_seed: Number(e.target.value) }
              }))} />
            </label>
            <label className="flex items-center gap-2"><span className="w-32">Per-behavior</span>
              <Input type="number" min={1} className="w-28" value={(covSettings.sampler?.per_behavior_total) ?? 100} onChange={e => setCovSettings((x:any)=>({
                ...x, sampler: { ...(x.sampler||{}), per_behavior_total: Number(e.target.value) }
              }))} />
            </label>
            <label className="flex items-center gap-2"><span className="w-32">Min/domain</span>
              <Input type="number" min={0} className="w-28" value={(covSettings.sampler?.min_per_domain) ?? 3} onChange={e => setCovSettings((x:any)=>({
                ...x, sampler: { ...(x.sampler||{}), min_per_domain: Number(e.target.value) }
              }))} />
            </label>
            <div className="col-span-full flex items-center gap-2">
              <Button onClick={async ()=>{
                setSaveCovMsg(null); setSaveCovBusy(true)
                try {
                  const r = await fetch('/coverage/settings', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(covSettings) })
                  const js = await r.json().catch(()=>({}))
                  if (!r.ok) throw new Error(js?.detail || 'Save failed')
                  setSaveCovMsg('Saved. New generations will use these settings.')
                } catch(e:any) {
                  setSaveCovMsg(e.message || 'Save failed')
                } finally { setSaveCovBusy(false) }
              }} disabled={saveCovBusy}>{saveCovBusy?'Saving…':'Save'}</Button>
              {saveCovMsg && <span className="text-xs">{saveCovMsg}</span>}
            </div>
          </div>
        ) : (
          <div className="text-sm text-gray-600">Loading…</div>
        )}
      </Card>
      <Card title="Dataset Generator">
        <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-3 text-sm">
          <label className="flex items-center gap-2"><span className="w-28">Domains</span>
            <Select multiple value={selectedDomains} onChange={e => setSelectedDomains(Array.from(e.target.selectedOptions).map(o => o.value))} className="grow min-h-28">
              {domains.map(d => <option key={d} value={d}>{d}</option>)}
            </Select>
          </label>
          <label className="flex items-center gap-2"><span className="w-28">Behaviors</span>
            <Select multiple value={selectedBehaviors} onChange={e => setSelectedBehaviors(Array.from(e.target.selectedOptions).map(o => o.value))} className="grow min-h-28">
              {behaviors.map(b => <option key={b} value={b}>{b}</option>)}
            </Select>
          </label>
          <div className="flex flex-col gap-2">
            <label className="inline-flex items-center gap-2"><Checkbox checked={combined} onChange={e => setCombined((e.target as HTMLInputElement).checked)} /> Combined (per-domain + global)</label>
            <label className="inline-flex items-center gap-2"><Checkbox checked={save} onChange={e => setSave((e.target as HTMLInputElement).checked)} /> Save to server</label>
            <label className="inline-flex items-center gap-2"><Checkbox checked={overwrite} onChange={e => setOverwrite((e.target as HTMLInputElement).checked)} disabled={!save} /> Overwrite</label>
            <div className="flex gap-2 mt-1">
              <Button variant="primary" onClick={loadManifest} disabled={busy}>Preview coverage</Button>
              <Button variant="success" onClick={triggerGenerate} disabled={busy}>Generate</Button>
              <a className="rounded-md px-3 py-2 bg-primary text-white text-sm" href={`/coverage/report.csv?type=summary${selectedDomains.length?`&domains=${encodeURIComponent(selectedDomains.join(','))}`:''}${selectedBehaviors.length?`&behaviors=${encodeURIComponent(selectedBehaviors.join(','))}`:''}`} download>
                Download Summary CSV
              </a>
              <a className="rounded-md px-3 py-2 bg-success text-white text-sm" href={`/coverage/report.csv?type=heatmap${selectedDomains.length?`&domains=${encodeURIComponent(selectedDomains.join(','))}`:''}${selectedBehaviors.length?`&behaviors=${encodeURIComponent(selectedBehaviors.join(','))}`:''}`} download>
                Download Heatmap CSV
              </a>
            </div>
            {msg && <div className="text-success">{msg}</div>}
            {err && <div className="text-danger">{err}</div>}
            <div className="mt-4 border-t pt-3">
              <div className="font-medium mb-1">Regenerate (optimized)</div>
              <div className="flex items-center gap-2">
                <input className="border rounded px-2 py-1 grow" placeholder="coverage-<domain>-combined-<version>" value={existingDatasetId} onChange={e => setExistingDatasetId(e.target.value)} />
                <Button onClick={regenerateOptimized} disabled={regenBusy}>{regenBusy ? 'Regenerating…' : 'Run'}</Button>
              </div>
              <div className="text-xs text-gray-600 mt-1">Overwrites the specified combined dataset using current optimization settings (configs/coverage.json)</div>
              {regenMsg && <div className="text-xs mt-1">{regenMsg}</div>}
            </div>
          </div>
        </div>
      </Card>

      <Card title="Dataset Coverage Preview">
        {!manifestPairs.length && <div className="text-sm text-gray-700">No selection yet. Click Preview coverage.</div>}
        {!!manifestPairs.length && (
          <div className="space-y-4">
            {manifestPairs.map(p => (
              <div key={`${p.domain}-${p.behavior}`} className="rounded border border-gray-200 p-3">
                <div className="flex items-center gap-2 text-sm">
                  <span className="font-semibold">{p.domain}</span>
                  <span className="text-gray-500">/ {p.behavior}</span>
                  <div className="grow" />
                  <span className="text-xs">Raw: {p.raw_total} | Final: {p.final_total}</span>
                  <a className="ml-2 text-xs underline text-primary" href={`/coverage/report.csv?type=summary&domains=${encodeURIComponent(p.domain)}&behaviors=${encodeURIComponent(p.behavior)}`} download>
                    CSV
                  </a>
                </div>
                <div className="mt-2 text-xs text-gray-700">
                  {p.breakdown.map(b => (
                    <div key={b.name} className="flex gap-2">
                      <span className="w-56 truncate" title={b.name}>{b.name}</span>
                      <span className="text-gray-500">({b.type})</span>
                      <span className="ml-auto">-exclude: {b.removed_exclude}, -cap: {b.removed_cap}</span>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        )}
      </Card>
    </div>
  )
}
