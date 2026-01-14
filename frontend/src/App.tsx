import React, { useEffect, useState } from 'react'
import { BrowserRouter, Link, NavLink, Route, Routes } from 'react-router-dom'
import DatasetsPage from './pages/Datasets'
import RunsPage from './pages/Runs'
import ReportsPage from './pages/Reports'
import SettingsPage from './pages/Settings'
import GoldenGeneratorPage from './pages/GoldenGenerator'
import MetricsPage from './pages/Metrics'
// Golden Editor disabled per scope
import CoverageGeneratorPage from './pages/CoverageGenerator'
import Card from './components/Card'
import { VerticalProvider, useVertical } from './context/VerticalContext'
import Button from './components/Button'

function VerticalSelector() {
  const { vertical, supported, setVertical, loading } = useVertical()
  return (
    <div className="flex items-center gap-2">
      <span className="rounded-md px-2 py-1 font-medium bg-primary text-white hover:opacity-90 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/60 text-sm">Vertical</span>
      <select
        className="border rounded px-2 py-1 text-sm"
        value={vertical}
        onChange={e => setVertical(e.target.value)}
        disabled={loading}
        title="Vertical"
      >
        {supported.map(v => <option key={v} value={v}>{v}</option>)}
      </select>
    </div>
  )
}

function Layout({ children }: { children: React.ReactNode }) {
  return (
    <div className="min-h-full">
      <header className="sticky top-0 z-10 bg-white/80 backdrop-blur border-b border-gray-200">
        <div className="mx-auto max-w-6xl px-4 h-14 flex items-center justify-between">
          <Link to="/" className="flex items-center gap-2">
            <div className="flex flex-col leading-tight">
              <span className="font-extrabold text-4xl text-orange-600">LLM Evals</span>
            </div>
          </Link>
          <nav className="flex items-center gap-4 text-sm">
            {(() => {
              type Color = 'primary' | 'success' | 'warning' | 'danger'
              const links: Array<{to: string, label: string, color: Color}> = [
                // Move Dataset Generator to the first position
                { to: '/coverage', label: 'Dataset Generator', color: 'warning' },
                { to: '/datasets', label: 'Datasets Viewer', color: 'success' },
                { to: '/metrics', label: 'Metrics', color: 'primary' },
                { to: '/settings', label: 'LLM Settings', color: 'danger' },
                { to: '/runs', label: 'Runs', color: 'primary' },
                { to: '/reports', label: 'Reports', color: 'warning' },
                // Golden Generator deprecated/hidden per scope
              ]
              const cls = (color: Color, _isActive: boolean) => {
                const base = 'rounded-md px-2 py-1 font-medium focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-offset-2 transition'
                switch (color) {
                  case 'success':
                    return `${base} bg-success text-white hover:opacity-90 focus-visible:ring-success/60`
                  case 'warning':
                    return `${base} bg-warning text-white hover:opacity-90 focus-visible:ring-warning/60`
                  case 'danger':
                    return `${base} bg-danger text-white hover:opacity-90 focus-visible:ring-danger/60`
                  default:
                    return `${base} bg-primary text-white hover:opacity-90 focus-visible:ring-primary/60`
                }
              }
              return links.map(link => (
                <NavLink key={link.to} to={link.to} className={({isActive}) => cls(link.color, isActive)}>
                  {link.label}
                </NavLink>
              ))
            })()}
          </nav>
          <VerticalSelector />
        </div>
      </header>
      <main className="mx-auto max-w-6xl px-4 py-6">
        {children}
      </main>
    </div>
  )
}

function HomeCards() {
  const [status, setStatus] = useState<{ ok: boolean; version?: string; vertical?: string; ollama_host?: string; models?: Record<string, string>; gemini?: boolean; openai?: boolean; embed_model?: string; errors: string[] }>({ ok: false, errors: [] })
  const { vertical } = useVertical()

  useEffect(() => {
    const load = async () => {
      const errs: string[] = []
      let ok = false
      let version: any = null
      try {
        const h = await fetch('/health')
        ok = h.ok
      } catch (e: any) {
        errs.push('Backend not reachable at /health')
      }
      try {
        const v = await fetch('/version')
        if (v.ok) version = await v.json()
      } catch (e: any) {
        errs.push('Failed to load /version')
      }
      try {
        const s = await fetch('/settings')
        if (s.ok) {
          const js = await s.json()
          setStatus(prev => ({
            ...prev,
            ok,
            version: version?.version,
            vertical,
            ollama_host: js?.ollama_host,
            models: js?.models,
            gemini: js?.gemini_enabled,
            openai: js?.openai_enabled,
            embed_model: js?.embed_model,
            errors: errs,
          }))
          return
        }
      } catch (e: any) {
        errs.push('Failed to load /settings')
      }
      setStatus(prev => ({ ...prev, ok, version: version?.version, vertical, errors: errs }))
    }
    load()
  }, [vertical])

  return (
    <div className="grid grid-cols-2 gap-4 items-start">
      <div className="w-full">
        <Card title="Quick Start" className="h-full" borderless>
          <div className="text-base text-gray-800 space-y-3">
            <ol className="list-decimal pl-5 space-y-1">
            <li>
              Configure models and thresholds
              <div className="mt-2">
                <Link to="/settings"><Button variant="primary" className="bg-[#4285F4] text-white">Open Settings</Button></Link>
              </div>
            </li>
            <li>
              Prepare or generate a dataset for the "{vertical}" vertical
              <div className="mt-2 flex gap-2 flex-wrap">
                <Link to="/coverage"><Button variant="warning" className="bg-[#F4B400] text-black">Dataset Generator</Button></Link>
                <Link to="/datasets"><Button variant="secondary" className="bg-[#0F9D58] text-white">Datasets</Button></Link>
              </div>
            </li>
            <li>
              Choose metrics, then run and review
              <div className="mt-2 flex gap-2 flex-wrap">
                <Link to="/metrics"><Button variant="secondary" className="bg-[#4285F4] text-white">Metrics</Button></Link>
                <Link to="/runs"><Button variant="success" className="bg-[#DB4437] text-white">Runs</Button></Link>
                <Link to="/reports"><Button variant="ghost" className="bg-[#F4B400] text-white">Reports</Button></Link>
              </div>
            </li>
            </ol>
          </div>
        </Card>
      </div>
      <div className="w-full">
        <Card title="Health" className="h-full" borderless>
          <div className="text-base text-gray-800 space-y-2">
          <div>
            Backend: {status.ok ? <span className="text-green-700 font-medium">OK</span> : <span className="text-red-700 font-medium">Unavailable</span>} {status.version ? <span className="text-gray-500">v{status.version}</span> : null}
          </div>
          <div>Vertical: <span className="font-medium">{status.vertical || vertical}</span></div>
          {status.ollama_host ? <div>Ollama host: <span className="font-mono">{status.ollama_host}</span></div> : null}
          {status.embed_model ? <div>Embed model: <span className="font-mono">{status.embed_model}</span></div> : null}
          {status.models ? (
            <div className="text-xs text-gray-600">Defaults → ollama: <span className="font-mono">{status.models.ollama}</span>, gemini: <span className="font-mono">{status.models.gemini}</span>, openai: <span className="font-mono">{status.models.openai}</span></div>
          ) : null}
          <div className="text-xs text-gray-600">
            Providers: Gemini {status.gemini ? '✓' : '×'} · OpenAI {status.openai ? '✓' : '×'}
          </div>
          {status.errors.length > 0 && (
            <ul className="list-disc pl-5 text-xs text-red-700 space-y-1">
              {status.errors.map((e, i) => <li key={i}>{e}</li>)}
            </ul>
          )}
          </div>
        </Card>
      </div>
    </div>
  )
}

function Placeholder({ title }: { title: string }) {
  return (
    <div className="grid gap-4">
      <Card title={title}>
        <div className="text-sm text-gray-700">Coming soon…</div>
      </Card>
    </div>
  )
}

export default function App() {
  return (
    <BrowserRouter>
      <VerticalProvider>
        <Layout>
          <Routes>
            <Route path="/" element={<HomeCards/>} />
            <Route path="/datasets" element={<DatasetsPage />} />
            <Route path="/runs" element={<RunsPage />} />
            <Route path="/reports" element={<ReportsPage />} />
            <Route path="/settings" element={<SettingsPage />} />
            <Route path="/metrics" element={<MetricsPage />} />
            <Route path="/coverage" element={<CoverageGeneratorPage />} />
          </Routes>
        </Layout>
      </VerticalProvider>
    </BrowserRouter>
  )
}
