import React from 'react'
import { BrowserRouter, Link, NavLink, Route, Routes } from 'react-router-dom'
import DatasetsPage from './pages/Datasets'
import RunsPage from './pages/Runs'
import ReportsPage from './pages/Reports'
import SettingsPage from './pages/Settings'
import GoldenGeneratorPage from './pages/GoldenGenerator'
import MetricsPage from './pages/Metrics'
import GoldenEditorPage from './pages/GoldenEditor'
import Card from './components/Card'

function Layout({ children }: { children: React.ReactNode }) {
  return (
    <div className="min-h-full">
      <header className="sticky top-0 z-10 bg-white/80 backdrop-blur border-b border-gray-200">
        <div className="mx-auto max-w-6xl px-4 h-14 flex items-center justify-between">
          <Link to="/" className="flex items-center gap-2">
            <span className="font-extrabold text-4xl text-orange-600">LLM Evals</span>
          </Link>
          <nav className="flex items-center gap-4 text-sm">
            {(() => {
              type Color = 'primary' | 'success' | 'warning' | 'danger'
              const links: Array<{to: string, label: string, color: Color}> = [
                { to: '/datasets', label: 'Datasets', color: 'success' },
                { to: '/runs', label: 'Runs', color: 'primary' },
                { to: '/reports', label: 'Reports', color: 'warning' },
                { to: '/settings', label: 'Settings', color: 'danger' },
                { to: '/golden-editor', label: 'Golden Editor', color: 'success' },
                { to: '/metrics', label: 'Metrics', color: 'primary' },
                { to: '/golden-generator', label: 'Golden Generator', color: 'primary' },
              ]
              const cls = (color: Color, _isActive: boolean) => {
                const base = 'rounded-md px-2 py-1 font-medium focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-offset-2 transition'
                switch (color) {
                  case 'success':
                    return `${base} bg-success text-white hover:opacity-90 focus-visible:ring-success/60`
                  case 'warning':
                    return `${base} bg-warning text-black hover:opacity-90 focus-visible:ring-warning/60`
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
        </div>
      </header>
      <main className="mx-auto max-w-6xl px-4 py-6">
        {children}
      </main>
    </div>
  )
}

function HomeCards() {
  return (
    <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
      <Card title="Quick Start">
        <ul className="list-disc pl-5 text-sm text-gray-700 space-y-1">
          <li>Upload a dataset and golden set</li>
          <li>Pick a model and metrics</li>
          <li>Run and review results</li>
        </ul>
      </Card>
      <Card title="Status">
        <div className="text-sm text-gray-700">Backend: configure at backend/app.py</div>
      </Card>
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
      <Layout>
        <Routes>
          <Route path="/" element={<HomeCards/>} />
          <Route path="/datasets" element={<DatasetsPage />} />
          <Route path="/runs" element={<RunsPage />} />
          <Route path="/reports" element={<ReportsPage />} />
          <Route path="/settings" element={<SettingsPage />} />
          <Route path="/golden-editor" element={<GoldenEditorPage />} />
          <Route path="/metrics" element={<MetricsPage />} />
          <Route path="/golden-generator" element={<GoldenGeneratorPage />} />
        </Routes>
      </Layout>
    </BrowserRouter>
  )
}
