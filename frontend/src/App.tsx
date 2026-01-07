import React from 'react'
import { BrowserRouter, Link, NavLink, Route, Routes } from 'react-router-dom'
import DatasetsPage from './pages/Datasets'
import RunsPage from './pages/Runs'
import ReportsPage from './pages/Reports'
import SettingsPage from './pages/Settings'

function Card({ title, children }: { title: string, children: React.ReactNode }) {
  return (
    <div className="rounded-lg border border-gray-200 bg-white shadow-sm">
      <div className="border-b border-gray-100 px-4 py-2 font-medium text-gray-800">{title}</div>
      <div className="p-4">{children}</div>
    </div>
  )
}

function Layout({ children }: { children: React.ReactNode }) {
  return (
    <div className="min-h-full">
      <header className="sticky top-0 z-10 bg-white/80 backdrop-blur border-b border-gray-200">
        <div className="mx-auto max-w-6xl px-4 h-14 flex items-center justify-between">
          <Link to="/" className="flex items-center gap-2">
            <div className="w-2.5 h-2.5 rounded-full bg-primary" />
            <span className="font-semibold">LLM Evals</span>
          </Link>
          <nav className="flex items-center gap-4 text-sm">
            {[
              { to: '/datasets', label: 'Datasets' },
              { to: '/runs', label: 'Runs' },
              { to: '/reports', label: 'Reports' },
              { to: '/settings', label: 'Settings' },
              { to: '/golden-editor', label: 'Golden Editor' },
              { to: '/metrics', label: 'Metrics' },
              { to: '/golden-generator', label: 'Golden Generator' },
            ].map(link => (
              <NavLink key={link.to} to={link.to} className={({isActive}) => `hover:text-primary ${isActive ? 'text-primary' : ''}`}>
                {link.label}
              </NavLink>
            ))}
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
      <Card title="Brand Palette">
        <div className="flex gap-2">
          <div className="w-8 h-8 rounded bg-primary" title="primary" />
          <div className="w-8 h-8 rounded bg-success" title="success" />
          <div className="w-8 h-8 rounded bg-warning" title="warning" />
          <div className="w-8 h-8 rounded bg-danger" title="danger" />
        </div>
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
          <Route path="/golden-editor" element={<Placeholder title="Golden Editor" />} />
          <Route path="/metrics" element={<Placeholder title="Metrics" />} />
          <Route path="/golden-generator" element={<Placeholder title="Golden Generator" />} />
        </Routes>
      </Layout>
    </BrowserRouter>
  )
}
