import React from 'react'

function Card({ title, children }: { title: string, children: React.ReactNode }) {
  return (
    <div className="rounded-lg border border-gray-200 bg-white shadow-sm">
      <div className="border-b border-gray-100 px-4 py-2 font-medium text-gray-800">{title}</div>
      <div className="p-4">{children}</div>
    </div>
  )
}

export default function App() {
  return (
    <div className="min-h-full">
      <header className="sticky top-0 z-10 bg-white/80 backdrop-blur border-b border-gray-200">
        <div className="mx-auto max-w-6xl px-4 h-14 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-2.5 h-2.5 rounded-full bg-primary" />
            <span className="font-semibold">LLM Evals</span>
          </div>
          <nav className="flex items-center gap-4 text-sm">
            <a className="hover:text-primary" href="#">Datasets</a>
            <a className="hover:text-primary" href="#">Runs</a>
            <a className="hover:text-primary" href="#">Reports</a>
            <a className="hover:text-primary" href="#">Settings</a>
            <a className="hover:text-primary" href="#">Golden Editor</a>
            <a className="hover:text-primary" href="#">Metrics</a>
            <a className="hover:text-primary" href="#">Golden Generator</a>
          </nav>
        </div>
      </header>
      <main className="mx-auto max-w-6xl px-4 py-6 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
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
      </main>
    </div>
  )
}
