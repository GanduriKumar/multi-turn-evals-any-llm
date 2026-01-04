import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor, fireEvent } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import RunDashboardPage from '../pages/RunDashboardPage'
import MetricsBreakdownPage from '../pages/MetricsBreakdownPage'

const progressPayload = {
  run_id: 'run-xyz',
  overall_status: 'completed',
  conversations: [],
  events: [] as any[],
}

function mockSuccessfulArtifactOnce(name: string) {
  const blob = new Blob([name === 'csv' ? 'a,b\n1,2' : '<html></html>'], { type: name === 'csv' ? 'text/csv' : name === 'markdown' ? 'text/markdown' : 'text/html' })
  ;(globalThis.fetch as any) = vi.fn(async (url: string) => {
    if (url.includes('/progress')) {
      return { ok: true, json: async () => progressPayload } as any
    }
    if (url.includes('/api/v1/results/')) {
      return { ok: true, json: async () => ({ results: [], run: {} }) } as any
    }
    if (url.includes('/artifacts')) {
      return {
        ok: true,
        headers: { get: (k: string) => (k.toLowerCase() === 'content-disposition' ? `attachment; filename="file.${name === 'markdown' ? 'md' : name}"` : null) },
        blob: async () => blob,
      } as any
    }
    throw new Error('unexpected ' + url)
  })
}

function mockFailedArtifactOnce() {
  ;(globalThis.fetch as any) = vi.fn(async (url: string) => {
    if (url.includes('/progress')) {
      return { ok: true, json: async () => progressPayload } as any
    }
    if (url.includes('/artifacts')) {
      return { ok: false, status: 404, statusText: 'Not Found', text: async () => 'missing' } as any
    }
    throw new Error('unexpected ' + url)
  })
}

// JSDOM: intercept anchor clicks created by downloader
function spyAnchor() {
  return vi.spyOn(document.body, 'appendChild')
}

describe('Artifact downloads', () => {
  beforeEach(() => {
    vi.restoreAllMocks()
  })

  it('downloads from dashboard and shows filename', async () => {
    mockSuccessfulArtifactOnce('json')
    render(
      <MemoryRouter initialEntries={[`/dashboard/run-xyz`]}>
        <Routes>
          <Route path="/dashboard/:runId" element={<RunDashboardPage />} />
        </Routes>
      </MemoryRouter>
    )

    const anchorSpy = spyAnchor()
    await waitFor(() => screen.getByTestId('dl-results'))
    fireEvent.click(screen.getByTestId('dl-results'))
    await waitFor(() => expect(screen.getByText(/Saved/)).toBeInTheDocument())
    anchorSpy.mockRestore()
  })

  it('shows error on failure from dashboard', async () => {
    mockFailedArtifactOnce()
    render(
      <MemoryRouter initialEntries={[`/dashboard/run-xyz`]}>
        <Routes>
          <Route path="/dashboard/:runId" element={<RunDashboardPage />} />
        </Routes>
      </MemoryRouter>
    )

    await waitFor(() => screen.getByTestId('dl-results'))
    fireEvent.click(screen.getByTestId('dl-results'))
    await waitFor(() => expect(screen.getByRole('alert')).toBeInTheDocument())
  })

  it('downloads from metrics page (html/csv/markdown)', async () => {
    mockSuccessfulArtifactOnce('csv')
    render(
      <MemoryRouter initialEntries={[`/metrics/run-xyz`]}>
        <Routes>
          <Route path="/metrics/:runId" element={<MetricsBreakdownPage />} />
        </Routes>
      </MemoryRouter>
    )
    // Ensure buttons exist and click one
    await waitFor(() => screen.getByTestId('dl-csv'))
    const anchorSpy = spyAnchor()
    fireEvent.click(screen.getByTestId('dl-csv'))
    await waitFor(() => expect(screen.getByText(/Saved/)).toBeInTheDocument())
    anchorSpy.mockRestore()
  })
})
