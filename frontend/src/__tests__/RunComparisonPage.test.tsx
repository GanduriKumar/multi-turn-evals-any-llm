import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor, fireEvent, within } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import RunComparisonPage from '../pages/RunComparisonPage'

const mockCompare = {
  baseline_run_id: 'runA',
  current_run_id: 'runB',
  summary: {
    overall: { baseline: 0.7, current: 0.8, delta: 0.1 },
    counts: { conversations_common: 10, baseline_only: 2, current_only: 1 },
  },
  per_dataset: [
    { dataset_id: 'ds1', baseline: 0.6, current: 0.7, delta: 0.1 },
    { dataset_id: 'ds2', baseline: 0.8, current: 0.75, delta: -0.05 },
  ],
  metrics_by_dataset: [
    { dataset_id: 'ds1', metric: 'correctness', baseline: 0.55, current: 0.65, delta: 0.1 },
    { dataset_id: 'ds1', metric: 'consistency', baseline: 0.7, current: 0.72, delta: 0.02 },
    { dataset_id: 'ds2', metric: 'correctness', baseline: 0.85, current: 0.8, delta: -0.05 },
  ],
  per_conversation: [],
}

function setup() {
  // mock fetch for compare
  ;(globalThis.fetch as any) = vi.fn(async (url: string) => {
    if (url.startsWith('/api/v1/runs/compare?')) {
      return { ok: true, json: async () => mockCompare } as any
    }
    throw new Error('unexpected fetch: ' + url)
  })

  render(
    <MemoryRouter initialEntries={[`/compare?baseline=runA&current=runB`]}>
      <Routes>
        <Route path="/compare" element={<RunComparisonPage />} />
      </Routes>
    </MemoryRouter>
  )
}

describe('RunComparisonPage', () => {
  beforeEach(() => {
    vi.restoreAllMocks()
  })

  it('displays overall and per-dataset/metrics tables', async () => {
    setup()

    // Click compare
    fireEvent.click(screen.getByText('Compare'))

    await waitFor(() => {
      expect(screen.getByText(/Overall delta/)).toBeInTheDocument()
      expect(screen.getByText('Per Dataset')).toBeInTheDocument()
      expect(screen.getByText('Metrics by Dataset')).toBeInTheDocument()
    })

    // Check a couple of cells in the tables (avoid select options)
    const perDs = screen.getByTestId('per-dataset-table')
    expect(within(perDs).getByText('ds1')).toBeInTheDocument()
    const metrics = screen.getByTestId('metrics-table')
    expect(within(metrics).getAllByText('correctness')[0]).toBeInTheDocument()
  })

  it('filters datasets and metrics', async () => {
    setup()

    // Trigger load
    fireEvent.click(screen.getByText('Compare'))
    await waitFor(() => screen.getByText('Per Dataset'))

    const dsSel = screen.getByLabelText('dataset-filter') as HTMLSelectElement
    const metricSel = screen.getByLabelText('metric-filter') as HTMLSelectElement

    // Filter to ds1 only
    fireEvent.change(dsSel, { target: { value: 'ds1' } })
    // After filter, ds2 row should not be present in per dataset table
    const perDs = screen.getByTestId('per-dataset-table')
    expect(within(perDs).queryByText('ds2')).toBeNull()

    // Filter metrics to correctness only
    fireEvent.change(metricSel, { target: { value: 'correctness' } })
    // We should still see correctness entries but not consistency rows in metrics table
    const mt = screen.getByTestId('metrics-table')
    expect(within(mt).queryByText('consistency')).toBeNull()
  })
})
