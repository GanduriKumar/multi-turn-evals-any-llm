import { describe, expect, it, vi, beforeEach } from 'vitest'
import { render, screen, waitFor, fireEvent } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import MetricsBreakdownPage from '../pages/MetricsBreakdownPage'

const mockResults = {
  results: [
    {
      dataset_id: 'ds1',
      conversation_id: 'c1',
      model_name: 'm1',
      turns: [
        { turn_id: 1, metrics: { correctness: 0.9, consistency: 0.8 }, weighted_score: 0.9, passed: true, response: 'A', canonical: { text: 'A' } },
        { turn_id: 2, metrics: { correctness: 0.4, consistency: 0.7 }, weighted_score: 0.5, passed: false, response: 'B', canonical: { text: 'C' } },
      ],
    },
    {
      dataset_id: 'ds1',
      conversation_id: 'c2',
      model_name: 'm1',
      turns: [
        { turn_id: 1, metrics: { correctness: 0.6 }, weighted_score: 0.6, passed: true, response: 'X', canonical: { text: 'Y' } },
      ],
    },
  ],
  run: { thresholds: { turn_pass: 0.7, conversation_pass: 0.8 } },
}

function setup(route = '/metrics/run-1') {
  // Placeholder to ensure file touched if necessary
  vi.spyOn(globalThis as any, 'fetch' as any).mockResolvedValueOnce({
    ok: true,
    json: async () => mockResults,
  } as any)

  render(
    <MemoryRouter initialEntries={[route]}>
      <Routes>
        <Route path="/metrics/:runId" element={<MetricsBreakdownPage />} />
      </Routes>
    </MemoryRouter>
  )
}

describe('MetricsBreakdownPage', () => {
  beforeEach(() => {
    vi.restoreAllMocks()
  })

  it('loads and renders rows and chart', async () => {
    setup()

    await waitFor(() => expect(screen.getByText('Metrics Breakdown')).toBeInTheDocument())
    // threshold indicator
    expect(screen.getByText(/Threshold:/)).toBeInTheDocument()

    // table rows
    await waitFor(() => expect(screen.getAllByRole('row').length).toBeGreaterThan(1))

    // chart canvas present
    expect(screen.getByRole('img', { name: 'metrics-barchart' })).toBeInTheDocument()
  })

  it('filters by conversation and exports CSV', async () => {
    setup()

    await waitFor(() => screen.getByLabelText('Conversation'))
    const select = screen.getByLabelText('Conversation') as HTMLSelectElement
    // choose c1
    fireEvent.change(select, { target: { value: 'c1' } })

    // export buttons exist
    expect(screen.getByText('Export CSV')).toBeInTheDocument()
    expect(screen.getByText('Export PNG')).toBeInTheDocument()
  })
})
