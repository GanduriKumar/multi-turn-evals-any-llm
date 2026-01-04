import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import RunDashboardPage from '../pages/RunDashboardPage';

const progressPayload = {
  run_id: 'run-123',
  overall_status: 'running',
  conversations: [
    { dataset_id: 'conv001', conversation_id: 'c1', model: 'dummy', status: 'running', progress: 0.5 },
  ],
  events: [{ event: 'job_started', job_id: 'job-1' }],
};

describe('RunDashboardPage', () => {
  beforeEach(() => {
    (global.fetch as any) = vi.fn(async (url: string) => {
      if (url === '/api/v1/runs/run-123/progress') {
        return { ok: true, json: async () => progressPayload } as any;
      }
      if (url === '/api/v1/jobs/job-1/cancel') {
        return { ok: true, json: async () => ({ status: 'ok' }) } as any;
      }
      throw new Error('unexpected fetch ' + url);
    });
  });

  it('shows progress and artifact links', async () => {
    render(
      <MemoryRouter initialEntries={["/dashboard/run-123"]}>
        <Routes>
          <Route path="/dashboard/:runId" element={<RunDashboardPage />} />
        </Routes>
      </MemoryRouter>
    );

    await waitFor(() => expect(screen.getByText(/Run ID:/)).toBeInTheDocument());
    expect(screen.getByText('running')).toBeInTheDocument();
    expect(screen.getByText('c1')).toBeInTheDocument();
    // artifact button present
    expect(screen.getByTestId('dl-results')).toBeInTheDocument();
  });

  it('downloads artifact and shows progress/errors', async () => {
    // mock artifacts response streaming fallback
    (global.fetch as any) = vi.fn(async (url: string) => {
      if (url === '/api/v1/runs/run-123/progress') {
        return { ok: true, json: async () => progressPayload } as any;
      }
      if (url.startsWith('/api/v1/runs/run-123/artifacts')) {
        const blob = new Blob([JSON.stringify({ ok: true })], { type: 'application/json' });
        return {
          ok: true,
          headers: { get: (k: string) => (k.toLowerCase() === 'content-disposition' ? 'attachment; filename="results.json"' : null) },
          blob: async () => blob,
        } as any;
      }
      throw new Error('unexpected fetch ' + url);
    });

    render(
      <MemoryRouter initialEntries={["/dashboard/run-123"]}>
        <Routes>
          <Route path="/dashboard/:runId" element={<RunDashboardPage />} />
        </Routes>
      </MemoryRouter>
    );

    await waitFor(() => screen.getByTestId('dl-results'))
    // Spy on link click to avoid navigation
    const clickSpy = vi.spyOn(document.body, 'appendChild');
    fireEvent.click(screen.getByTestId('dl-results'))
    // eventually success filename shows
    await waitFor(() => expect(screen.getByText(/Saved results\.json/)).toBeInTheDocument())
    clickSpy.mockRestore()
  });

  it('shows error on download failure', async () => {
    (global.fetch as any) = vi.fn(async (url: string) => {
      if (url === '/api/v1/runs/run-123/progress') {
        return { ok: true, json: async () => progressPayload } as any;
      }
      if (url.startsWith('/api/v1/runs/run-123/artifacts')) {
        return { ok: false, status: 500, statusText: 'Server error', text: async () => 'boom' } as any;
      }
      throw new Error('unexpected fetch ' + url);
    });

    render(
      <MemoryRouter initialEntries={["/dashboard/run-123"]}>
        <Routes>
          <Route path="/dashboard/:runId" element={<RunDashboardPage />} />
        </Routes>
      </MemoryRouter>
    );

    await waitFor(() => screen.getByTestId('dl-results'))
    fireEvent.click(screen.getByTestId('dl-results'))
    await waitFor(() => expect(screen.getByRole('alert')).toHaveTextContent(/500/))
  });
});
