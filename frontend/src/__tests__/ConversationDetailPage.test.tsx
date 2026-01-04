import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import ConversationDetailPage from '../pages/ConversationDetailPage';

const payload = {
  run_id: 'run-1',
  conversation_id: 'c-1',
  results: [
    {
      dataset_id: 'conv001',
      conversation_id: 'c-1',
      model_name: 'dummy',
      aggregate: { score: 0.9, passed: true },
      thresholds: { turn_pass: 0.6, conversation_pass: 0.7 },
      turns: [
        {
          turn_id: 1,
          prompt: 'Hello',
          response: 'Hi there',
          canonical: { text: 'Hello there' },
          metrics: { correctness: 0.7 },
          weighted_score: 0.7,
          passed: true,
        },
        {
          turn_id: 2,
          prompt: 'Bye',
          response: 'Goodbye',
          canonical: { text: 'Goodbye' },
          metrics: { correctness: 1.0 },
          weighted_score: 1.0,
          passed: true,
        },
      ],
    },
  ],
};

describe('ConversationDetailPage', () => {
  beforeEach(() => {
    (globalThis.fetch as any) = vi.fn(async (url: string) => {
      if (url.startsWith('/api/v1/runs/run-1/conversations/c-1')) {
        return { ok: true, json: async () => payload } as any;
      }
      throw new Error('unexpected fetch');
    });
  });

  it('renders conversation turns and scores', async () => {
    render(
      <MemoryRouter initialEntries={["/conversation/run-1/c-1"]}>
        <Routes>
          <Route path="/conversation/:runId/:conversationId" element={<ConversationDetailPage />} />
        </Routes>
      </MemoryRouter>
    );

    await waitFor(() => expect(screen.getByText(/Run:/)).toBeInTheDocument());
    expect(screen.getByText(/Aggregate Score/)).toBeInTheDocument();

    // Verify turn blocks
    expect(screen.getByText('Turn 1')).toBeInTheDocument();
    expect(screen.getByText('Turn 2')).toBeInTheDocument();
  });

  it('shows diff when texts differ and not when equal', async () => {
    render(
      <MemoryRouter initialEntries={["/conversation/run-1/c-1"]}>
        <Routes>
          <Route path="/conversation/:runId/:conversationId" element={<ConversationDetailPage />} />
        </Routes>
      </MemoryRouter>
    );

    await waitFor(() => expect(screen.getByText('Turn 1')).toBeInTheDocument());

    // Expand diff sections by clicking their headers
    expect(screen.getByText(/\+\s*Hello there/)).toBeInTheDocument(); // expected added
    expect(screen.getByText(/-\s*Hi there/)).toBeInTheDocument(); // actual deletion

    // For equal turn 2, we do not render a Diff accordion
    expect(screen.getAllByText('Turn 2')[0]).toBeInTheDocument();
  });

  it('allows entering feedback and submits to backend, preserving unsaved state', async () => {
    // simple localStorage polyfill for test env
    const store: Record<string, string> = {}
    ;(globalThis as any).localStorage = {
      getItem: (k: string) => (k in store ? store[k] : null),
      setItem: (k: string, v: string) => { store[k] = String(v) },
      removeItem: (k: string) => { delete store[k] },
      clear: () => { Object.keys(store).forEach(k => delete store[k]) },
      key: (i: number) => Object.keys(store)[i] ?? null,
      get length() { return Object.keys(store).length }
    } as any
    // Mock fetch: first for conversation payload, then for feedback submission
    (globalThis.fetch as any) = vi.fn(async (url: string, init?: any) => {
      if (url.startsWith('/api/v1/runs/run-1/conversations/c-1')) {
        return { ok: true, json: async () => payload } as any;
      }
      if (url === '/api/v1/runs/run-1/feedback' && init?.method === 'POST') {
        return { ok: true, json: async () => ({ run_id: 'run-1', stored_path: '/tmp/ann.json', total_records: 1 }) } as any;
      }
      throw new Error('unexpected fetch ' + url);
    });

    render(
      <MemoryRouter initialEntries={["/conversation/run-1/c-1"]}>
        <Routes>
          <Route path="/conversation/:runId/:conversationId" element={<ConversationDetailPage />} />
        </Routes>
      </MemoryRouter>
    );

    await waitFor(() => expect(screen.getByText('Turn 1')).toBeInTheDocument());

    const rating = await screen.findByLabelText('rating-1') as HTMLInputElement;
    const notes = await screen.findByLabelText('notes-1') as HTMLTextAreaElement;
    const override = await screen.findByLabelText('override-1') as HTMLSelectElement;

    // Enter partial feedback -> submit disabled
    fireEvent.change(rating, { target: { value: '4.5' } });
    const submitBtn = screen.getByLabelText('submitfb-1') as HTMLButtonElement;
    expect(submitBtn).toBeDisabled();

    // Fill remaining required fields
    fireEvent.change(notes, { target: { value: 'Looks good' } });
    fireEvent.change(override, { target: { value: 'true' } });
    expect(submitBtn).not.toBeDisabled();

    // Ensure unsaved state persisted to localStorage before submit
    const key = 'feedback:run-1:c-1:1';
    const cached = JSON.parse((globalThis as any).localStorage.getItem(key) || '{}');
    expect(cached.rating).toBe('4.5');
    expect(cached.notes).toBe('Looks good');

    // Submit => backend called
    fireEvent.click(submitBtn);
    await waitFor(() => expect(screen.getByRole('status')).toHaveTextContent('Feedback saved'));
    // verify fetch post payload
    const postCall = (globalThis.fetch as any).mock.calls.find((c: any[]) => c[0] === '/api/v1/runs/run-1/feedback');
    expect(postCall).toBeTruthy();

    // Modify inputs again to validate persistence on navigate between sections
    fireEvent.change(rating, { target: { value: '3' } });
    const cached2 = JSON.parse((globalThis as any).localStorage.getItem(key) || '{}');
    expect(cached2.rating).toBe('3');
  });
});
