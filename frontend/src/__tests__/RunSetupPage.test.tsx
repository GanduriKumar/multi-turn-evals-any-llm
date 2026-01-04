import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import * as React from 'react';

// Runtime mock for react-router-dom so tests don't require the real package
vi.mock('react-router-dom', () => {
  const noop = () => null;
  return {
    MemoryRouter: ({ children }: any) => React.createElement(React.Fragment, null, children),
    Link: ({ children }: any) => React.createElement('a', null, children),
    NavLink: ({ children }: any) => React.createElement('a', null, children),
    Outlet: noop,
    Navigate: noop,
    useNavigate: () => vi.fn(),
    useLocation: () => ({ pathname: '/', search: '', hash: '', state: null, key: 'test' }),
    useParams: () => ({}),
    useSearchParams: () => [new URLSearchParams(), vi.fn()],
    Routes: ({ children }: any) => React.createElement(React.Fragment, null, children),
    Route: ({ children }: any) => React.createElement(React.Fragment, null, children),
  };
});

import { MemoryRouter } from 'react-router-dom';
import RunSetupPage from '../pages/RunSetupPage';

const mockDatasets = [
  { id: 'a', name: 'Conv A', conversation: '/c/a.json', golden: '/g/a.golden.yaml', tags: ['general'], difficulty: 'easy' },
  { id: 'b', name: 'Conv B', conversation: '/c/b.json', golden: '/g/b.golden.yaml', tags: ['medical'], difficulty: 'hard' },
];

describe('RunSetupPage', () => {
  beforeEach(() => {
    (globalThis.fetch as any) = vi.fn(async (url: string, init?: any) => {
      if (url.startsWith('/api/v1/datasets')) {
        return { ok: true, json: async () => mockDatasets } as any;
      }
      if (url === '/api/v1/runs/' && init?.method === 'POST') {
        const body = JSON.parse(init.body);
        // verify payload minimally
        expect(body.version).toBe('1.0.0');
        expect(body.datasets.length).toBeGreaterThan(0);
        expect(body.models[0]).toEqual(expect.objectContaining({ provider: expect.any(String), model: expect.any(String) }));
        return { ok: true, json: async () => ({ run_id: 'run-123', job_id: 'job-1', status: 'queued' }) } as any;
      }
      throw new Error('unexpected fetch');
    });
  });

  it('submits run config and shows success', async () => {
    render(
      <MemoryRouter>
        <RunSetupPage />
      </MemoryRouter>
    );

    await waitFor(() => expect(screen.getByText('Conv A')).toBeInTheDocument());

    // Select a dataset
    const checkbox = screen.getAllByRole('checkbox')[0];
    fireEvent.click(checkbox);

    // Set provider/model
    fireEvent.change(screen.getByLabelText('Provider'), { target: { value: 'openai' } });
    fireEvent.change(screen.getByLabelText('Model'), { target: { value: 'gpt-4o' } });

    // Submit
    fireEvent.click(screen.getByText('Start Run'));

    await waitFor(() => expect(screen.getByText(/Run started successfully/)).toBeInTheDocument());
    expect(screen.getByText('run-123')).toBeInTheDocument();
  });

  it('shows Ollama and sets default models per provider', async () => {
    render(
      <MemoryRouter>
        <RunSetupPage />
      </MemoryRouter>
    );

    await waitFor(() => expect(screen.getByText('Conv A')).toBeInTheDocument());

    const provider = screen.getByLabelText('Provider') as HTMLSelectElement;
    const model = screen.getByLabelText('Model') as HTMLInputElement;

    // Provider menu includes Ollama
    expect(provider).toHaveTextContent('Ollama');

    // Ollama default
    fireEvent.change(provider, { target: { value: 'ollama' } });
    expect(model.value).toBe('llama3.2');

    // Google default
    fireEvent.change(provider, { target: { value: 'google' } });
    expect(model.value).toBe('gemini-2.5');

    // OpenAI default
    fireEvent.change(provider, { target: { value: 'openai' } });
    expect(model.value).toBe('gpt-5.2');

    // Azure OpenAI default (MoE)
    fireEvent.change(provider, { target: { value: 'azure_openai' } });
    expect(model.value).toBe('phi-3.5-moe');

    // AWS Bedrock default (MoE)
    fireEvent.change(provider, { target: { value: 'aws_bedrock' } });
    expect(model.value).toBe('mixtral-8x7b-instruct');
  });
});
