import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import DatasetsPage from '../pages/DatasetsPage';

const mockDatasets = [
  { id: 'a', name: 'Conv A', conversation: '/c/a.json', golden: '/g/a.golden.yaml', tags: ['general'], difficulty: 'easy' },
  { id: 'b', name: 'Conv B', conversation: '/c/b.json', golden: '/g/b.golden.yaml', tags: ['medical'], difficulty: 'hard' },
];

global.fetch = vi.fn(async () => ({ ok: true, json: async () => mockDatasets })) as any;

describe('DatasetsPage', () => {
  beforeEach(() => {
    (fetch as any).mockClear();
  });

  it('fetches and displays datasets', async () => {
    render(
      <MemoryRouter>
        <DatasetsPage />
      </MemoryRouter>
    );

    expect(fetch).toHaveBeenCalledWith('/api/v1/datasets/?page=1&page_size=1000');

    // Wait for items
    await waitFor(() => expect(screen.getByText('Conv A')).toBeInTheDocument());
    expect(screen.getByText('Conv B')).toBeInTheDocument();

    // Filter by domain
    fireEvent.change(screen.getByLabelText('Domain'), { target: { value: 'general' } });
    expect(screen.getByText('Conv A')).toBeInTheDocument();
    expect(screen.queryByText('Conv B')).not.toBeInTheDocument();
  });
});
