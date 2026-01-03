import React from 'react';
import { act } from 'react-dom/test-utils';
import { createRoot } from 'react-dom/client';
import FeedbackForm from '../components/FeedbackForm';
// Jest globals are provided by the test environment; declare them for TypeScript without importing '@jest/globals'
declare var expect: any;
declare var jest: any;
declare var it: any;
declare var beforeEach: any;
declare var afterEach: any;

type RenderResult = { container: HTMLElement; unmount: () => void };

function render(ui: React.ReactElement): RenderResult {
  const container = document.createElement('div');
  document.body.appendChild(container);
  const root = createRoot(container);
  act(() => {
    root.render(ui);
  });
  return {
    container,
    unmount: () => {
      act(() => root.unmount());
      container.remove();
    },
  };
}

function getByLabelText(
  container: HTMLElement,
  text: RegExp | string
): HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement {
  const matcher = typeof text === 'string' ? new RegExp(text, 'i') : text;
  const labels = Array.from(container.querySelectorAll('label')) as HTMLLabelElement[];
  const label = labels.find((l) => matcher.test(l.textContent || ''));
  if (!label) throw new Error(`Label not found for ${text}`);
  const forId = label.getAttribute('for');
  if (forId) {
    const esc = (globalThis as any).CSS?.escape ?? ((s: string) => s);
    const byId = container.querySelector('#' + esc(forId)) as any;
    if (byId) return byId;
  }
  const control = label.querySelector('input, textarea, select') as any;
  if (control) return control;
  throw new Error(`Control for label not found for ${text}`);
}

function getByRole(
  container: HTMLElement,
  role: string,
  options?: { name?: RegExp | string }
): HTMLElement {
  const nameMatcher = options?.name
    ? typeof options.name === 'string'
      ? new RegExp(options.name, 'i')
      : options.name
    : null;

  let candidates = Array.from(container.querySelectorAll(`[role="${role}"]`));
  if (role === 'button') {
    candidates = candidates.concat(Array.from(container.querySelectorAll('button')));
  }

  const match = candidates.find((el) =>
    nameMatcher ? nameMatcher.test(el.textContent || '') : true
  );
  if (!match) throw new Error(`Role ${role} not found`);
  return match as HTMLElement;
}

function change(el: HTMLElement, value: string) {
  act(() => {
    (el as HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement).value = value as any;
    el.dispatchEvent(new Event('input', { bubbles: true }));
    el.dispatchEvent(new Event('change', { bubbles: true }));
  });
}

function click(el: HTMLElement) {
  act(() => {
    el.dispatchEvent(new MouseEvent('click', { bubbles: true }));
  });
}

async function waitFor(
  assertion: () => void,
  timeout = 2000,
  interval = 50
): Promise<void> {
  const end = Date.now() + timeout;
  let lastErr: any;
  while (Date.now() < end) {
    try {
      assertion();
      return;
    } catch (e) {
      lastErr = e;
      await new Promise((r) => setTimeout(r, interval));
    }
  }
  throw lastErr;
}

// Minimal mock for fetch
const originalFetch = (globalThis as any).fetch;

beforeEach(() => {
  (globalThis as any).fetch = jest.fn(async (input: RequestInfo, init?: RequestInit) => {
    // Simulate OK response
    return {
      ok: true,
      json: async () => ({ run_id: 'run-123', stored_path: '/runs/run-123/annotations.json', total_records: 1 }),
      text: async () => 'ok',
      status: 200,
    } as any;
  });
});

afterEach(() => {
  (globalThis as any).fetch = originalFetch;
});

it('submits feedback payload to backend and shows success', async () => {
  const onSubmitted = jest.fn();
  const { container } = render(
    <FeedbackForm
      runId="run-123"
      datasetId="ds1"
      conversationId="c1"
      modelName="m1"
      turnId={1}
      onSubmitted={onSubmitted}
    />
  );

  // Fill form
  const rating = getByLabelText(container, /Rating/i) as HTMLInputElement;
  change(rating, '4.5');

  const notes = getByLabelText(container, /Notes/i) as HTMLTextAreaElement;
  change(notes, 'Looks good');

  const overridePass = getByLabelText(container, /Override Pass/i) as HTMLSelectElement;
  change(overridePass, 'true');

  const overrideScore = getByLabelText(container, /Override Score/i) as HTMLInputElement;
  change(overrideScore, '0.9');

  // Submit
  const btn = getByRole(container, 'button', { name: /Submit Feedback/i });
  click(btn);

  await waitFor(() => expect(onSubmitted).toHaveBeenCalled());

  // Verify success message
  expect(getByRole(container, 'status')).toHaveTextContent('annotations.json');

  // Verify fetch called with correct payload
  expect((globalThis as any).fetch).toHaveBeenCalled();
  const call = ((globalThis as any).fetch as any).mock.calls[0];
  const body = JSON.parse(call[1].body as string);
  expect(body.run_id).toBe('run-123');
  expect(body.feedback[0]).toEqual({
    dataset_id: 'ds1',
    conversation_id: 'c1',
    model_name: 'm1',
    turn_id: 1,
    rating: 4.5,
    notes: 'Looks good',
    override_pass: true,
    override_score: 0.9,
  });
});
