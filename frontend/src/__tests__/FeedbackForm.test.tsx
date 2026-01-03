/// <reference types="jest" />
import '@testing-library/jest-dom';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import FeedbackForm from '../components/FeedbackForm';
import { submitFeedback } from '../utils/api';

// Mock the API module
jest.mock('../utils/api', () => ({
  submitFeedback: jest.fn(),
}));

describe('FeedbackForm', () => {
  const defaultProps = {
    runId: 'run-1',
    datasetId: 'ds-1',
    conversationId: 'conv-1',
    modelName: 'gpt-4',
    turnId: 1,
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('renders all fields correctly', () => {
    render(<FeedbackForm {...defaultProps} />);
    
    expect(screen.getByLabelText(/Rating/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/Notes/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/Override Pass/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/Override Score/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Submit Feedback/i })).toBeInTheDocument();
  });

  test('submits form with entered values', async () => {
    (submitFeedback as jest.Mock).mockResolvedValue({
      run_id: 'run-1',
      stored_path: '/path/to/annotations.json',
      total_records: 1,
    });

    render(<FeedbackForm {...defaultProps} />);

    // Fill out form
    fireEvent.change(screen.getByLabelText(/Rating/i), { target: { value: '4.5' } });
    fireEvent.change(screen.getByLabelText(/Notes/i), { target: { value: 'Great response' } });
    fireEvent.change(screen.getByLabelText(/Override Pass/i), { target: { value: 'true' } });
    fireEvent.change(screen.getByLabelText(/Override Score/i), { target: { value: '0.9' } });

    // Submit
    fireEvent.click(screen.getByRole('button', { name: /Submit Feedback/i }));

    // Verify API call
    await waitFor(() => {
      expect(submitFeedback).toHaveBeenCalledWith('run-1', [
        expect.objectContaining({
          dataset_id: 'ds-1',
          conversation_id: 'conv-1',
          model_name: 'gpt-4',
          turn_id: 1,
          rating: 4.5,
          notes: 'Great response',
          override_pass: true,
          override_score: 0.9,
        }),
      ]);
    });

    // Verify success message
    expect(await screen.findByText(/Saved to/i)).toBeInTheDocument();
  });

  test('handles submission error', async () => {
    (submitFeedback as jest.Mock).mockRejectedValue(new Error('Network error'));

    render(<FeedbackForm {...defaultProps} />);

    fireEvent.click(screen.getByRole('button', { name: /Submit Feedback/i }));

    expect(await screen.findByText(/Network error/i)).toBeInTheDocument();
  });
});
