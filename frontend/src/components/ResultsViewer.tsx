import React, { useState, useEffect } from 'react';
import { submitFeedback } from '../utils/api';
import type { TurnFeedback } from '../utils/api';

export type ResultsViewerProps = {
  runId: string;
};

type Turn = {
  turn_id: number | string;
  prompt: string;
  response: string;
  weighted_score: number;
  passed: boolean;
  metrics: Record<string, number>;
};

type ConversationResult = {
  dataset_id: string;
  conversation_id: string;
  model_name: string;
  turns: Turn[];
  aggregate: {
    score: number;
    passed: boolean;
  };
};

type RunResults = {
  run: {
    run_id: string;
  };
  results: ConversationResult[];
};

export default function ResultsViewer({ runId }: ResultsViewerProps) {
  const [results, setResults] = useState<RunResults | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedTurn, setSelectedTurn] = useState<{
    datasetId: string;
    conversationId: string;
    modelName: string;
    turn: Turn;
  } | null>(null);

  // Feedback state
  const [rating, setRating] = useState<number | ''>('');
  const [notes, setNotes] = useState('');
  const [overridePass, setOverridePass] = useState<boolean | ''>('');
  const [overrideScore, setOverrideScore] = useState<number | ''>('');
  const [submitting, setSubmitting] = useState(false);
  const [feedbackSuccess, setFeedbackSuccess] = useState<string | null>(null);

  useEffect(() => {
    if (!runId) return;
    
    const fetchResults = async () => {
      setLoading(true);
      setError(null);
      try {
        const res = await fetch(`/api/v1/results/${runId}/results`);
        if (!res.ok) {
          const text = await res.text();
          throw new Error(`Failed to fetch results: ${res.status} ${text}`);
        }
        const data = await res.json();
        setResults(data);
      } catch (err: any) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchResults();
  }, [runId]);

  const handleTurnClick = (datasetId: string, conversationId: string, modelName: string, turn: Turn) => {
    setSelectedTurn({ datasetId, conversationId, modelName, turn });
    // Reset form
    setRating('');
    setNotes('');
    setOverridePass('');
    setOverrideScore('');
    setFeedbackSuccess(null);
  };

  const handleSubmitFeedback = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedTurn) return;

    setSubmitting(true);
    try {
      const payload: TurnFeedback = {
        dataset_id: selectedTurn.datasetId,
        conversation_id: selectedTurn.conversationId,
        model_name: selectedTurn.modelName,
        turn_id: selectedTurn.turn.turn_id,
        rating: rating === '' ? null : Number(rating),
        notes: notes || null,
        override_pass: overridePass === '' ? null : Boolean(overridePass),
        override_score: overrideScore === '' ? null : Number(overrideScore),
      };

      await submitFeedback(runId, [payload]);
      setFeedbackSuccess("Feedback saved successfully!");
      
      // Clear success message after 3 seconds
      setTimeout(() => setFeedbackSuccess(null), 3000);
    } catch (err: any) {
      alert(`Failed to submit: ${err.message}`);
    } finally {
      setSubmitting(false);
    }
  };

  // Mock data for demonstration if no results loaded
  const loadMockData = () => {
    setResults({
      run: { run_id: runId || "mock-run-123" },
      results: [
        {
          dataset_id: "ds-1",
          conversation_id: "conv-1",
          model_name: "gpt-4",
          aggregate: { score: 0.85, passed: true },
          turns: [
            {
              turn_id: 1,
              prompt: "Hello, how are you?",
              response: "I'm doing well, thank you! How can I help you today?",
              weighted_score: 0.9,
              passed: true,
              metrics: { "helpfulness": 0.9 }
            },
            {
              turn_id: 2,
              prompt: "Write a poem about coding.",
              response: "Code flows like a river stream...",
              weighted_score: 0.8,
              passed: true,
              metrics: { "creativity": 0.8 }
            }
          ]
        }
      ]
    });
    setError(null);
  };

  return (
    <div className="results-viewer">
      <div className="controls">
        <button onClick={loadMockData} style={{ marginBottom: '1rem' }}>Load Demo Data</button>
      </div>

      {error && <div className="error-message">{error}</div>}
      
      <div className="viewer-layout" style={{ display: 'flex', gap: '20px' }}>
        {/* Left Panel: Conversation List */}
        <div className="conversation-list" style={{ flex: 1, borderRight: '1px solid #ccc', paddingRight: '20px' }}>
          <h3>Conversations</h3>
          {results?.results.map((res, idx) => (
            <div key={idx} className="conversation-item" style={{ marginBottom: '20px', padding: '10px', background: '#f5f5f5', borderRadius: '4px' }}>
              <div style={{ fontWeight: 'bold' }}>{res.conversation_id}</div>
              <div style={{ fontSize: '0.8em', color: '#666' }}>{res.model_name}</div>
              <div style={{ marginTop: '5px' }}>
                {res.turns.map(turn => (
                  <div 
                    key={turn.turn_id}
                    onClick={() => handleTurnClick(res.dataset_id, res.conversation_id, res.model_name, turn)}
                    style={{ 
                      cursor: 'pointer', 
                      padding: '5px', 
                      margin: '2px 0',
                      background: selectedTurn?.turn.turn_id === turn.turn_id && selectedTurn?.conversationId === res.conversation_id ? '#e0e0ff' : 'white',
                      border: '1px solid #ddd',
                      borderRadius: '3px'
                    }}
                  >
                    Turn {turn.turn_id}: {turn.passed ? '✅' : '❌'} ({turn.weighted_score.toFixed(2)})
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>

        {/* Right Panel: Turn Details & Feedback */}
        <div className="turn-details" style={{ flex: 2 }}>
          {selectedTurn ? (
            <>
              <h3>Turn Details</h3>
              <div className="turn-content" style={{ background: '#fff', padding: '15px', border: '1px solid #ddd', borderRadius: '4px', marginBottom: '20px' }}>
                <div style={{ marginBottom: '10px' }}>
                  <strong>Prompt:</strong>
                  <pre style={{ whiteSpace: 'pre-wrap', background: '#f9f9f9', padding: '10px' }}>{selectedTurn.turn.prompt}</pre>
                </div>
                <div style={{ marginBottom: '10px' }}>
                  <strong>Response:</strong>
                  <pre style={{ whiteSpace: 'pre-wrap', background: '#f0f8ff', padding: '10px' }}>{selectedTurn.turn.response}</pre>
                </div>
                <div>
                  <strong>Metrics:</strong>
                  <ul>
                    {Object.entries(selectedTurn.turn.metrics).map(([k, v]) => (
                      <li key={k}>{k}: {v}</li>
                    ))}
                  </ul>
                </div>
              </div>

              <div className="feedback-section" style={{ background: '#f0f0f0', padding: '15px', borderRadius: '4px' }}>
                <h4>Evaluator Feedback</h4>
                <form onSubmit={handleSubmitFeedback}>
                  <div style={{ marginBottom: '10px' }}>
                    <label style={{ display: 'block', marginBottom: '5px' }}>Rating (0-5)</label>
                    <input 
                      type="number" 
                      min="0" 
                      max="5" 
                      step="0.5" 
                      value={rating} 
                      onChange={e => setRating(e.target.value === '' ? '' : Number(e.target.value))}
                      style={{ width: '100%', padding: '5px' }}
                    />
                  </div>
                  
                  <div style={{ marginBottom: '10px' }}>
                    <label style={{ display: 'block', marginBottom: '5px' }}>Notes</label>
                    <textarea 
                      value={notes} 
                      onChange={e => setNotes(e.target.value)}
                      style={{ width: '100%', padding: '5px', minHeight: '80px' }}
                    />
                  </div>

                  <div style={{ display: 'flex', gap: '10px', marginBottom: '10px' }}>
                    <div style={{ flex: 1 }}>
                      <label style={{ display: 'block', marginBottom: '5px' }}>Override Pass</label>
                      <select 
                        value={overridePass === '' ? '' : overridePass ? 'true' : 'false'}
                        onChange={e => setOverridePass(e.target.value === '' ? '' : e.target.value === 'true')}
                        style={{ width: '100%', padding: '5px' }}
                      >
                        <option value="">No Change</option>
                        <option value="true">Pass</option>
                        <option value="false">Fail</option>
                      </select>
                    </div>
                    <div style={{ flex: 1 }}>
                      <label style={{ display: 'block', marginBottom: '5px' }}>Override Score</label>
                      <input 
                        type="number" 
                        min="0" 
                        max="1" 
                        step="0.01" 
                        value={overrideScore} 
                        onChange={e => setOverrideScore(e.target.value === '' ? '' : Number(e.target.value))}
                        style={{ width: '100%', padding: '5px' }}
                      />
                    </div>
                  </div>

                  <button 
                    type="submit" 
                    disabled={submitting}
                    style={{ 
                      background: '#007bff', 
                      color: 'white', 
                      border: 'none', 
                      padding: '10px 20px', 
                      borderRadius: '4px', 
                      cursor: submitting ? 'not-allowed' : 'pointer' 
                    }}
                  >
                    {submitting ? 'Saving...' : 'Save Feedback'}
                  </button>
                  
                  {feedbackSuccess && (
                    <div style={{ marginTop: '10px', color: 'green', fontWeight: 'bold' }}>
                      {feedbackSuccess}
                    </div>
                  )}
                </form>
              </div>
            </>
          ) : (
            <div style={{ padding: '20px', textAlign: 'center', color: '#666' }}>
              Select a turn from the left to view details and provide feedback.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
