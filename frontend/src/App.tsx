import './App.css'
import ResultsViewer from './components/ResultsViewer'
import { useState } from 'react'

function App() {
  const [runId, setRunId] = useState('')

  return (
    <div className="container" style={{ padding: 24, maxWidth: '1200px', margin: '0 auto' }}>
      <h1>Evaluator Workbench</h1>
      
      <div className="run-selector" style={{ marginBottom: '2rem', padding: '1rem', background: '#f5f5f5', borderRadius: '8px' }}>
        <label style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
          <strong>Run ID:</strong>
          <input 
            value={runId} 
            onChange={(e) => setRunId(e.target.value)} 
            placeholder="Enter run ID (e.g. run-20240101...)" 
            style={{ padding: '8px', flex: 1, maxWidth: '400px' }}
          />
        </label>
      </div>

      <ResultsViewer runId={runId} />
    </div>
  )
}

export default App
