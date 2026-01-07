import { render, screen } from '@testing-library/react'
import '@testing-library/jest-dom'
import App from './App'

describe('App scaffold', () => {
  it('renders nav links and cards', () => {
    render(<App />)
    expect(screen.getByText('LLM Evals')).toBeInTheDocument()
    expect(screen.getByText('Quick Start')).toBeInTheDocument()
    expect(screen.getByText('Status')).toBeInTheDocument()
  })
})
