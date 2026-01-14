import React from 'react'

type Props = {
  content: string
  role: 'user' | 'assistant'
}

// Simple markdown-style formatter for chat messages
function formatContent(text: string): React.ReactNode {
  if (!text) return null
  
  // Split by code blocks first (``` delimited)
  const codeBlockRegex = /```(\w*)\n?([\s\S]*?)```/g
  const parts: React.ReactNode[] = []
  let lastIndex = 0
  let match: RegExpExecArray | null
  
  while ((match = codeBlockRegex.exec(text)) !== null) {
    // Add text before code block
    if (match.index > lastIndex) {
      parts.push(formatInline(text.slice(lastIndex, match.index)))
    }
    // Add code block
    const lang = match[1] || 'text'
    const code = match[2]
    parts.push(
      <pre key={match.index} className="bg-gray-800 text-gray-100 p-3 rounded text-xs overflow-x-auto my-2">
        <code>{code}</code>
      </pre>
    )
    lastIndex = match.index + match[0].length
  }
  
  // Add remaining text
  if (lastIndex < text.length) {
    parts.push(formatInline(text.slice(lastIndex)))
  }
  
  return <>{parts}</>
}

function formatInline(text: string): React.ReactNode {
  // Split by lines and handle lists, bold, inline code
  const lines = text.split('\n')
  const elements: React.ReactNode[] = []
  
  lines.forEach((line, idx) => {
    if (!line.trim()) {
      elements.push(<br key={`br-${idx}`} />)
      return
    }
    
    // Bullet list
    if (line.match(/^[\s]*[-*]\s+/)) {
      const content = line.replace(/^[\s]*[-*]\s+/, '')
      elements.push(
        <div key={idx} className="ml-4 flex gap-2">
          <span>•</span>
          <span>{formatInlineStyles(content)}</span>
        </div>
      )
      return
    }
    
    // Numbered list
    if (line.match(/^[\s]*\d+\.\s+/)) {
      const match = line.match(/^[\s]*(\d+)\.\s+(.*)/)
      if (match) {
        elements.push(
          <div key={idx} className="ml-4 flex gap-2">
            <span>{match[1]}.</span>
            <span>{formatInlineStyles(match[2])}</span>
          </div>
        )
        return
      }
    }
    
    // Regular paragraph
    elements.push(
      <div key={idx} className="mb-1">
        {formatInlineStyles(line)}
      </div>
    )
  })
  
  return <>{elements}</>
}

function formatInlineStyles(text: string): React.ReactNode {
  // Handle **bold**, `code`, and plain text
  const parts: React.ReactNode[] = []
  let remaining = text
  let key = 0
  
  // Bold
  const boldRegex = /\*\*([^*]+)\*\*/g
  // Inline code
  const codeRegex = /`([^`]+)`/g
  
  // Combine patterns
  const combinedRegex = /(\*\*[^*]+\*\*|`[^`]+`)/g
  let lastIndex = 0
  let match: RegExpExecArray | null
  
  while ((match = combinedRegex.exec(text)) !== null) {
    // Add text before match
    if (match.index > lastIndex) {
      parts.push(text.slice(lastIndex, match.index))
    }
    
    const matched = match[0]
    if (matched.startsWith('**') && matched.endsWith('**')) {
      // Bold
      parts.push(<strong key={key++}>{matched.slice(2, -2)}</strong>)
    } else if (matched.startsWith('`') && matched.endsWith('`')) {
      // Inline code
      parts.push(
        <code key={key++} className="bg-gray-700 text-gray-100 px-1 rounded text-xs">
          {matched.slice(1, -1)}
        </code>
      )
    }
    
    lastIndex = match.index + match[0].length
  }
  
  // Add remaining text
  if (lastIndex < text.length) {
    parts.push(text.slice(lastIndex))
  }
  
  return parts.length > 0 ? <>{parts}</> : text
}

export default function ChatMessage({ content, role }: Props) {
  return (
    <div className={`p-3 rounded ${role === 'user' ? 'bg-base-100' : 'bg-base-300'}`}>
      <div className="text-base font-bold opacity-80 mb-1">{role === 'user' ? 'User' : 'Assistant'}</div>
      <div className="text-sm leading-relaxed">
        {formatContent(content)}
      </div>
    </div>
  )
}
