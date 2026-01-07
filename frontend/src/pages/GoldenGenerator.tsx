import React, { useMemo, useState } from 'react'
import Card from '../components/Card'
import Button from '../components/Button'
import { Select } from '../components/Form'

 

type Domain = 'commerce' | 'banking'

type ConversationTurn = { role: 'user' | 'assistant', text: string }

type DatasetDoc = {
  dataset_id: string
  version: string
  metadata: { domain: Domain, difficulty: 'easy'|'medium'|'hard', tags?: string[] }
  conversations: { conversation_id: string, turns: ConversationTurn[] }[]
}

type GoldenDoc = {
  dataset_id: string
  version: string
  entries: { conversation_id: string, turns: { turn_index: number, expected: { variants: string[] } }[], final_outcome: { decision: 'ALLOW'|'DENY'|'PARTIAL', refund_amount?: number, reason_code?: string, next_action?: string, policy_flags?: string[] } }[]
}

function genId(prefix: string) {
  const n = Math.floor(Math.random()*1e6).toString(36)
  return `${prefix}-${n}`
}

function buildConversation(domain: Domain, difficulty: 'easy'|'medium'|'hard', outcome: 'ALLOW'|'DENY'|'PARTIAL'): {dataset: DatasetDoc, golden: GoldenDoc} {
  const dataset_id = genId(`${domain}-${difficulty}-${outcome}`)
  const version = '1.0.0'
  const convoId = genId('conv')

  const userProblem = domain === 'commerce'
    ? 'My order arrived damaged. I want a refund.'
    : 'I noticed a suspicious transaction. Can you help?'

  const assistantProbe = domain === 'commerce'
    ? 'I am sorry to hear that. Could you share your order ID and item details?'
    : 'I can help. Could you share the transaction ID and amount?'

  const userDetails = domain === 'commerce'
    ? 'Order #A123, item: headphones, price $79.'
    : 'Transaction T-9876 for $250 yesterday.'

  const assistantPolicy = domain === 'commerce'
    ? 'Based on the policy, we can process a refund if damage is confirmed.'
    : 'According to policy, we can freeze the card and start a dispute.'

  const assistantResolution = outcome === 'ALLOW'
    ? (domain === 'commerce' ? 'I have approved a full refund of $79.' : 'I have blocked your card and started a dispute; you will be reimbursed $250 if validated.')
    : outcome === 'PARTIAL'
      ? (domain === 'commerce' ? 'I can offer a partial refund of $40.' : 'We can issue a temporary credit of $100 pending investigation.')
      : (domain === 'commerce' ? 'We cannot refund without proof of damage. Please provide photos.' : 'We cannot credit without verification. Please submit a dispute form.')

  const dataset: DatasetDoc = {
    dataset_id,
    version,
    metadata: { domain, difficulty, tags: ['template'] },
    conversations: [
      {
        conversation_id: convoId,
        turns: [
          { role: 'user', text: userProblem },
          { role: 'assistant', text: assistantProbe },
          { role: 'user', text: userDetails },
          { role: 'assistant', text: assistantPolicy },
          { role: 'user', text: 'Thanks, what can you do for me now?' },
          { role: 'assistant', text: assistantResolution },
        ],
      }
    ]
  }

  const golden: GoldenDoc = {
    dataset_id,
    version,
    entries: [
      {
        conversation_id: convoId,
        turns: [
          { turn_index: 1, expected: { variants: [assistantProbe] } },
          { turn_index: 3, expected: { variants: [assistantPolicy] } },
          { turn_index: 5, expected: { variants: [assistantResolution] } },
        ],
        final_outcome: {
          decision: outcome,
          refund_amount: domain === 'commerce' ? (outcome === 'ALLOW' ? 79 : outcome === 'PARTIAL' ? 40 : 0) : undefined,
          next_action: domain === 'banking' ? (outcome === 'ALLOW' ? 'dispute' : 'verify') : undefined,
          policy_flags: outcome === 'DENY' ? ['NEEDS_EVIDENCE'] : [],
        }
      }
    ]
  }

  return { dataset, golden }
}

export default function GoldenGeneratorPage() {
  const [domain, setDomain] = useState<Domain>('commerce')
  const [difficulty, setDifficulty] = useState<'easy'|'medium'|'hard'>('easy')
  const [outcome, setOutcome] = useState<'ALLOW'|'DENY'|'PARTIAL'>('ALLOW')
  const [bundle, setBundle] = useState<{dataset: any, golden: any} | null>(null)

  const generate = () => {
    const b = buildConversation(domain, difficulty, outcome)
    setBundle(b)
  }

  const download = (name: 'dataset'|'golden') => {
    if (!bundle) return
    const blob = new Blob([JSON.stringify(bundle[name], null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `${bundle[name].dataset_id}.${name}.json`
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div className="grid gap-4">
      <Card title="Golden Generator">
        <div className="grid sm:grid-cols-3 gap-4 text-sm">
          <label className="flex items-center gap-2"><span className="w-28">Domain</span>
            <Select className="grow" value={domain} onChange={e => setDomain(e.target.value as Domain)}>
              <option value="commerce">Commerce</option>
              <option value="banking">Banking</option>
            </Select>
          </label>
          <label className="flex items-center gap-2"><span className="w-28">Difficulty</span>
            <Select className="grow" value={difficulty} onChange={e => setDifficulty(e.target.value as any)}>
              <option value="easy">Easy</option>
              <option value="medium">Medium</option>
              <option value="hard">Hard</option>
            </Select>
          </label>
          <label className="flex items-center gap-2"><span className="w-28">Outcome</span>
            <Select className="grow" value={outcome} onChange={e => setOutcome(e.target.value as any)}>
              <option value="ALLOW">ALLOW</option>
              <option value="DENY">DENY</option>
              <option value="PARTIAL">PARTIAL</option>
            </Select>
          </label>
        </div>
        <div className="mt-4">
          <Button variant="primary" onClick={generate}>Generate</Button>
        </div>
      </Card>

      {bundle && (
        <Card title="Preview & Download">
          <div className="flex flex-wrap gap-2 mb-2">
            <Button variant="secondary" onClick={() => download('dataset')}>Download dataset.json</Button>
            <Button variant="secondary" onClick={() => download('golden')}>Download golden.json</Button>
          </div>
          <pre className="text-xs bg-gray-50 p-3 rounded overflow-auto max-h-80">{JSON.stringify(bundle, null, 2)}</pre>
        </Card>
      )}
    </div>
  )
}
