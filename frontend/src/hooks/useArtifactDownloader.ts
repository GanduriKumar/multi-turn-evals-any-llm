import { useCallback, useRef, useState } from 'react'

type DownloadState = {
  downloading: boolean
  progress: number | null // 0-100 or null when unknown
  error: string | null
  filename: string | null
}

function parseFilenameFromContentDisposition(v: string | null): string | null {
  if (!v) return null
  // Simple parser for: attachment; filename="name.ext"
  const m = /filename\*=UTF-8''([^;]+)|filename="?([^";]+)"?/i.exec(v)
  if (m) return decodeURIComponent(m[1] || m[2])
  return null
}

function saveBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  // Allow some time before revoking in case of slow handlers
  setTimeout(() => URL.revokeObjectURL(url), 2000)
}

export function useArtifactDownloader(runId: string) {
  const [state, setState] = useState<DownloadState>({ downloading: false, progress: null, error: null, filename: null })
  const abortRef = useRef<AbortController | null>(null)

  const download = useCallback(
    async (artifacts: string[], opts?: { theme?: 'default' | 'dark' | 'compact' }) => {
      if (!runId) {
        setState((s) => ({ ...s, error: 'Run ID is required' }))
        return
      }
      if (state.downloading) return
      setState({ downloading: true, progress: 0, error: null, filename: null })
      const ac = new AbortController()
      abortRef.current = ac
      try {
        const params = new URLSearchParams()
        for (const a of artifacts) params.append('artifact', a)
        if (opts?.theme) params.set('theme', opts.theme)
        const url = `/api/v1/runs/${encodeURIComponent(runId)}/artifacts?${params.toString()}`
        const res = await fetch(url, { signal: ac.signal })
        if (!res.ok) {
          const text = await res.text().catch(() => '')
          throw new Error(`${res.status} ${text || res.statusText || 'Download failed'}`)
        }

        const cd = res.headers?.get?.('content-disposition') || null
        const cl = res.headers?.get?.('content-length') || null
        const inferredName = parseFilenameFromContentDisposition(cd) || (artifacts.length > 1 ? `${runId}_artifacts.zip` : `${artifacts[0]}.${
          artifacts[0] === 'csv' ? 'csv' : artifacts[0] === 'html' ? 'html' : artifacts[0] === 'markdown' ? 'md' : artifacts[0] === 'summary' || artifacts[0] === 'results' ? 'json' : 'bin'
        }`)

        // Stream with progress if possible
        const total = cl ? Number(cl) : NaN
        const reader = (res as any).body?.getReader ? (res as any).body.getReader() : null
        if (reader) {
          const chunks: Uint8Array[] = []
          let received = 0
          for (;;) {
            const { done, value } = await reader.read()
            if (done) break
            if (value) {
              chunks.push(value)
              received += value.byteLength
              if (Number.isFinite(total)) {
                const pct = Math.max(0, Math.min(100, Math.round((received / total) * 100)))
                setState((s) => ({ ...s, progress: pct }))
              } else {
                setState((s) => ({ ...s, progress: null }))
              }
            }
          }
          const blob = new Blob(chunks, { type: res.headers?.get?.('content-type') || 'application/octet-stream' })
          saveBlob(blob, inferredName)
          setState({ downloading: false, progress: 100, error: null, filename: inferredName })
        } else {
          // Fallback: no streaming available in this env
          const blob = await (res as any).blob?.()
          if (!blob) throw new Error('Empty response')
          saveBlob(blob, inferredName)
          setState({ downloading: false, progress: null, error: null, filename: inferredName })
        }
      } catch (e: any) {
        if (e?.name === 'AbortError') {
          setState((s) => ({ ...s, downloading: false, error: 'Download cancelled' }))
        } else {
          setState({ downloading: false, progress: null, error: e?.message || String(e), filename: null })
        }
      } finally {
        abortRef.current = null
      }
    },
    [runId, state.downloading]
  )

  const cancel = useCallback(() => {
    abortRef.current?.abort()
  }, [])

  return {
    ...state,
    download,
    cancel,
  }
}

export type UseArtifactDownloader = ReturnType<typeof useArtifactDownloader>
