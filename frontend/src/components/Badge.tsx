import React from 'react'

type Variant = 'primary' | 'success' | 'warning' | 'danger' | 'neutral'

export default function Badge({ variant = 'neutral', className = '', children }: { variant?: Variant, className?: string, children: React.ReactNode }) {
  const base = 'inline-flex items-center px-2 py-0.5 rounded text-xs font-medium'
  const styles: Record<Variant, string> = {
    primary: 'bg-primary/10 text-primary',
    success: 'bg-success/10 text-success',
    warning: 'bg-warning/10 text-warning',
    danger: 'bg-danger/10 text-danger',
    neutral: 'bg-gray-100 text-gray-700',
  }
  return <span className={`${base} ${styles[variant]} ${className}`}>{children}</span>
}
