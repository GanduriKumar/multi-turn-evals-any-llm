import React from 'react'

type Variant = 'primary' | 'secondary' | 'success' | 'warning' | 'danger' | 'ghost'

export default function Button({ variant = 'primary', className = '', ...props }: React.ButtonHTMLAttributes<HTMLButtonElement> & { variant?: Variant }) {
  const base = 'px-3 py-1.5 rounded font-medium focus:outline-none focus-visible:ring-2 focus-visible:ring-offset-2'
  const styles: Record<Variant, string> = {
    primary: 'bg-primary text-white hover:opacity-90 focus-visible:ring-primary/50',
    secondary: 'border border-primary text-primary hover:bg-primary/5 focus-visible:ring-primary/40',
    success: 'bg-success text-white hover:opacity-90 focus-visible:ring-success/50',
    warning: 'bg-warning text-white hover:opacity-90 focus-visible:ring-warning/50',
    danger: 'bg-danger text-white hover:opacity-90 focus-visible:ring-danger/50',
    ghost: 'text-primary hover:bg-primary/10 focus-visible:ring-primary/30'
  }
  return <button className={`${base} ${styles[variant]} ${className}`} {...props} />
}
