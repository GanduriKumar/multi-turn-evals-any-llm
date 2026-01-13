import React from 'react'

type Props = {
  title: string
  children: React.ReactNode
  className?: string
  borderless?: boolean
}

export default function Card({ title, children, className = '', borderless = false }: Props) {
  const containerBase = 'rounded-lg bg-white shadow-sm'
  const containerBorder = borderless ? '' : 'border border-gray-200'
  const headerBase = 'px-4 py-2 font-medium text-gray-800'
  const headerBorder = borderless ? '' : 'border-b border-gray-100'
  return (
    <div className={`${containerBase} ${containerBorder} ${className}`.trim()}>
      <div className={`${headerBase} ${headerBorder}`.trim()}>{title}</div>
      <div className="p-4">{children}</div>
    </div>
  )
}
