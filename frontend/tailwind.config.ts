import type { Config } from 'tailwindcss'

export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        primary: '#4285F4', // Google blue
        success: '#0F9D58', // Google green
        warning: '#F4B400', // Google yellow
        danger: '#DB4437',  // Google red
      },
    },
  },
  plugins: [],
} satisfies Config
