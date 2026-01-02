/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        brand: {
          50: '#f5faff',
          100: '#e6f0ff',
          200: '#bfd8ff',
          300: '#91bbff',
          400: '#5f9cff',
          500: '#2d7dff',
          600: '#1f5fd6',
          700: '#1849a6',
          800: '#123578',
          900: '#0c2350',
        },
      },
    },
  },
  plugins: [],
}
