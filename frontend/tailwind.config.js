/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        background: '#0f172a', // slate-900
        surface: '#1e293b', // slate-800
        primary: '#38bdf8', // light blue
        accent: '#f59e0b', // amber-500
        critical: '#ef4444', // red-500
        warning: '#f59e0b', // amber-500
        ok: '#10b981', // emerald-500
      }
    },
  },
  plugins: [],
}
