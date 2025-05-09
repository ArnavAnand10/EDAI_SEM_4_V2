// tailwind.config.js
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        primary: {
          DEFAULT: '#2563eb',  // blue-600
          light: '#3b82f6',    // blue-500
          dark: '#1d4ed8',     // blue-700
        },
        secondary: {
          DEFAULT: '#4b5563',  // gray-600
          light: '#6b7280',    // gray-500
          dark: '#374151',     // gray-700
        },
      },
      fontFamily: {
        sans: ['Inter', 'sans-serif'],
      },
    },
  },
  plugins: [
    require('@tailwindcss/forms'), // for better form styling
  ],
}