/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        background: '#120e15',
        panel: '#1e1724',
        line: '#32263d',
        text: '#f3eef9',
        muted: '#a395b5',
        accent: {
          DEFAULT: '#ff6eb0',
          hover: '#ff8ec2',
        },
        bad: '#ff4d6d',
        lead: {
          ant: '#4cc9f0',
          inf: '#f6c453',
          lat: '#7bd88f',
          neutral: '#9aa3b2',
        },
        ischemic: {
          lateral: '#b31225',
          anterior: '#d91a32',
          inferior: '#8f0d1c',
        },
        control: '#2a1f33',
        tooltip: '#140f19',
        ecg: {
          background: '#0a080c',
          grid: '#231b2e',
        },
      },
      keyframes: {
        'pulse-error': {
          '0%, 100%': {
            fill: '#ff0000',
            filter: 'drop-shadow(0 0 10px #ff0000)',
          },
          '50%': {
            fill: '#ffffff',
            filter: 'drop-shadow(0 0 15px #ffffff)',
          },
        },
      },
      animation: {
        'pulse-error': 'pulse-error 1s infinite',
      },
    },
  },
  plugins: [],
};
