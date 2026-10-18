/** @type {import('tailwindcss').Config} */
const token = (name) => `rgb(var(--${name}) / <alpha-value>)`;

export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        bg: token('bg'),
        surface: token('surface'),
        sunken: token('sunken'),
        line: token('line'),
        'line-strong': token('line-strong'),
        text: token('text'),
        muted: token('muted'),
        subtle: token('subtle'),
        accent: token('accent'),
        danger: token('danger'),
        warning: token('warning'),
        success: token('success'),
        recon: token('recon'),
        wall: {
          ant: token('wall-ant'),
          inf: token('wall-inf'),
          lat: token('wall-lat'),
          neu: token('wall-neu'),
        },
      },
      fontFamily: {
        sans: ['Inter', 'ui-sans-serif', 'system-ui', 'Segoe UI', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'ui-monospace', 'SFMono-Regular', 'Consolas', 'monospace'],
      },
      borderRadius: {
        panel: '10px',
      },
    },
  },
  plugins: [],
};
