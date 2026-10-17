/** Küçük, tekrar kullanılan form bileşenleri. */

export function Button({ variant = 'secondary', className = '', ...props }) {
  const variants = {
    primary: 'border-accent bg-accent text-white hover:bg-accent/90',
    secondary: 'border-line-strong bg-surface text-text hover:bg-sunken',
    ghost: 'border-transparent bg-transparent text-muted hover:bg-sunken hover:text-text',
  };
  return (
    <button
      className={`inline-flex h-8 items-center justify-center gap-1.5 rounded-md border px-3 text-xs font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-50 ${variants[variant]} ${className}`}
      type="button"
      {...props}
    />
  );
}

export function SegmentedControl({ options, value, onChange, label, columns }) {
  return (
    <div
      aria-label={label}
      className="grid gap-0.5 rounded-md border border-line bg-sunken p-0.5"
      role="radiogroup"
      style={{ gridTemplateColumns: `repeat(${columns ?? options.length}, minmax(0, 1fr))` }}
    >
      {options.map((option) => {
        const active = option.value === value;
        return (
          <button
            aria-checked={active}
            className={`h-7 truncate rounded-[5px] px-2 text-xs font-medium transition-colors ${
              active ? 'bg-surface text-text shadow-[0_1px_2px_rgb(0_0_0/0.08)]' : 'text-muted hover:text-text'
            }`}
            key={option.value}
            onClick={() => onChange(option.value)}
            role="radio"
            type="button"
          >
            {option.label}
          </button>
        );
      })}
    </div>
  );
}

export function Slider({ label, value, display, ...props }) {
  return (
    <label className="block">
      <span className="flex items-center justify-between text-xs text-muted">
        {label}
        <span className="num text-text">{display ?? value}</span>
      </span>
      <input className="mt-1.5 w-full accent-[rgb(var(--accent))]" type="range" value={value} {...props} />
    </label>
  );
}

export function ProbabilityBar({ value, tone = 'accent' }) {
  const tones = { accent: 'bg-accent', danger: 'bg-danger', subtle: 'bg-line-strong' };
  return (
    <div className="h-1.5 overflow-hidden rounded-full bg-sunken">
      <div
        className={`h-full rounded-full transition-[width] duration-500 ${tones[tone]}`}
        style={{ width: `${Math.round(value * 100)}%` }}
      />
    </div>
  );
}
