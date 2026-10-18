/** Uygulamadaki tüm kartların ortak kabuğu: başlık satırı + gövde. */
export default function Panel({ title, meta, actions, children, className = '', bodyClassName = 'p-4' }) {
  return (
    <section className={`flex flex-col rounded-panel border border-line bg-surface ${className}`}>
      {(title || meta || actions) && (
        <header className="flex min-h-[44px] items-center justify-between gap-3 border-b border-line px-4 py-2.5">
          <div className="flex min-w-0 items-baseline gap-2">
            {title && <h2 className="m-0 truncate text-[13px] font-semibold text-text">{title}</h2>}
            {meta && <span className="truncate text-xs text-subtle">{meta}</span>}
          </div>
          {actions && <div className="flex shrink-0 items-center gap-1.5">{actions}</div>}
        </header>
      )}
      <div className={`min-h-0 flex-1 ${bodyClassName}`}>{children}</div>
    </section>
  );
}
