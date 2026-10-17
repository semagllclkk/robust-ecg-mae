function StatusBadge({ children, tone = 'accent' }) {
  const toneClasses =
    tone === 'bad'
      ? 'border-bad text-bad bg-bad/10'
      : 'border-accent text-accent bg-accent/10';

  return (
    <div
      className={`rounded-full border px-3 py-1 text-xs ${toneClasses}`}
      role="status"
    >
      {children}
    </div>
  );
}

export default function DashboardLayout({
  children,
  apiStatus,
  backend,
  fps,
  onDownloadReport,
}) {
  const apiTone = apiStatus === 'Hata' ? 'bad' : 'accent';
  const backendLabel =
    backend.mode === 'fastapi'
      ? `FastAPI · ${backend.modelSource}`
      : backend.mode === 'mock'
        ? 'Mock-API (backend kapalı)'
        : 'Backend yoklanıyor…';

  return (
    <div className="min-h-screen">
      <header className="flex flex-wrap items-center justify-between gap-2.5 border-b border-line bg-panel px-5 py-3">
        <h1 className="m-0 text-base font-semibold">
          EKG İzleme Paneli{' '}
          <span className="font-normal text-accent">· MAE Klinik Karar Destek</span>
        </h1>
        <div className="flex items-center gap-2">
          <StatusBadge tone={apiTone}>
            {backendLabel} · {apiStatus}
          </StatusBadge>
          <StatusBadge>FPS: {fps ?? '–'}</StatusBadge>
          <button
            className="rounded-full border border-accent bg-accent/10 px-3 py-1 text-xs font-medium text-accent transition hover:bg-accent/20"
            onClick={onDownloadReport}
            type="button"
          >
            PDF İndir
          </button>
        </div>
      </header>
      <main className="grid min-h-[calc(100vh-54px)] grid-cols-1 gap-4 px-5 py-4 min-[901px]:grid-cols-[260px_minmax(0,1fr)] min-[1201px]:grid-cols-[260px_minmax(0,1fr)_400px] min-[1201px]:items-start">
        {children}
      </main>
    </div>
  );
}
