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
  fps,
}) {
  const apiTone = apiStatus === 'Hata' ? 'bad' : 'accent';

  return (
    <div className="min-h-screen">
      <header className="flex flex-wrap items-center justify-between gap-2.5 border-b border-line bg-panel px-5 py-3">
        <h1 className="m-0 text-base font-semibold">
          EKG İzleme Paneli{' '}
          <span className="font-normal text-accent">· İP2 &amp; İP4 XAI Modülü</span>
        </h1>
        <div className="flex gap-2">
          <StatusBadge tone={apiTone}>API: {apiStatus}</StatusBadge>
          <StatusBadge>FPS: {fps ?? '–'}</StatusBadge>
        </div>
      </header>
      <main className="grid min-h-[calc(100vh-54px)] grid-cols-1 gap-4 px-5 py-4 min-[901px]:grid-cols-[280px_minmax(0,1fr)] min-[1201px]:grid-cols-[280px_minmax(0,1fr)_300px]">
        {children}
      </main>
    </div>
  );
}
