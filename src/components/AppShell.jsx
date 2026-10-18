import { useFps } from '../services/metrics.js';
import { Button } from './ui/controls.jsx';

function FpsBadge() {
  const fps = useFps();
  return <span className="num hidden text-xs text-subtle sm:inline">{fps ?? '–'} fps</span>;
}

function BackendStatus({ backend, streamStatus }) {
  const ok = streamStatus !== 'Hata';
  const label =
    backend.mode === 'fastapi'
      ? `FastAPI · ${backend.modelSource}`
      : backend.mode === 'mock'
        ? 'Mock-API · backend kapalı'
        : 'Backend yoklanıyor';
  const dot = !ok ? 'bg-danger' : backend.mode === 'fastapi' ? 'bg-success' : 'bg-warning';

  return (
    <div className="flex items-center gap-2 text-xs text-muted" role="status">
      <span className={`h-2 w-2 rounded-full ${dot}`} />
      <span className="max-w-[220px] truncate">{label}</span>
    </div>
  );
}

export default function AppShell({ backend, streamStatus, onDownloadReport, sidebar, children }) {
  return (
    <div className="min-h-screen">
      <header className="sticky top-0 z-40 border-b border-line bg-surface/90 backdrop-blur">
        <div className="mx-auto flex h-14 max-w-[1600px] items-center justify-between gap-4 px-4 lg:px-6">
          <div className="flex min-w-0 items-center gap-3">
            <svg aria-hidden="true" className="h-6 w-6 shrink-0 text-accent" fill="none" viewBox="0 0 24 24">
              <rect height="22" rx="5" stroke="currentColor" strokeWidth="1.5" width="22" x="1" y="1" />
              <path
                d="M4 13h4l1.6-4 2.4 8 2-6 1.2 2H20"
                stroke="currentColor"
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="1.6"
              />
            </svg>
            <div className="min-w-0">
              <h1 className="m-0 truncate text-sm font-semibold leading-tight">Robust ECG-MAE</h1>
              <p className="m-0 truncate text-[11px] text-subtle">
                Eksik derivasyon toleranslı MI tespiti ve anatomik haritalama
              </p>
            </div>
          </div>
          <div className="flex items-center gap-4">
            <BackendStatus backend={backend} streamStatus={streamStatus} />
            <FpsBadge />
            <Button onClick={onDownloadReport}>Rapor (PDF)</Button>
          </div>
        </div>
      </header>

      <div className="mx-auto grid max-w-[1600px] gap-4 p-4 lg:grid-cols-[272px_minmax(0,1fr)] lg:p-6">
        <aside className="flex flex-col gap-4">{sidebar}</aside>
        <main className="flex min-w-0 flex-col gap-4">{children}</main>
      </div>
    </div>
  );
}
