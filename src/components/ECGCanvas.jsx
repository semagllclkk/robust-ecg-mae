import { useEffect, useRef } from 'react';
import { api } from '../services/ecgApi.js';
import { LEADS, LEAD_COLORS, RECONSTRUCTION_COLOR } from '../data/ecg.js';
import { publishFps } from '../services/metrics.js';
import Panel from './ui/Panel.jsx';

const SAMPLE_RATE = 100;
const WINDOW_SAMPLES = 1000;
const CAPACITY = 4096;
// Tampon 1 s'nin altına düşünce 2 s'lik blok istenir (~0.5 istek/s; önceden her karede istek gidiyordu)
const LOW_WATER = 100;
const BATCH = 200;
const LABEL_WIDTH = 44;
// EKG kâğıdı: 25 mm/s → 10 s = 250 mm; genlik 10 mm/mV (satıra sığmazsa küçültülür)
const PAPER_MM = (WINDOW_SAMPLES / SAMPLE_RATE) * 25;
const PAPER = { bg: '#fffbfa', minor: '#f7dcdc', major: '#eeb4b4', ink: '#1d2127' };

export default function ECGCanvas({
  disconnectedLeads,
  recordKey = 0,
  scenario,
  onApiStatusChange,
}) {
  const canvasRef = useRef(null);
  const latestProps = useRef({ disconnectedLeads, scenario });
  const refetchRef = useRef(false);

  latestProps.current = { disconnectedLeads, scenario };

  // Senaryo, hasta kaydı veya kopuk kanal kümesi değişince görünen 10 s'lik pencere yeniden alınır
  useEffect(() => {
    refetchRef.current = true;
  }, [disconnectedLeads, scenario, recordKey]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const context = canvas?.getContext('2d');
    const container = canvas?.parentElement;

    if (!canvas || !context || !container) {
      return undefined;
    }

    const samples = new Float32Array(LEADS.length * CAPACITY);
    const reconstructed = new Float32Array(LEADS.length * CAPACITY);
    const hasReconstruction = new Uint8Array(LEADS.length);
    let width = 0;
    let height = 0;
    let position = WINDOW_SAMPLES;
    let available = 0;
    let pending = false;
    let retryAt = 0;
    let lastFrame = performance.now();
    let fpsStart = lastFrame;
    let frames = 0;
    let animationFrame = 0;
    let stopped = false;

    const grid = document.createElement('canvas');
    let pxPerMm = 1;

    // Izgara yalnızca boyut değişince çizilir, her karede kopyalanır
    function drawGrid(pixelRatio) {
      grid.width = canvas.width;
      grid.height = canvas.height;
      const g = grid.getContext('2d');
      g.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
      g.fillStyle = PAPER.bg;
      g.fillRect(0, 0, width, height);
      pxPerMm = (width - LABEL_WIDTH) / PAPER_MM;
      for (const [step, color] of [[1, PAPER.minor], [5, PAPER.major]]) {
        g.strokeStyle = color;
        g.lineWidth = step === 5 ? 0.8 : 0.5;
        g.beginPath();
        for (let mm = 0; mm * pxPerMm <= width; mm += step) {
          const x = width - mm * pxPerMm;
          g.moveTo(x, 0);
          g.lineTo(x, height);
        }
        for (let mm = 0; mm * pxPerMm <= height; mm += step) {
          g.moveTo(0, mm * pxPerMm);
          g.lineTo(width, mm * pxPerMm);
        }
        g.stroke();
      }
    }

    function resize() {
      const pixelRatio = window.devicePixelRatio || 1;
      width = container.clientWidth;
      height = container.clientHeight;
      canvas.width = Math.round(width * pixelRatio);
      canvas.height = Math.round(height * pixelRatio);
      context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
      drawGrid(pixelRatio);
    }

    function prefetch() {
      const refetch = refetchRef.current && available > 0;
      if (pending || (!refetch && available >= Math.floor(position) + LOW_WATER) || performance.now() < retryAt) {
        return;
      }
      const required = Math.floor(position) + LOW_WATER + BATCH;

      // Sekme arka plandayken rAF durur; dönüşte yalnızca görünen pencere istenir
      const windowStart = Math.max(0, Math.floor(position) - WINDOW_SAMPLES);
      const requestStart = refetch ? windowStart : Math.max(available, windowStart);
      const requestEnd = Math.max(required, available);
      const requestedLeads = latestProps.current.disconnectedLeads;
      refetchRef.current = false;
      pending = true;

      api
        .chunk({
          start: requestStart,
          count: requestEnd - requestStart,
          condition: latestProps.current.scenario,
          disconnectedLeads: requestedLeads,
        })
        .then(({ data, reconstructed: recon = {} }) => {
          if (stopped) return;
          const count = data[0]?.length ?? 0;
          for (let leadIndex = 0; leadIndex < LEADS.length; leadIndex += 1) {
            const reconSeries = recon[leadIndex];
            hasReconstruction[leadIndex] = reconSeries ? 1 : 0;
            for (let k = 0; k < count; k += 1) {
              const slot = leadIndex * CAPACITY + ((requestStart + k) % CAPACITY);
              samples[slot] = data[leadIndex][k];
              reconstructed[slot] = reconSeries ? reconSeries[k] : 0;
            }
          }
          available = Math.max(available, requestStart + count);
          onApiStatusChange('Bağlı');
        })
        .catch((error) => {
          if (stopped) return;
          console.error('EKG sinyal verisi alınamadı:', error);
          onApiStatusChange('Hata');
          retryAt = performance.now() + 1000;
          refetchRef.current = refetch;
        })
        .finally(() => {
          pending = false;
        });
    }

    function tracePath(buffer, leadIndex, centerY, scale, pixelsPerSample, newest) {
      context.beginPath();
      let firstPoint = true;
      for (
        let sample = Math.max(0, Math.floor(position) - WINDOW_SAMPLES);
        sample <= newest;
        sample += 1
      ) {
        const x = LABEL_WIDTH + (width - LABEL_WIDTH) - (position - sample) * pixelsPerSample;
        if (x < LABEL_WIDTH) continue;
        const y = centerY - buffer[leadIndex * CAPACITY + (sample % CAPACITY)] * scale;
        if (firstPoint) {
          context.moveTo(x, y);
          firstPoint = false;
        } else {
          context.lineTo(x, y);
        }
      }
      context.stroke();
    }

    function draw(now) {
      if (stopped) return;
      position += ((now - lastFrame) / 1000) * SAMPLE_RATE;
      lastFrame = now;
      prefetch();
      context.setTransform(1, 0, 0, 1, 0, 0);
      context.drawImage(grid, 0, 0);
      context.setTransform(canvas.width / width, 0, 0, canvas.height / height, 0, 0);

      const rowHeight = height / LEADS.length;
      const pixelsPerSample = (width - LABEL_WIDTH) / WINDOW_SAMPLES;
      const scale = Math.min(10 * pxPerMm, rowHeight * 0.48);
      const newest = Math.min(Math.floor(position), available - 1);
      const disconnected = new Set(latestProps.current.disconnectedLeads);

      context.textBaseline = 'middle';
      for (let leadIndex = 0; leadIndex < LEADS.length; leadIndex += 1) {
        const lead = LEADS[leadIndex];
        const centerY = rowHeight * (leadIndex + 0.5);
        const isDisconnected = disconnected.has(leadIndex);

        // Etiket: duvar rengi işareti + derivasyon adı
        context.fillStyle = isDisconnected ? RECONSTRUCTION_COLOR : LEAD_COLORS[lead.wall];
        context.fillRect(8, centerY - 5, 3, 10);
        context.fillStyle = isDisconnected ? RECONSTRUCTION_COLOR : PAPER.ink;
        context.font = '600 11px Inter, system-ui, sans-serif';
        context.fillText(lead.name, 15, centerY);

        context.lineJoin = 'round';
        if (!isDisconnected) {
          context.strokeStyle = PAPER.ink;
          context.lineWidth = 1.25;
          tracePath(samples, leadIndex, centerY, scale, pixelsPerSample, newest);
          continue;
        }

        if (hasReconstruction[leadIndex]) {
          context.strokeStyle = RECONSTRUCTION_COLOR;
          context.lineWidth = 1.4;
          context.setLineDash([5, 3]);
          tracePath(reconstructed, leadIndex, centerY, scale, pixelsPerSample, newest);
          context.setLineDash([]);
          context.font = '500 10px Inter, system-ui, sans-serif';
          context.textAlign = 'right';
          context.fillText('rekonstrükte', width - 8, centerY - rowHeight * 0.32);
          context.textAlign = 'left';
        } else {
          context.strokeStyle = '#c8102e';
          context.lineWidth = 1;
          context.setLineDash([4, 4]);
          context.beginPath();
          context.moveTo(LABEL_WIDTH, centerY);
          context.lineTo(width, centerY);
          context.stroke();
          context.setLineDash([]);
        }
      }

      frames += 1;
      if (now - fpsStart >= 500) {
        publishFps(Math.round((frames * 1000) / (now - fpsStart)));
        frames = 0;
        fpsStart = now;
      }
      animationFrame = window.requestAnimationFrame(draw);
    }

    resize();
    const observer = new ResizeObserver(resize);
    observer.observe(container);
    onApiStatusChange('Bağlanıyor…');
    animationFrame = window.requestAnimationFrame(draw);

    return () => {
      stopped = true;
      observer.disconnect();
      window.cancelAnimationFrame(animationFrame);
    };
  }, [onApiStatusChange]);

  return (
    <Panel
      actions={
        <span className="flex items-center gap-1.5 text-xs text-muted">
          <i className="inline-block w-5 border-t-2 border-dashed border-recon" />
          Rekonstrükte
        </span>
      }
      bodyClassName="p-0"
      className="overflow-hidden"
      meta={<span className="num">25 mm/s · 10 mm/mV · 100 Hz</span>}
      title="12 derivasyonlu EKG"
    >
      <div className="h-[540px]">
        <canvas aria-label="12 derivasyonlu canlı EKG sinyalleri" className="block h-full w-full" ref={canvasRef} />
      </div>
    </Panel>
  );
}
