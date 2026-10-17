import { useEffect, useRef } from 'react';
import { api } from '../services/ecgApi.js';
import { LEADS, LEAD_COLORS, RECONSTRUCTION_COLOR } from '../data/ecg.js';

const SAMPLE_RATE = 100;
const WINDOW_SAMPLES = 1000;
const CAPACITY = 4096;
const PREFETCH = 100;
const LABEL_WIDTH = 34;

export default function ECGCanvas({
  disconnectedLeads,
  scenario,
  onFpsChange,
  onApiStatusChange,
}) {
  const canvasRef = useRef(null);
  const latestProps = useRef({ disconnectedLeads, scenario });
  const refetchRef = useRef(false);

  latestProps.current = { disconnectedLeads, scenario };

  // Senaryo veya kopuk kanal kümesi değişince görünen 10 s'lik pencere yeniden alınır
  useEffect(() => {
    refetchRef.current = true;
  }, [disconnectedLeads, scenario]);

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

    function resize() {
      const pixelRatio = window.devicePixelRatio || 1;
      width = container.clientWidth;
      height = container.clientHeight;
      canvas.width = Math.round(width * pixelRatio);
      canvas.height = Math.round(height * pixelRatio);
      context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
    }

    function prefetch() {
      const required = Math.floor(position) + PREFETCH;
      const refetch = refetchRef.current && available > 0;
      if (pending || (!refetch && available >= required) || performance.now() < retryAt) {
        return;
      }

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
      context.clearRect(0, 0, width, height);

      const rowHeight = height / LEADS.length;
      const pixelsPerSample = (width - LABEL_WIDTH) / WINDOW_SAMPLES;
      const scale = rowHeight * 0.42;
      const newest = Math.min(Math.floor(position), available - 1);

      context.lineWidth = 1;
      context.strokeStyle = '#231b2e';
      context.beginPath();
      for (let row = 1; row < LEADS.length; row += 1) {
        context.moveTo(0, row * rowHeight);
        context.lineTo(width, row * rowHeight);
      }
      context.stroke();

      context.font = '600 11px system-ui';
      context.textBaseline = 'middle';
      const disconnected = new Set(latestProps.current.disconnectedLeads);

      for (let leadIndex = 0; leadIndex < LEADS.length; leadIndex += 1) {
        const lead = LEADS[leadIndex];
        const centerY = rowHeight * (leadIndex + 0.5);
        const isDisconnected = disconnected.has(leadIndex);
        const color = LEAD_COLORS[lead.wall];

        context.fillStyle = isDisconnected ? RECONSTRUCTION_COLOR : color;
        context.fillText(lead.name, 8, centerY);

        if (!isDisconnected) {
          context.strokeStyle = color;
          context.lineWidth = 1.4;
          tracePath(samples, leadIndex, centerY, scale, pixelsPerSample, newest);
          continue;
        }

        if (hasReconstruction[leadIndex]) {
          context.strokeStyle = RECONSTRUCTION_COLOR;
          context.lineWidth = 1.4;
          context.setLineDash([5, 3]);
          tracePath(reconstructed, leadIndex, centerY, scale, pixelsPerSample, newest);
          context.setLineDash([]);
          context.font = '10px system-ui';
          context.fillText('rekonstrükte', width - 76, centerY - rowHeight * 0.32);
          context.font = '600 11px system-ui';
        } else {
          context.strokeStyle = '#ff4d6d';
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
        onFpsChange(Math.round((frames * 1000) / (now - fpsStart)));
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
  }, [onApiStatusChange, onFpsChange]);

  return (
    <section className="flex h-[620px] flex-col overflow-hidden rounded-2xl border border-line bg-panel min-[1201px]:sticky min-[1201px]:top-4 min-[1201px]:self-start">
      <div className="flex items-center justify-between gap-2 border-b border-line px-4 py-2 text-[11px] text-muted">
        <span className="font-semibold uppercase tracking-wider text-accent">
          12 Derivasyon · 10 s · 100 Hz
        </span>
        <span className="flex items-center gap-1.5">
          <i
            className="inline-block h-0 w-5 border-t-2 border-dashed align-middle"
            style={{ borderColor: RECONSTRUCTION_COLOR }}
          />
          Kopuk kanal → rekonstrükte sinyal
        </span>
      </div>
      <div className="flex-1 overflow-hidden bg-ecg-background">
        <canvas aria-label="12 derivasyonlu canlı EKG sinyalleri" className="block h-full w-full" ref={canvasRef} />
      </div>
    </section>
  );
}
