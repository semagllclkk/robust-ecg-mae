import { useEffect, useRef } from 'react';
import { api } from '../services/ecgApi.js';
import { LEADS, LEAD_COLORS } from '../data/ecg.js';

const SAMPLE_RATE = 100;
const WINDOW_SAMPLES = 1000;
const CAPACITY = 4096;
const PREFETCH = 100;

export default function ECGCanvas({
  disconnectedLeads,
  scenario,
  onFpsChange,
  onApiStatusChange,
}) {
  const canvasRef = useRef(null);
  const latestProps = useRef({ disconnectedLeads, scenario });

  latestProps.current = { disconnectedLeads, scenario };

  useEffect(() => {
    const canvas = canvasRef.current;
    const context = canvas?.getContext('2d');
    const container = canvas?.parentElement;

    if (!canvas || !context || !container) {
      return undefined;
    }

    const samples = new Float32Array(LEADS.length * CAPACITY);
    let width = 0;
    let height = 0;
    let position = 1000;
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
      if (pending || available >= required || performance.now() < retryAt) {
        return;
      }

      pending = true;
      const requestStart = available;
      api
        .chunk(requestStart, required - requestStart, latestProps.current.scenario)
        .then(({ data }) => {
          if (stopped) return;
          const count = data[0]?.length ?? 0;
          for (let leadIndex = 0; leadIndex < LEADS.length; leadIndex += 1) {
            for (let sampleIndex = 0; sampleIndex < count; sampleIndex += 1) {
              samples[
                leadIndex * CAPACITY + ((requestStart + sampleIndex) % CAPACITY)
              ] = data[leadIndex][sampleIndex];
            }
          }
          available = requestStart + count;
          onApiStatusChange('Bağlı');
        })
        .catch((error) => {
          if (stopped) return;
          console.error('EKG sinyal verisi alınamadı:', error);
          onApiStatusChange('Hata');
          retryAt = performance.now() + 1000;
        })
        .finally(() => {
          pending = false;
        });
    }

    function draw(now) {
      if (stopped) return;
      position += ((now - lastFrame) / 1000) * SAMPLE_RATE;
      lastFrame = now;
      prefetch();
      context.clearRect(0, 0, width, height);

      const rowHeight = height / LEADS.length;
      const pixelsPerSample = width / WINDOW_SAMPLES;
      const scale = rowHeight * 0.35;
      const newest = Math.min(Math.floor(position), available - 1);

      context.lineWidth = 1;
      context.strokeStyle = '#231b2e';
      context.beginPath();
      for (let row = 1; row < LEADS.length; row += 1) {
        context.moveTo(0, row * rowHeight);
        context.lineTo(width, row * rowHeight);
      }
      context.stroke();

      context.font = '12px system-ui';
      context.textBaseline = 'middle';
      const disconnected = new Set(latestProps.current.disconnectedLeads);

      for (let leadIndex = 0; leadIndex < LEADS.length; leadIndex += 1) {
        const lead = LEADS[leadIndex];
        const centerY = rowHeight * (leadIndex + 0.5);
        const isDisconnected = disconnected.has(leadIndex);
        const color = isDisconnected ? '#ff4d6d' : LEAD_COLORS[lead.wall];
        context.fillStyle = color;
        context.fillText(lead.name, 10, centerY - rowHeight * 0.25);

        if (isDisconnected) {
          context.strokeStyle = color;
          context.setLineDash([4, 4]);
          context.beginPath();
          context.moveTo(0, centerY);
          context.lineTo(width, centerY);
          context.stroke();
          context.setLineDash([]);
          continue;
        }

        context.strokeStyle = color;
        context.lineWidth = 1.5;
        context.beginPath();
        let firstPoint = true;
        for (
          let sample = Math.max(0, Math.floor(position) - WINDOW_SAMPLES);
          sample <= newest;
          sample += 1
        ) {
          const x = width - (position - sample) * pixelsPerSample;
          const y =
            centerY -
            samples[leadIndex * CAPACITY + (sample % CAPACITY)] * scale;
          if (firstPoint) {
            context.moveTo(x, y);
            firstPoint = false;
          } else {
            context.lineTo(x, y);
          }
        }
        context.stroke();
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
    <section className="flex min-h-[500px] flex-col overflow-hidden rounded-2xl border border-line bg-panel p-0">
      <div className="min-h-[500px] flex-1 overflow-hidden rounded-lg border border-line bg-ecg-background">
        <canvas aria-label="12 derivasyonlu canlı EKG sinyalleri" className="block h-full w-full" ref={canvasRef} />
      </div>
    </section>
  );
}
