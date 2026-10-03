import { useEffect, useRef } from 'react';

const MAX_POINTS = 200;

function getEcgY(cycle) {
  if (cycle > 0.15 && cycle < 0.25) {
    return 45 - 8 * Math.sin((cycle - 0.15) * 10 * Math.PI);
  }
  if (cycle >= 0.35 && cycle < 0.38) return 51;
  if (cycle >= 0.38 && cycle < 0.43) return 7;
  if (cycle >= 0.43 && cycle < 0.47) return 55;
  if (cycle >= 0.58 && cycle < 0.75) {
    return 45 - 14 * Math.sin(((cycle - 0.58) / 0.17) * Math.PI);
  }
  return 45;
}

export default function HeartEcgMonitor({ bpm }) {
  const canvasRef = useRef(null);
  const bpmRef = useRef(bpm);
  bpmRef.current = bpm;

  useEffect(() => {
    const canvas = canvasRef.current;
    const context = canvas?.getContext('2d');
    const container = canvas?.parentElement;
    if (!canvas || !context || !container) return undefined;

    const points = Array(MAX_POINTS).fill(45);
    let width = 0;
    let height = 0;
    let elapsed = 0;
    let lastFrame = performance.now();
    let frame = 0;
    let stopped = false;

    function resize() {
      width = container.clientWidth;
      height = container.clientHeight;
      const pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = Math.round(width * pixelRatio);
      canvas.height = Math.round(height * pixelRatio);
      context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
    }

    function draw(now) {
      if (stopped) return;
      elapsed += (now - lastFrame) / 1000;
      lastFrame = now;
      const cycle = ((elapsed * bpmRef.current) / 60) % 1;
      points.push(getEcgY(cycle));
      points.shift();

      context.clearRect(0, 0, width, height);
      context.strokeStyle = '#00ff9d';
      context.lineWidth = 2;
      context.beginPath();
      for (let index = 0; index < points.length; index += 1) {
        const x = (index / (MAX_POINTS - 1)) * width;
        const y = (points[index] / 90) * height;
        if (index === 0) context.moveTo(x, y);
        else context.lineTo(x, y);
      }
      context.stroke();
      frame = window.requestAnimationFrame(draw);
    }

    const observer = new ResizeObserver(resize);
    observer.observe(container);
    resize();
    frame = window.requestAnimationFrame(draw);

    return () => {
      stopped = true;
      observer.disconnect();
      window.cancelAnimationFrame(frame);
    };
  }, []);

  return (
    <div>
      <div className="mb-2 flex items-center justify-between gap-2">
        <h3 className="m-0 text-xs font-semibold text-text">
          Canlı EKG monitörü (Lead II)
        </h3>
        <span className="rounded bg-accent/10 px-1.5 py-0.5 text-[10px] text-accent">
          {bpm} BPM
        </span>
      </div>
      <div className="h-[90px] rounded-md border border-line bg-[#090d13] p-1">
        <canvas
          aria-label={`Lead II canlı EKG simülasyonu, ${bpm} BPM`}
          className="block h-full w-full"
          ref={canvasRef}
          role="img"
        />
      </div>
    </div>
  );
}
