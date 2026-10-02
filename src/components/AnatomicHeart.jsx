import { useRef, useState } from 'react';
import { LEADS, REGION_INFO } from '../data/ecg.js';

export default function AnatomicHeart({ disconnectedLeads }) {
  const [hoveredRegion, setHoveredRegion] = useState(null);
  const [tooltipPosition, setTooltipPosition] = useState({ x: 0, y: 0 });
  const heartAreaRef = useRef(null);
  const regionOrder = ['lat', 'ant', 'inf'];

  return (
    <section className="relative flex flex-1 flex-col rounded-2xl border border-line bg-panel p-4">
      <h2 className="mb-3 text-xs font-semibold uppercase tracking-wider text-accent">
        3D Anatomik Harita
      </h2>
      <div className="relative flex flex-1 items-center justify-center" ref={heartAreaRef}>
        <svg
          aria-label="Anatomik kalp duvarları"
          className="w-full max-w-[250px] overflow-visible [filter:drop-shadow(0_0_20px_rgba(255,0,0,0.15))]"
          viewBox="0 0 100 100"
        >
          {regionOrder.map((region) => {
            const info = REGION_INFO[region];
            const isAffected = info.leads.some((index) =>
              disconnectedLeads.includes(index),
            );
            return (
              <path
                aria-label={info.title}
                className={`cursor-crosshair stroke-line stroke-2 transition-all duration-300 hover:brightness-125 hover:stroke-accent hover:stroke-[3px] ${
                  isAffected
                    ? 'animate-pulse-error stroke-white stroke-[3px]'
                    : info.fillClass
                }`}
                d={info.path}
                key={region}
                onMouseEnter={() => setHoveredRegion(region)}
                onMouseLeave={() => setHoveredRegion(null)}
                onMouseMove={(event) => {
                  const bounds = heartAreaRef.current.getBoundingClientRect();
                  setTooltipPosition({
                    x: Math.max(
                      8,
                      Math.min(event.clientX - bounds.left + 15, bounds.width - 228),
                    ),
                    y: Math.max(
                      8,
                      Math.min(event.clientY - bounds.top + 15, bounds.height - 140),
                    ),
                  });
                }}
              />
            );
          })}
        </svg>
        {hoveredRegion && (
          <HeartTooltip
            info={REGION_INFO[hoveredRegion]}
            disconnectedLeads={disconnectedLeads}
            position={tooltipPosition}
          />
        )}
      </div>
      <p className="mb-0 mt-2.5 text-center text-xs text-muted">
        * Kalp segmentlerinin üzerine gelerek detayları görebilirsiniz.
      </p>
    </section>
  );
}

function HeartTooltip({ info, disconnectedLeads, position }) {
  const broken = info.leads
    .filter((index) => disconnectedLeads.includes(index))
    .map((index) => LEADS[index].name);

  return (
    <div
      className="pointer-events-none absolute z-10 w-[220px] rounded-xl border border-accent bg-tooltip/95 p-3 text-[13px] text-text shadow-xl backdrop-blur"
      style={{ left: position.x, top: position.y }}
    >
      <h3 className="mb-1.5 mt-0 text-sm text-accent">{info.title}</h3>
      <p className="m-0 leading-snug text-muted">{info.description}</p>
      {broken.length > 0 && (
        <p className="mb-0 mt-2 text-bad">
          ⚠️ Kopuk Kanallar: {broken.join(', ')}
        </p>
      )}
    </div>
  );
}
