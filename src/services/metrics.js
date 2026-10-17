import { useSyncExternalStore } from 'react';

/*
 * Sık değişen ölçümler (FPS) React state'inde tutulursa her güncellemede tüm
 * uygulama yeniden render olur; burada yalnızca abone olan bileşen güncellenir.
 */
let fps = null;
const listeners = new Set();

export function publishFps(value) {
  if (value === fps) return;
  fps = value;
  listeners.forEach((listener) => listener());
}

function subscribe(listener) {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

export function useFps() {
  return useSyncExternalStore(subscribe, () => fps);
}
