// Path A interactive demo — CUSTOM WGSL backend. trunk8 (encode once/box) + perclick (per click),
// feeding the trunk's GPU output buffers straight into perclick (no readback).
import { initDevice, makeRunner, Net } from './wgpu-net.js?v=9';
let dev, R, trunk, perclick, P, N;

self.onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.type === 'init') {
      P = m.patch || 128; N = P * P * P;
      const t0 = performance.now();
      ({ dev } = await initDevice()); R = makeRunner(dev);
      trunk = new Net(dev, R); await trunk.load(`models/pathA/trunk8_${P}.graph.json`, `models/pathA/trunk8_${P}.weights.bin`);
      perclick = new Net(dev, R); await perclick.load(`models/pathA/perclick_${P}.graph.json`, `models/pathA/perclick_${P}.weights.bin`);
      // warm (shader compile) with zeros
      trunk.setInputData('img8', new Float32Array(8 * N)); trunk.run();
      perclick.setInputBuffer('s0', trunk.outBuf('s0')); perclick.setInputBuffer('s1', trunk.outBuf('s1'));
      perclick.setInputData('inter', new Float32Array(7 * N)); perclick.run();
      await perclick.read('logits'); await dev.queue.onSubmittedWorkDone();
      self.postMessage({ type: 'ready', backend: 'custom-wgsl', patch: P, ms: Math.round(performance.now() - t0) });
      return;
    }
    if (m.type === 'encode') {
      const img8 = new Float32Array(8 * N); img8.set(new Float32Array(m.image), 0);   // ch0 = z-scored image, ch1-7 = 0
      const t0 = performance.now();
      trunk.setInputData('img8', img8); trunk.run(); await dev.queue.onSubmittedWorkDone();
      perclick.setInputBuffer('s0', trunk.outBuf('s0')); perclick.setInputBuffer('s1', trunk.outBuf('s1'));
      self.postMessage({ type: 'encoded', ms: Math.round(performance.now() - t0) });
      return;
    }
    if (m.type === 'infer') {
      perclick.setInputData('inter', new Float32Array(m.inter));
      const t0 = performance.now();
      perclick.run();
      const lg = await perclick.read('logits');                 // f32 [2*N], channel-major
      const ms = Math.round(performance.now() - t0);
      const mask = new Uint8Array(N); let vox = 0;
      for (let i = 0; i < N; i++) { const fg = lg[N + i] > lg[i] ? 1 : 0; mask[i] = fg; vox += fg; }
      self.postMessage({ type: 'result', ms, vox, mask: mask.buffer }, [mask.buffer]);
      return;
    }
  } catch (err) { self.postMessage({ type: 'error', msg: String((err && err.message) || err) }); }
};
