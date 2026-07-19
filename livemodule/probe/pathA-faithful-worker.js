// Faithful single-resolution (192³) worker: image-only trunk (encode once per patch position/zoom)
// + perclick decode (per interaction). The 7-ch inter carries prev_seg for autoregressive refinement.
import { initDevice, makeRunner, Net } from './wgpu-net.js?v=9';
let dev, R, net = null, N = 0, P = 0;

self.onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.type === 'init') {
      const t0 = performance.now();
      P = m.res || 192; N = P*P*P;
      ({ dev } = await initDevice()); R = makeRunner(dev);
      const trunk = new Net(dev, R); await trunk.load(`models/pathA/faithful/trunk8_${P}.graph.json?v=${m.v||1}`, `models/pathA/faithful/trunk8_${P}.weights.bin?v=${m.v||1}`);
      const perclick = new Net(dev, R); await perclick.load(`models/pathA/faithful/perclick_${P}.graph.json?v=${m.v||1}`, `models/pathA/faithful/perclick_${P}.weights.bin?v=${m.v||1}`);
      trunk.setInputData('img8', new Float32Array(8*N)); trunk.run();                   // warm
      perclick.setInputBuffer('s0', trunk.outBuf('s0')); perclick.setInputBuffer('s1', trunk.outBuf('s1'));
      perclick.setInputData('inter', new Float32Array(7*N)); perclick.run(); await perclick.read('logits');
      net = { trunk, perclick };
      await dev.queue.onSubmittedWorkDone();
      self.postMessage({ type: 'ready', res: P, ms: Math.round(performance.now() - t0) });
      return;
    }
    if (m.type === 'encode') {                    // {image: Float32Array(P^3) already z-scored}
      const img8 = new Float32Array(8*N); img8.set(new Float32Array(m.image), 0);
      const t0 = performance.now();
      net.trunk.setInputData('img8', img8); net.trunk.run(); await dev.queue.onSubmittedWorkDone();
      net.perclick.setInputBuffer('s0', net.trunk.outBuf('s0')); net.perclick.setInputBuffer('s1', net.trunk.outBuf('s1'));
      self.postMessage({ type: 'encoded', ms: Math.round(performance.now() - t0) });
      return;
    }
    if (m.type === 'infer') {                      // {inter: Float32Array(7*P^3)}
      net.perclick.setInputData('inter', new Float32Array(m.inter));
      const t0 = performance.now();
      net.perclick.run(); const lg = await net.perclick.read('logits');
      const mask = new Uint8Array(N); let vox = 0, nan = 0;
      for (let i = 0; i < N; i++) { const a=lg[i], b=lg[N+i];
        if (Number.isNaN(a)||Number.isNaN(b)) nan++;
        const fg = b > a ? 1 : 0; mask[i] = fg; vox += fg; }
      self.postMessage({ type: 'result', ms: Math.round(performance.now() - t0), vox, nan, mask: mask.buffer }, [mask.buffer]);
      return;
    }
  } catch (err) { self.postMessage({ type: 'error', msg: String((err && err.message) || err) }); }
};
