// Path A interactive demo worker (WASM, 64^3 fp32). encode-once trunk + per-click decode.
import * as ort from './ort/ort.webgpu.bundle.min.mjs';
ort.env.wasm.wasmPaths = self.location.origin + '/ort/';
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(8, navigator.hardwareConcurrency || 4) : 1;
ort.env.logLevel = 'error';

let P = 64, N = P * P * P;
let trunk = null, perclick = null, s0 = null, s1 = null;

self.onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.type === 'init') {
      P = m.patch || 64; N = P * P * P;
      const t0 = performance.now();
      const opt = { executionProviders: ['wasm'] };
      trunk = await ort.InferenceSession.create(`models/pathA/trunk_${P}.onnx`, opt);
      perclick = await ort.InferenceSession.create(`models/pathA/perclick_${P}.onnx`, opt);
      // warm both (first run pays kernel setup)
      const tr = await trunk.run({ image: new ort.Tensor('float32', new Float32Array(N), [1, 1, P, P, P]) });
      await perclick.run({ s0: tr.s0, s1: tr.s1, inter: new ort.Tensor('float32', new Float32Array(7 * N), [1, 7, P, P, P]) });
      self.postMessage({ type: 'ready', threads: ort.env.wasm.numThreads, ms: Math.round(performance.now() - t0) });
      return;
    }
    if (m.type === 'encode') {
      const image = new ort.Tensor('float32', new Float32Array(m.image), [1, 1, P, P, P]);
      const t0 = performance.now();
      const tr = await trunk.run({ image });
      s0 = tr.s0; s1 = tr.s1;
      self.postMessage({ type: 'encoded', ms: Math.round(performance.now() - t0) });
      return;
    }
    if (m.type === 'infer') {
      const inter = new ort.Tensor('float32', new Float32Array(m.inter), [1, 7, P, P, P]);
      const t0 = performance.now();
      const out = await perclick.run({ s0, s1, inter });
      const lg = out.logits.data;                       // [1,2,P,P,P] channel-major
      const mask = new Uint8Array(N);
      let vox = 0;
      for (let i = 0; i < N; i++) { const fg = lg[N + i] > lg[i] ? 1 : 0; mask[i] = fg; vox += fg; }
      self.postMessage({ type: 'result', ms: Math.round(performance.now() - t0), vox, mask: mask.buffer }, [mask.buffer]);
      return;
    }
  } catch (err) {
    self.postMessage({ type: 'error', msg: String((err && err.message) || err) });
  }
};
