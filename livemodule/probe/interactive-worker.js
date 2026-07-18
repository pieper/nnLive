// Persistent warm WebGPU session for click-local interactive segmentation. Create the session ONCE,
// warm it (first run compiles shaders), then each click is just a short warm forward on a small patch
// around the click — so latency after the first click is the steady-state compute, and the submission
// is short enough that WindowServer never starves. No chunking needed at 64^3.
const VER = '1.27.0';
const CDN = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${VER}/dist/`;
const post = (type, data, tr) => self.postMessage({ type, data }, tr || []);

let ort = null, session = null, device = null, P = 64;

self.onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.type === 'init') {
      P = m.patch;
      ort = await import(CDN + 'ort.webgpu.min.mjs');
      ort.env.wasm.wasmPaths = CDN;
      ort.env.logLevel = 'error';
      const t0 = performance.now();
      session = await ort.InferenceSession.create(m.modelUrl, { executionProviders: ['webgpu'] });
      device = ort.env.webgpu.device;
      const loadMs = Math.round(performance.now() - t0);
      // warm run (compile shaders) on zeros
      const warm = new ort.Tensor('float32', new Float32Array(8 * P * P * P), [1, 8, P, P, P]);
      const w0 = performance.now();
      await session.run({ input: warm });
      if (device) await device.queue.onSubmittedWorkDone();
      post('ready', { loadMs, warmMs: Math.round(performance.now() - w0), model: m.modelUrl });
    } else if (m.type === 'infer') {
      const feeds = { input: new ort.Tensor('float32', new Float32Array(m.input), [1, 8, P, P, P]) };
      const t0 = performance.now();
      const out = await session.run(feeds);
      if (device) await device.queue.onSubmittedWorkDone();
      const runMs = Math.round(performance.now() - t0);
      const lg = out['logits'], data = lg.data, D3 = P * P * P;
      const mask = new Uint8Array(D3);
      let fg = 0;
      for (let i = 0; i < D3; i++) { if (data[D3 + i] > data[i]) { mask[i] = 1; fg++; } }
      post('result', { runMs, fg, seq: m.seq, lo: m.lo, mask: mask.buffer }, [mask.buffer]);
    }
  } catch (err) {
    post('error', String((err && err.message) || err));
  }
};
