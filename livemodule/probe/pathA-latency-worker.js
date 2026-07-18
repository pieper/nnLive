// Path A latency harness. Modes (query params on the page):
//   ?ep=wasm|webgpu &p=64|128 &iters=N      -> plain measure (trunk once + N perclick)
//   ?ep=webgpu &gc=1 &p=64|128              -> combined 'full' graph + graph capture + io-binding (single session)
//   ?profile=1                              -> plain webgpu, per-kernel profiling to console
//   ?diag=1                                 -> verbose session-create log (EP partition)
import * as ort from './ort/ort.webgpu.bundle.min.mjs';
ort.env.wasm.wasmPaths = self.location.origin + '/ort/';
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(8, navigator.hardwareConcurrency || 4) : 1;
ort.env.logLevel = 'error';

const f16 = (dims) => new ort.Tensor('float16', new Uint16Array(dims.reduce((a, b) => a * b, 1)), dims);
const f32 = (dims) => new ort.Tensor('float32', new Float32Array(dims.reduce((a, b) => a * b, 1)), dims);
let PREC = 'fp16';
const mk = (dims) => (PREC === 'fp32' ? f32(dims) : f16(dims));
const M = (base, P) => `models/pathA/${base}_${P}${PREC === 'fp32' ? '' : '_fp16'}.onnx`;

self.onmessage = async (e) => {
  const { ep, iters, diag, profile, gc } = e.data;
  const P = e.data.patch || 128;
  PREC = e.data.prec || 'fp16';
  const post = (m) => self.postMessage(m);
  const stats = (t) => { t.sort((a, b) => a - b); return { median: t[t.length >> 1], min: t[0] }; };
  try {
    if (profile) {
      ort.env.webgpu.profiling = { mode: 'default' };
      const perclick = await ort.InferenceSession.create(M('perclick', P), { executionProviders: ['webgpu'] });
      const trunk = await ort.InferenceSession.create(M('trunk', P), { executionProviders: ['webgpu'] });
      const tr = await trunk.run({ image: f16([1, 1, P, P, P]) });
      const feeds = { s0: tr['s0'], s1: tr['s1'], inter: f16([1, 7, P, P, P]) };
      await perclick.run(feeds); post({ log: '--- PROFILE RUN ---' }); await perclick.run(feeds);
      post({ done: true, ep: 'webgpu', patch: P, trunkMs: 0, perclickMedianMs: 0, perclickMinMs: 0, loadMs: 0 });
      return;
    }
    if (diag) {
      ort.env.logLevel = 'verbose';
      await ort.InferenceSession.create(M('perclick', P), { executionProviders: ['webgpu'] });
      post({ done: true, ep, patch: P, trunkMs: 0, perclickMedianMs: 0, perclickMinMs: 0, loadMs: 0 });
      return;
    }

    if (gc) {  // combined 'full' graph, single session, io-binding + graph capture (OWN device per ORT source)
      post({ log: `GC: full graph p=${P}, own-device io-binding + graph capture` });
      const adapter = await navigator.gpu.requestAdapter();
      const feats = adapter.features.has('shader-f16') ? ['shader-f16'] : [];
      const device = await adapter.requestDevice({ requiredFeatures: feats });
      const full = await ort.InferenceSession.create(M('full', P),
        { executionProviders: [{ name: 'webgpu', device }], enableGraphCapture: true, preferredOutputLocation: 'gpu-buffer' });
      const gpuBuf = (dims) => {
        const bytes = dims.reduce((a, b) => a * b, 1) * 2;
        const buffer = device.createBuffer({ size: Math.ceil(bytes / 4) * 4,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
        return ort.Tensor.fromGpuBuffer(buffer, { dataType: 'float16', dims });
      };
      const image = gpuBuf([1, 1, P, P, P]), inter = gpuBuf([1, 7, P, P, P]), logits = gpuBuf([1, 2, P, P, P]);
      const feeds = { image, inter }, fetches = { logits };
      await full.run(feeds, fetches); await logits.getData(true);   // capture
      await full.run(feeds, fetches); await logits.getData(true);   // warm
      const times = [];
      for (let i = 0; i < iters; i++) {
        const a = performance.now(); await full.run(feeds, fetches); await logits.getData(true);
        times.push(performance.now() - a); post({ log: `  full run ${i}: ${times[i].toFixed(1)}ms` });
      }
      const s = stats(times);
      post({ done: true, ep: 'webgpu-gc', patch: P, trunkMs: 0, loadMs: 0, perclickMedianMs: s.median, perclickMinMs: s.min });
      return;
    }

    // plain measurement (works on both EPs; webgpu here is NO graph capture)
    const opt = { executionProviders: [ep] };
    let t0 = performance.now();
    const trunk = await ort.InferenceSession.create(M('trunk', P), opt);
    const perclick = await ort.InferenceSession.create(M('perclick', P), opt);
    const loadMs = performance.now() - t0;
    post({ log: `loaded ${loadMs | 0}ms (ep=${ep} p=${P} prec=${PREC})` });
    const image = mk([1, 1, P, P, P]);
    let tt = performance.now();
    const tr = await trunk.run({ image });
    const trunkMs = performance.now() - tt;
    const s0 = tr['s0'], s1 = tr['s1'], inter = mk([1, 7, P, P, P]);
    await perclick.run({ s0, s1, inter }); await perclick.run({ s0, s1, inter });
    const times = [];
    for (let i = 0; i < iters; i++) {
      const a = performance.now(); await perclick.run({ s0, s1, inter });
      times.push(performance.now() - a); post({ log: `  perclick ${i}: ${times[i].toFixed(1)}ms` });
    }
    const s = stats(times);
    post({ done: true, ep, patch: P, loadMs, trunkMs, perclickMedianMs: s.median, perclickMinMs: s.min });
  } catch (err) {
    post({ error: String((err && err.message) || err), ep });
  }
};
