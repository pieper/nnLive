// Student U-Net on real data: ONE WebGPU session, no chunking (it fits). Times a warm forward pass
// (first run compiles shaders, so we time the SECOND run for steady-state interactive latency).
const VER = '1.27.0';
const CDN = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${VER}/dist/`;
const post = (type, data, tr) => self.postMessage({ type, data }, tr || []);

self.onmessage = async (e) => {
  const { modelUrl, inputBuffer, inputDims } = e.data;
  try {
    const ort = await import(CDN + 'ort.webgpu.min.mjs');
    ort.env.wasm.wasmPaths = CDN;
    ort.env.logLevel = 'error';

    post('status', { msg: 'creating session (loading weights to GPU)…' });
    const lt = performance.now();
    const s = await ort.InferenceSession.create(modelUrl, { executionProviders: ['webgpu'] });
    const loadMs = Math.round(performance.now() - lt);
    const dev = ort.env.webgpu.device;
    const feeds = { input: new ort.Tensor('float32', new Float32Array(inputBuffer), inputDims) };

    post('status', { msg: 'warmup run (shader compile)…' });
    let o = await s.run(feeds);
    if (dev) await dev.queue.onSubmittedWorkDone();
    const warmMs = Math.round(performance.now() - lt - loadMs);

    post('status', { msg: 'timed run…' });
    const c0 = performance.now();
    o = await s.run(feeds);
    if (dev) await dev.queue.onSubmittedWorkDone();
    const runMs = Math.round(performance.now() - c0);

    const logits = o['logits'];
    const dims = logits.dims, data = logits.data;
    const D3 = dims[2] * dims[3] * dims[4];
    const mask = new Uint8Array(D3);
    let fg = 0;
    for (let i = 0; i < D3; i++) { if (data[D3 + i] > data[i]) { mask[i] = 1; fg++; } }
    await s.release();
    post('done', { loadMs, warmMs, runMs, dims, fg, mask: mask.buffer }, [mask.buffer]);
  } catch (err) {
    post('error', String((err && err.message) || err));
  }
};
