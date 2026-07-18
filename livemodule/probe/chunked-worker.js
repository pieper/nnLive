// Chunked WebGPU inference worker with per-chunk feedback. Runs N sub-models in sequence, draining
// the GPU between each (so WindowServer can repaint) and reporting before AND after every chunk, so
// a stall is visible and attributable to a specific chunk. Single instrumented run (no silent warmup).
const VER = '1.27.0';
const CDN = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${VER}/dist/`;
const post = (type, data) => self.postMessage({ type, data });

self.onmessage = async (e) => {
  const { baseUrl, manifest, inputDims } = e.data;
  const N = manifest.chunks.length;
  try {
    const ort = await import(CDN + 'ort.webgpu.min.mjs');
    ort.env.wasm.wasmPaths = CDN;
    ort.env.logLevel = 'error';

    // Frugal streaming: load -> run -> dispose ONE session at a time so peak memory stays low
    // (16 concurrent sessions exhausted the WASM heap). GPU intermediates persist across dispose.
    let device = null;
    const inName = manifest.chunks[0].inputs[0];
    const nIn = inputDims.reduce((a, b) => a * b, 1);
    const tensors = { [inName]: new ort.Tensor('float32', new Float32Array(nIn), inputDims) };
    const times = [];
    const start = performance.now();
    for (let k = 0; k < N; k++) {
      const c = manifest.chunks[k];
      post('chunkstart', { k, n: N });                 // visible even if it stalls here
      const lt = performance.now();
      const s = await ort.InferenceSession.create(baseUrl + c.file, {
        executionProviders: ['webgpu'],
        preferredOutputLocation: 'gpu-buffer',
      });
      device = device || ort.env.webgpu.device;
      const loadMs = Math.round(performance.now() - lt);
      const feeds = {};
      for (const nm of c.inputs) feeds[nm] = tensors[nm];
      const c0 = performance.now();
      const out = await s.run(feeds);
      await device.queue.onSubmittedWorkDone();          // DRAIN -> compositor repaints
      const dt = Math.round(performance.now() - c0);
      times.push(dt);
      for (const nm of c.outputs) tensors[nm] = out[nm];
      try { await s.release(); } catch (_) {}            // free this chunk's session before the next
      post('chunkdone', { k, n: N, dt, loadMs, elapsed: Math.round(performance.now() - start) });
      await new Promise(r => setTimeout(r, 0));           // YIELD
    }
    const total = Math.round(performance.now() - start);
    const last = tensors[manifest.chunks[N - 1].outputs[0]];
    post('done', { total, times, outDims: last.dims });
  } catch (err) {
    post('error', String((err && err.message) || err));
  }
};
