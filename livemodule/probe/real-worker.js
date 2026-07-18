// Chunked WebGPU teacher on REAL data. Streams N sub-models: load -> run -> DRAIN -> dispose, one at a
// time. Tensors crossing a chunk boundary pass through CPU (default output location), NOT gpu-buffer:
//   * only ONE chunk's activations live on the GPU at a time -> peak WIRED memory = a single chunk, so
//     the big full-resolution skip sits in cheap CPU RAM instead of pinned GPU memory (the 16GB ceiling);
//   * CPU tensors survive session.release(), so the frugal per-chunk dispose is valid (gpu-buffer outputs
//     are owned by their session and die on release -> the cross-device "Buffer ... cannot be used with
//     Device" validation error we hit). ORT re-uploads each chunk's inputs on run().
// Reports before AND after each chunk so any stall is attributable to a specific sub-model.
const VER = '1.27.0';
const CDN = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${VER}/dist/`;
const post = (type, data, tr) => self.postMessage({ type, data }, tr || []);

self.onmessage = async (e) => {
  const { baseUrl, manifest, inputBuffer, inputDims } = e.data;
  const N = manifest.chunks.length;
  try {
    const ort = await import(CDN + 'ort.webgpu.min.mjs');
    ort.env.wasm.wasmPaths = CDN;
    ort.env.logLevel = 'error';

    let device = null;
    const inName = manifest.chunks[0].inputs[0];
    const tensors = { [inName]: new ort.Tensor('float32', new Float32Array(inputBuffer), inputDims) };
    const times = [];
    const start = performance.now();

    for (let k = 0; k < N; k++) {
      const c = manifest.chunks[k];
      post('chunkstart', { k, n: N, file: c.file });          // visible even if it stalls here
      const lt = performance.now();
      const s = await ort.InferenceSession.create(baseUrl + c.file, {
        executionProviders: ['webgpu'],
        // NO gpu-buffer: outputs come back on CPU so they survive release() and never bind cross-device.
      });
      device = ort.env.webgpu.device || device;
      const loadMs = Math.round(performance.now() - lt);
      const feeds = {};
      for (const nm of c.inputs) feeds[nm] = tensors[nm];
      const c0 = performance.now();
      const out = await s.run(feeds);
      try { if (device) await device.queue.onSubmittedWorkDone(); } catch (_) {} // DRAIN -> repaint
      const dt = Math.round(performance.now() - c0);
      times.push(dt);
      for (const nm of c.outputs) tensors[nm] = out[nm];       // CPU tensors: valid after release()
      try { await s.release(); } catch (_) {}                  // free this chunk's GPU memory before the next
      post('chunkdone', { k, n: N, dt, loadMs, elapsed: Math.round(performance.now() - start) });
      await new Promise(r => setTimeout(r, 0));                 // YIELD to the event loop
    }

    const total = Math.round(performance.now() - start);
    const logits = tensors['logits'];                          // cpu fp32 [1,2,D,D,D]
    const dims = logits.dims;
    const data = logits.data;
    const D3 = dims[2] * dims[3] * dims[4];
    const mask = new Uint8Array(D3);                           // argmax over the 2 class channels
    let fg = 0;
    for (let i = 0; i < D3; i++) { if (data[D3 + i] > data[i]) { mask[i] = 1; fg++; } }
    post('done', { total, times, dims, fg, mask: mask.buffer }, [mask.buffer]);
  } catch (err) {
    post('error', String((err && err.message) || err));
  }
};
