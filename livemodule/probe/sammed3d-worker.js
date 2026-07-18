// SAM-Med3D in the browser: encode-once (heavy ViT on wasm — safe, no GPU-watchdog risk on M4),
// decode-per-click (tiny transformer on WebGPU — ~ms). The encoder image embedding is cached
// after one encode; every click just re-runs the decoder with the accumulated points + the
// previous low-res mask (SAM-Med3D's own iterative-refinement input) so the mask grows and
// respects earlier clicks instead of overwriting them.
const ORT_DIR = self.location.origin + '/ort/';   // self-hosted ORT (COEP-safe, offline)
const post = (type, data, tr) => self.postMessage({ type, data }, tr || []);
const P = 128, EMB = 8, LOW = 32, C = 384;

let ort = null, encSess = null, decSess = null, device = null, embData = null;

self.onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.type === 'init') {
      ort = await import(ORT_DIR + 'ort.webgpu.bundle.min.mjs');
      ort.env.wasm.wasmPaths = ORT_DIR;
      ort.env.logLevel = 'error';
      ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(8, navigator.hardwareConcurrency || 4) : 1;
      const t0 = performance.now();
      encSess = await ort.InferenceSession.create(m.encoderUrl, { executionProviders: ['wasm'] });
      const encLoad = Math.round(performance.now() - t0);
      const t1 = performance.now();
      // webgpu for the bulk; wasm fallback catches any op the WebGPU EP doesn't implement
      decSess = await ort.InferenceSession.create(m.decoderUrl, { executionProviders: ['webgpu', 'wasm'] });
      device = ort.env.webgpu.device || null;
      const decLoad = Math.round(performance.now() - t1);
      // warm the decoder (compile shaders) on a dummy embedding so the first real click is steady-state
      const warmEmb = new ort.Tensor('float32', new Float32Array(C * EMB * EMB * EMB), [1, C, EMB, EMB, EMB]);
      const wf = {
        image_embeddings: warmEmb,
        point_coords: new ort.Tensor('float32', new Float32Array([64, 64, 64]), [1, 1, 3]),
        point_labels: new ort.Tensor('float32', new Float32Array([1]), [1, 1]),
        prev_mask: new ort.Tensor('float32', new Float32Array(LOW * LOW * LOW), [1, 1, LOW, LOW, LOW]),
        has_mask: new ort.Tensor('float32', new Float32Array([0]), [1]),
      };
      const tw = performance.now();
      await decSess.run(wf);
      if (device) await device.queue.onSubmittedWorkDone();
      post('ready', { encLoad, decLoad, warmMs: Math.round(performance.now() - tw), threads: ort.env.wasm.numThreads });

    } else if (m.type === 'encode') {
      const t0 = performance.now();
      const inp = new ort.Tensor('float32', new Float32Array(m.crop), [1, 1, P, P, P]);
      const out = await encSess.run({ image: inp });
      // detach embedding to a plain Float32Array we re-wrap for each decode feed
      embData = Float32Array.from(out['image_embeddings'].data);
      post('encoded', { ms: Math.round(performance.now() - t0), emb: embData.buffer.slice(0) });

    } else if (m.type === 'infer') {
      if (!embData) { post('error', 'no embedding yet — encode first'); return; }
      const t0 = performance.now();
      const feeds = {
        image_embeddings: new ort.Tensor('float32', embData, [1, C, EMB, EMB, EMB]),
        point_coords: new ort.Tensor('float32', new Float32Array(m.coords), [1, m.n, 3]),
        point_labels: new ort.Tensor('float32', new Float32Array(m.labels), [1, m.n]),
        prev_mask: new ort.Tensor('float32', new Float32Array(m.prev), [1, 1, LOW, LOW, LOW]),
        has_mask: new ort.Tensor('float32', new Float32Array([m.hasMask]), [1]),
      };
      const out = await decSess.run(feeds);
      if (device) await device.queue.onSubmittedWorkDone();
      const logits = Float32Array.from(out['low_res_logits'].data);   // [32^3]
      post('result', { ms: Math.round(performance.now() - t0), seq: m.seq, logits: logits.buffer }, [logits.buffer]);
    }
  } catch (err) {
    post('error', String((err && err.message) || err));
  }
};
