// Progressive multi-resolution custom-WGSL worker: holds a trunk8+perclick Net pair per resolution.
// Main thread drives coarse->fine (64 then 128, 192 placeholder). Trunk encodes once per box/res.
import { initDevice, makeRunner, Net } from './wgpu-net.js?v=9';
let dev, R; const nets = {};   // P -> {trunk, perclick, N}

self.onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.type === 'init') {
      const t0 = performance.now();
      ({ dev } = await initDevice()); R = makeRunner(dev);
      for (const P of m.res) {
        const N = P * P * P;
        const trunk = new Net(dev, R); await trunk.load(`models/pathA/trunk8_${P}.graph.json`, `models/pathA/trunk8_${P}.weights.bin`);
        const perclick = new Net(dev, R); await perclick.load(`models/pathA/perclick_${P}.graph.json`, `models/pathA/perclick_${P}.weights.bin`);
        trunk.setInputData('img8', new Float32Array(8 * N)); trunk.run();          // warm
        perclick.setInputBuffer('s0', trunk.outBuf('s0')); perclick.setInputBuffer('s1', trunk.outBuf('s1'));
        perclick.setInputData('inter', new Float32Array(7 * N)); perclick.run(); await perclick.read('logits');
        nets[P] = { trunk, perclick, N };
        self.postMessage({ type: 'loaded', res: P });
      }
      await dev.queue.onSubmittedWorkDone();
      self.postMessage({ type: 'ready', ms: Math.round(performance.now() - t0) });
      return;
    }
    if (m.type === 'encode') {                 // {res, image} — z-scored P^3 crop (1 ch)
      const n = nets[m.res]; const img8 = new Float32Array(8 * n.N); img8.set(new Float32Array(m.image), 0);
      const t0 = performance.now();
      n.trunk.setInputData('img8', img8); n.trunk.run(); await dev.queue.onSubmittedWorkDone();
      n.perclick.setInputBuffer('s0', n.trunk.outBuf('s0')); n.perclick.setInputBuffer('s1', n.trunk.outBuf('s1'));
      self.postMessage({ type: 'encoded', res: m.res, ms: Math.round(performance.now() - t0) });
      return;
    }
    if (m.type === 'infer') {                   // {res, inter}
      const n = nets[m.res];
      n.perclick.setInputData('inter', new Float32Array(m.inter));
      const t0 = performance.now();
      n.perclick.run(); const lg = await n.perclick.read('logits');
      const mask = new Uint8Array(n.N); let vox = 0;
      // DIAG: logits stats (bg=ch0, fg=ch1)
      let mn0=1e30,mx0=-1e30,mn1=1e30,mx1=-1e30,nan=0,s0=0,s1=0;
      for (let i = 0; i < n.N; i++) { const a=lg[i], b=lg[n.N+i];
        if(Number.isNaN(a)||Number.isNaN(b)) nan++;
        if(a<mn0)mn0=a; if(a>mx0)mx0=a; if(b<mn1)mn1=b; if(b>mx1)mx1=b; s0+=a; s1+=b;
        const fg = b > a ? 1 : 0; mask[i] = fg; vox += fg; }
      const diag={nan, bg:[mn0,mx0,s0/n.N], fg:[mn1,mx1,s1/n.N]};
      self.postMessage({ type: 'result', res: m.res, ms: Math.round(performance.now() - t0), vox, diag, mask: mask.buffer }, [mask.buffer]);
      return;
    }
  } catch (err) { self.postMessage({ type: 'error', msg: String((err && err.message) || err) }); }
};
