// Minimal ONNX-graph executor over custom WGSL kernels (fp16). Runs the perclick graph
// (Conv/InstanceNorm/LeakyRelu/Add/Concat/AvgPool/MaxPool/Resize) in batched command encoders.
// Ops: Conv = V4 vec4 register-blocked (generalized K∈{1,3}, S∈{1,2}, pad, bias).
export const U = GPUBufferUsage;
export const f32tof16 = (() => { const b = new ArrayBuffer(4), f = new Float32Array(b), i = new Uint32Array(b);
  return x => { f[0] = x; const bits = i[0]; const s = (bits >>> 16) & 0x8000; let e = (bits >>> 23) & 0xff; let m = bits & 0x7fffff;
    if (e === 255) return s | 0x7c00 | (m ? 0x200 : 0); e = e - 127 + 15; if (e >= 31) return s | 0x7c00;
    if (e <= 0) { if (e < -10) return s; m |= 0x800000; return s | (m >>> (14 - e)); } return s | (e << 10) | (m >>> 13); }; })();
export const f16tof32 = h => { const s = (h & 0x8000) ? -1 : 1; const e = (h >> 10) & 0x1f; const m = h & 0x3ff;
  if (e === 0) return s * Math.pow(2, -14) * (m / 1024); if (e === 31) return m ? NaN : s * Infinity; return s * Math.pow(2, e - 15) * (1 + m / 1024); };
export const toF16 = a => { const u = new Uint16Array(a.length); for (let i = 0; i < a.length; i++) u[i] = f32tof16(a[i]); return u; };
export const fromF16 = u => { const a = new Float32Array(u.length); for (let i = 0; i < u.length; i++) a[i] = f16tof32(u[i]); return a; };

export async function initDevice() {
  const ad = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  const dev = await ad.requestDevice({ requiredFeatures: ['shader-f16'],
    requiredLimits: { maxBufferSize: ad.limits.maxBufferSize, maxStorageBufferBindingSize: ad.limits.maxStorageBufferBindingSize } });
  return { dev, info: ad.info || {} };
}

// ---------- WGSL kernels ----------
const CONV = `
struct D{Ci:u32,Co:u32,OD:u32,OH:u32,OW:u32,ID:u32,IH:u32,IW:u32,K:u32,S:u32,pad:u32,aux:u32};
@group(0)@binding(0) var<storage,read> inp:array<f16>; @group(0)@binding(1) var<storage,read> wgt:array<f16>;
@group(0)@binding(2) var<storage,read> bias:array<f16>; @group(0)@binding(3) var<storage,read_write> outp:array<f16>;
@group(0)@binding(4) var<uniform> d:D;
var<workgroup> As:array<vec4<f16>,256>; var<workgroup> Bs:array<vec4<f16>,256>;
fn gB(gk:u32,oz:u32,oy:u32,ox:u32)->f16{
  let taps=d.K*d.K*d.K; let ci=gk/taps; let tap=gk%taps; let kz=tap/(d.K*d.K); let ky=(tap/d.K)%d.K; let kx=tap%d.K;
  let iz=i32(oz*d.S)+i32(kz)-i32(d.pad); let iy=i32(oy*d.S)+i32(ky)-i32(d.pad); let ix=i32(ox*d.S)+i32(kx)-i32(d.pad);
  if(iz<0||iz>=i32(d.ID)||iy<0||iy>=i32(d.IH)||ix<0||ix>=i32(d.IW)){return f16(0);}
  return inp[((ci*d.ID+u32(iz))*d.IH+u32(iy))*d.IW+u32(ix)]; }
@compute @workgroup_size(16,16)
fn main(@builtin(workgroup_id) wid:vec3<u32>,@builtin(local_invocation_id) lid:vec3<u32>){
  let Nvox=d.OD*d.OH*d.OW; let Ktot=d.Ci*d.K*d.K*d.K; let Ktiles=(Ktot+15u)/16u;
  let tid=lid.y*16u+lid.x; let rowBase=wid.y*64u; let ntile=wid.z*d.aux+wid.x; let colBase=ntile*64u;
  let OHW=d.OH*d.OW; let ar=lid.y*4u; let br=lid.x*4u;
  var acc:array<f32,16>; for(var i=0u;i<16u;i++){acc[i]=0.0;}
  for(var kk:u32=0u;kk<Ktiles;kk++){
    { let m=tid/4u;let k4=tid%4u;let gm=rowBase+m;let base=kk*16u+k4*4u; var v=vec4<f16>(0.0,0.0,0.0,0.0);
      if(gm<d.Co){let o=gm*Ktot+base; if(base<Ktot){v.x=wgt[o];} if(base+1u<Ktot){v.y=wgt[o+1u];} if(base+2u<Ktot){v.z=wgt[o+2u];} if(base+3u<Ktot){v.w=wgt[o+3u];}} As[tid]=v; }
    { let n=tid/4u;let k4=tid%4u;let gn=colBase+n;let base=kk*16u+k4*4u; var v=vec4<f16>(0.0,0.0,0.0,0.0);
      if(gn<Nvox){let oz=gn/OHW;let rem=gn%OHW;let oy=rem/d.OW;let ox=rem%d.OW;
        v.x=gB(base,oz,oy,ox);v.y=gB(base+1u,oz,oy,ox);v.z=gB(base+2u,oz,oy,ox);v.w=gB(base+3u,oz,oy,ox);} Bs[tid]=v; }
    workgroupBarrier();
    for(var k4=0u;k4<4u;k4++){ let a0=As[(ar+0u)*4u+k4];let a1=As[(ar+1u)*4u+k4];let a2=As[(ar+2u)*4u+k4];let a3=As[(ar+3u)*4u+k4];
      let b0=Bs[(br+0u)*4u+k4];let b1=Bs[(br+1u)*4u+k4];let b2=Bs[(br+2u)*4u+k4];let b3=Bs[(br+3u)*4u+k4];
      acc[0]+=f32(dot(a0,b0));acc[1]+=f32(dot(a0,b1));acc[2]+=f32(dot(a0,b2));acc[3]+=f32(dot(a0,b3));
      acc[4]+=f32(dot(a1,b0));acc[5]+=f32(dot(a1,b1));acc[6]+=f32(dot(a1,b2));acc[7]+=f32(dot(a1,b3));
      acc[8]+=f32(dot(a2,b0));acc[9]+=f32(dot(a2,b1));acc[10]+=f32(dot(a2,b2));acc[11]+=f32(dot(a2,b3));
      acc[12]+=f32(dot(a3,b0));acc[13]+=f32(dot(a3,b1));acc[14]+=f32(dot(a3,b2));acc[15]+=f32(dot(a3,b3)); }
    workgroupBarrier();
  }
  for(var i=0u;i<4u;i++){let gm=rowBase+ar+i; if(gm>=d.Co){continue;}
    for(var j=0u;j<4u;j++){let gn=colBase+br+j; if(gn>=Nvox){continue;} outp[gm*Nvox+gn]=f16(acc[i*4u+j])+bias[gm];}}
}`;
const IN_STATS = `struct P{C:u32,N:u32}; @group(0)@binding(0) var<storage,read> x:array<f16>;
@group(0)@binding(1) var<storage,read_write> st:array<f32>; @group(0)@binding(2) var<uniform> p:P;
var<workgroup> ss:array<f32,256>; var<workgroup> sq:array<f32,256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid:vec3<u32>,@builtin(local_invocation_id) lid:vec3<u32>){
  let c=wid.x; let base=c*p.N; var s=0.0; var q=0.0;
  for(var i=lid.x;i<p.N;i+=256u){ let v=f32(x[base+i]); s+=v; q+=v*v; }
  ss[lid.x]=s; sq[lid.x]=q; workgroupBarrier();
  for(var t=128u;t>0u;t>>=1u){ if(lid.x<t){ss[lid.x]+=ss[lid.x+t];sq[lid.x]+=sq[lid.x+t];} workgroupBarrier(); }
  if(lid.x==0u){ let m=ss[0]/f32(p.N); let vr=sq[0]/f32(p.N)-m*m; st[c*2u]=m; st[c*2u+1u]=inverseSqrt(max(vr,0.0)+1e-5); } }`;
const IN_APPLY = `struct P{C:u32,N:u32,gx:u32}; @group(0)@binding(0) var<storage,read> x:array<f16>;
@group(0)@binding(1) var<storage,read> st:array<f32>; @group(0)@binding(2) var<storage,read> sc:array<f16>;
@group(0)@binding(3) var<storage,read> bi:array<f16>; @group(0)@binding(4) var<storage,read_write> y:array<f16>;
@group(0)@binding(5) var<uniform> p:P;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid:vec3<u32>,@builtin(local_invocation_id) lid:vec3<u32>){
  let idx=(wid.y*p.gx+wid.x)*256u+lid.x; if(idx>=p.C*p.N){return;} let c=idx/p.N;
  y[idx]=f16((f32(x[idx])-st[c*2u])*st[c*2u+1u]*f32(sc[c])+f32(bi[c])); }`;
const IN_APPLY_LEAKY = IN_APPLY.replace(
  'y[idx]=f16((f32(x[idx])-st[c*2u])*st[c*2u+1u]*f32(sc[c])+f32(bi[c])); }',
  'let v=(f32(x[idx])-st[c*2u])*st[c*2u+1u]*f32(sc[c])+f32(bi[c]); y[idx]=f16(select(v*0.01,v,v>0.0)); }');
const ADD = `struct P{n:u32,gx:u32}; @group(0)@binding(0) var<storage,read> a:array<f16>;
@group(0)@binding(1) var<storage,read> b:array<f16>; @group(0)@binding(2) var<storage,read_write> y:array<f16>;
@group(0)@binding(3) var<uniform> p:P;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid:vec3<u32>,@builtin(local_invocation_id) lid:vec3<u32>){
  let i=(wid.y*p.gx+wid.x)*256u+lid.x; if(i>=p.n){return;} y[i]=a[i]+b[i]; }`;
const LEAKY = `struct P{n:u32,gx:u32}; @group(0)@binding(0) var<storage,read> a:array<f16>;
@group(0)@binding(1) var<storage,read_write> y:array<f16>; @group(0)@binding(2) var<uniform> p:P;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid:vec3<u32>,@builtin(local_invocation_id) lid:vec3<u32>){
  let i=(wid.y*p.gx+wid.x)*256u+lid.x; if(i>=p.n){return;} let v=f32(a[i]); y[i]=f16(select(v*0.01,v,v>0.0)); }`;
const POOL = (isMax) => `struct D{C:u32,OD:u32,OH:u32,OW:u32,ID:u32,IH:u32,IW:u32,K:u32,S:u32,gx:u32};
@group(0)@binding(0) var<storage,read> x:array<f16>; @group(0)@binding(1) var<storage,read_write> y:array<f16>;
@group(0)@binding(2) var<uniform> d:D;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid:vec3<u32>,@builtin(local_invocation_id) lid:vec3<u32>){
  let idx=(wid.y*d.gx+wid.x)*256u+lid.x; let Nout=d.C*d.OD*d.OH*d.OW; if(idx>=Nout){return;}
  let OHW=d.OH*d.OW; let c=idx/(d.OD*OHW); let r=idx%(d.OD*OHW); let oz=r/OHW; let rr=r%OHW; let oy=rr/d.OW; let ox=rr%d.OW;
  var acc=${isMax ? 'f16(-65504.0)' : 'f32(0.0)'};
  for(var dz=0u;dz<d.K;dz++){for(var dy=0u;dy<d.K;dy++){for(var dx=0u;dx<d.K;dx++){
    let iz=oz*d.S+dz; let iy=oy*d.S+dy; let ix=ox*d.S+dx;
    let v=x[((c*d.ID+iz)*d.IH+iy)*d.IW+ix]; ${isMax ? 'acc=max(acc,v);' : 'acc+=f32(v);'} }}}
  y[idx]=${isMax ? 'acc' : 'f16(acc/f32(d.K*d.K*d.K))'}; }`;
const RESIZE = `struct D{C:u32,OD:u32,OH:u32,OW:u32,ID:u32,IH:u32,IW:u32,gx:u32};
@group(0)@binding(0) var<storage,read> x:array<f16>; @group(0)@binding(1) var<storage,read_write> y:array<f16>;
@group(0)@binding(2) var<uniform> d:D;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid:vec3<u32>,@builtin(local_invocation_id) lid:vec3<u32>){
  let idx=(wid.y*d.gx+wid.x)*256u+lid.x; let Nout=d.C*d.OD*d.OH*d.OW; if(idx>=Nout){return;}
  let OHW=d.OH*d.OW; let c=idx/(d.OD*OHW); let r=idx%(d.OD*OHW); let oz=r/OHW; let rr=r%OHW; let oy=rr/d.OW; let ox=rr%d.OW;
  let iz=(oz*d.ID)/d.OD; let iy=(oy*d.IH)/d.OH; let ix=(ox*d.IW)/d.OW;   // nearest/floor asymmetric
  y[idx]=x[((c*d.ID+iz)*d.IH+iy)*d.IW+ix]; }`;

export function makeRunner(dev) {
  const pc = new Map();
  const pipe = src => { if (pc.has(src)) return pc.get(src); const m = dev.createShaderModule({ code: 'enable f16;\n' + src });
    const p = dev.createComputePipeline({ layout: 'auto', compute: { module: m, entryPoint: 'main' } }); pc.set(src, p); return p; };
  const uni = arr => { const b = dev.createBuffer({ size: Math.max(16, arr.length * 4), usage: U.UNIFORM | U.COPY_DST }); dev.queue.writeBuffer(b, 0, new Uint32Array(arr)); return b; };
  const pass = (enc, src, bufs, wg) => { const p = pipe(src);
    const bg = dev.createBindGroup({ layout: p.getBindGroupLayout(0), entries: bufs.map((b, i) => ({ binding: i, resource: { buffer: b } })) });
    const cp = enc.beginComputePass(); cp.setPipeline(p); cp.setBindGroup(0, bg); cp.dispatchWorkgroups(wg[0], wg[1] || 1, wg[2] || 1); cp.end(); };
  const grid = n => { const nWG = Math.ceil(n / 256), gx = Math.min(65535, nWG); return { gx, wg: [gx, Math.ceil(nWG / gx), 1] }; };
  return { pipe, uni, pass, grid, CONV, IN_STATS, IN_APPLY, IN_APPLY_LEAKY, ADD, LEAKY, POOL, RESIZE };
}
