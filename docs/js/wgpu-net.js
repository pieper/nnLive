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
// small-M conv variant (BM=32,BN=128,TM=2,TN=8) for low-channel high-res layers (decoder)
const CONV_VSM = `
struct D{Ci:u32,Co:u32,OD:u32,OH:u32,OW:u32,ID:u32,IH:u32,IW:u32,K:u32,S:u32,pad:u32,aux:u32};
@group(0)@binding(0) var<storage,read> inp:array<f16>; @group(0)@binding(1) var<storage,read> wgt:array<f16>;
@group(0)@binding(2) var<storage,read> bias:array<f16>; @group(0)@binding(3) var<storage,read_write> outp:array<f16>;
@group(0)@binding(4) var<uniform> d:D;
var<workgroup> As:array<vec4<f16>,128>; var<workgroup> Bs:array<vec4<f16>,512>;
fn gB(gk:u32,oz:u32,oy:u32,ox:u32)->f16{ let taps=d.K*d.K*d.K; let ci=gk/taps; let tap=gk%taps;
  let kz=tap/(d.K*d.K); let ky=(tap/d.K)%d.K; let kx=tap%d.K;
  let iz=i32(oz*d.S)+i32(kz)-i32(d.pad); let iy=i32(oy*d.S)+i32(ky)-i32(d.pad); let ix=i32(ox*d.S)+i32(kx)-i32(d.pad);
  if(iz<0||iz>=i32(d.ID)||iy<0||iy>=i32(d.IH)||ix<0||ix>=i32(d.IW)){return f16(0);}
  return inp[((ci*d.ID+u32(iz))*d.IH+u32(iy))*d.IW+u32(ix)]; }
@compute @workgroup_size(16,16)
fn main(@builtin(workgroup_id) wid:vec3<u32>,@builtin(local_invocation_id) lid:vec3<u32>){
  let Nvox=d.OD*d.OH*d.OW; let Ktot=d.Ci*d.K*d.K*d.K; let Ktiles=(Ktot+15u)/16u;
  let tid=lid.y*16u+lid.x; let rowBase=wid.y*32u; let ntile=wid.z*d.aux+wid.x; let colBase=ntile*128u;
  let OHW=d.OH*d.OW; let ar=lid.y*2u; let br=lid.x*8u;
  var acc:array<f32,16>; for(var i=0u;i<16u;i++){acc[i]=0.0;}
  for(var kk:u32=0u;kk<Ktiles;kk++){
    if(tid<128u){ let m=tid/4u;let k4=tid%4u;let gm=rowBase+m;let base=kk*16u+k4*4u; var v=vec4<f16>(0.0,0.0,0.0,0.0);
      if(gm<d.Co){let o=gm*Ktot+base; if(base<Ktot){v.x=wgt[o];} if(base+1u<Ktot){v.y=wgt[o+1u];} if(base+2u<Ktot){v.z=wgt[o+2u];} if(base+3u<Ktot){v.w=wgt[o+3u];}} As[tid]=v; }
    for(var li=0u;li<2u;li++){ let e=tid+li*256u;let n=e/4u;let k4=e%4u;let gn=colBase+n;let base=kk*16u+k4*4u; var v=vec4<f16>(0.0,0.0,0.0,0.0);
      if(gn<Nvox){let oz=gn/OHW;let rem=gn%OHW;let oy=rem/d.OW;let ox=rem%d.OW; v.x=gB(base,oz,oy,ox);v.y=gB(base+1u,oz,oy,ox);v.z=gB(base+2u,oz,oy,ox);v.w=gB(base+3u,oz,oy,ox);} Bs[e]=v; }
    workgroupBarrier();
    for(var k4=0u;k4<4u;k4++){ var af:array<vec4<f16>,2>; var bf:array<vec4<f16>,8>;
      for(var i=0u;i<2u;i++){af[i]=As[(ar+i)*4u+k4];} for(var j=0u;j<8u;j++){bf[j]=Bs[(br+j)*4u+k4];}
      for(var i=0u;i<2u;i++){for(var j=0u;j<8u;j++){acc[i*8u+j]+=f32(dot(af[i],bf[j]));}} }
    workgroupBarrier();
  }
  for(var i=0u;i<2u;i++){let gm=rowBase+ar+i; if(gm>=d.Co){continue;}
    for(var j=0u;j<8u;j++){let gn=colBase+br+j; if(gn>=Nvox){continue;} outp[gm*Nvox+gn]=f16(acc[i*8u+j])+bias[gm];}}
}`;
// Parameterized conv (generalizes CONV/CONV_VSM) for per-GPU autotuning. 16x16 workgroup; each thread owns a
// TMxTN output micro-tile; tile = 16TM x 16TN; KT=16 K-tiling; vec4<f16> shared staging; f32 accumulate.
// Same GEMM math for any (TM,TN) — a different (TM,TN) is a different tiling of the identical computation.
function genConv(TM, TN) {
  const AS = 64 * TM, BS = 64 * TN, ACC = TM * TN;
  let af = '', bf = '', fma = '', wr = '';
  for (let i = 0; i < TM; i++) af += `af[${i}u]=As[(ar+${i}u)*4u+k4];`;
  for (let j = 0; j < TN; j++) bf += `bf[${j}u]=Bs[(br+${j}u)*4u+k4];`;
  for (let i = 0; i < TM; i++) for (let j = 0; j < TN; j++) fma += `acc[${i * TN + j}u]+=f32(dot(af[${i}u],bf[${j}u]));`;
  for (let i = 0; i < TM; i++) for (let j = 0; j < TN; j++) wr += `{let gm=rowBase+ar+${i}u;let gn=colBase+br+${j}u;if(gm<d.Co&&gn<Nvox){outp[gm*Nvox+gn]=f16(acc[${i * TN + j}u])+bias[gm];}}`;
  return `struct D{Ci:u32,Co:u32,OD:u32,OH:u32,OW:u32,ID:u32,IH:u32,IW:u32,K:u32,S:u32,pad:u32,aux:u32};
@group(0)@binding(0) var<storage,read> inp:array<f16>; @group(0)@binding(1) var<storage,read> wgt:array<f16>;
@group(0)@binding(2) var<storage,read> bias:array<f16>; @group(0)@binding(3) var<storage,read_write> outp:array<f16>;
@group(0)@binding(4) var<uniform> d:D;
var<workgroup> As:array<vec4<f16>,${AS}>; var<workgroup> Bs:array<vec4<f16>,${BS}>;
fn gB(gk:u32,oz:u32,oy:u32,ox:u32)->f16{ let taps=d.K*d.K*d.K; let ci=gk/taps; let tap=gk%taps;
  let kz=tap/(d.K*d.K); let ky=(tap/d.K)%d.K; let kx=tap%d.K;
  let iz=i32(oz*d.S)+i32(kz)-i32(d.pad); let iy=i32(oy*d.S)+i32(ky)-i32(d.pad); let ix=i32(ox*d.S)+i32(kx)-i32(d.pad);
  if(iz<0||iz>=i32(d.ID)||iy<0||iy>=i32(d.IH)||ix<0||ix>=i32(d.IW)){return f16(0);}
  return inp[((ci*d.ID+u32(iz))*d.IH+u32(iy))*d.IW+u32(ix)]; }
@compute @workgroup_size(16,16)
fn main(@builtin(workgroup_id) wid:vec3<u32>,@builtin(local_invocation_id) lid:vec3<u32>){
  let Nvox=d.OD*d.OH*d.OW; let Ktot=d.Ci*d.K*d.K*d.K; let Ktiles=(Ktot+15u)/16u;
  let tid=lid.y*16u+lid.x; let rowBase=wid.y*${16 * TM}u; let ntile=wid.z*d.aux+wid.x; let colBase=ntile*${16 * TN}u;
  let OHW=d.OH*d.OW; let ar=lid.y*${TM}u; let br=lid.x*${TN}u;
  var acc:array<f32,${ACC}>; for(var i=0u;i<${ACC}u;i++){acc[i]=0.0;}
  for(var kk:u32=0u;kk<Ktiles;kk++){
    for(var e=tid;e<${AS}u;e+=256u){ let m=e/4u;let k4=e%4u;let gm=rowBase+m;let base=kk*16u+k4*4u; var v=vec4<f16>(0.0,0.0,0.0,0.0);
      if(gm<d.Co){let o=gm*Ktot+base; if(base<Ktot){v.x=wgt[o];} if(base+1u<Ktot){v.y=wgt[o+1u];} if(base+2u<Ktot){v.z=wgt[o+2u];} if(base+3u<Ktot){v.w=wgt[o+3u];}} As[e]=v; }
    for(var e=tid;e<${BS}u;e+=256u){ let n=e/4u;let k4=e%4u;let gn=colBase+n;let base=kk*16u+k4*4u; var v=vec4<f16>(0.0,0.0,0.0,0.0);
      if(gn<Nvox){let oz=gn/OHW;let rem=gn%OHW;let oy=rem/d.OW;let ox=rem%d.OW; v.x=gB(base,oz,oy,ox);v.y=gB(base+1u,oz,oy,ox);v.z=gB(base+2u,oz,oy,ox);v.w=gB(base+3u,oz,oy,ox);} Bs[e]=v; }
    workgroupBarrier();
    for(var k4=0u;k4<4u;k4++){ var af:array<vec4<f16>,${TM}>; var bf:array<vec4<f16>,${TN}>; ${af} ${bf} ${fma} }
    workgroupBarrier();
  }
  ${wr}
}`;
}
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
// fused residual add + LeakyReLU (ResEnc blocks): one pass instead of add-then-leaky, avoids an fp16 round-trip
const ADD_LEAKY = ADD.replace('y[i]=a[i]+b[i]; }', 'let v=f32(a[i])+f32(b[i]); y[i]=f16(select(v*0.01,v,v>0.0)); }');
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

// Reusable graph net: load graph.json + weights.bin once, run per input (pooled buffers,
// swappable graph inputs so a trunk's GPU outputs feed straight into perclick — no readback).
export class Net {
  constructor(dev, R) { this.dev = dev; this.R = R; this.ext = {}; this.inBuf = {}; this.convSrc = R.CONV; this.convTM = 4; this.convTN = 4; }
  async load(graphUrl, weightsUrl) {
    const dev = this.dev;
    this.graph = await (await fetch(graphUrl)).json();
    const wblob = new Uint16Array(await (await fetch(weightsUrl)).arrayBuffer());
    this.prod = s => s.reduce((a, b) => a * b, 1);
    this.mk = b => dev.createBuffer({ size: Math.max(16, b), usage: U.STORAGE | U.COPY_SRC | U.COPY_DST });
    this.W = {}; for (const [n, w] of Object.entries(this.graph.weights)) { const b = this.mk(w.numel * 2); dev.queue.writeBuffer(b, 0, wblob.subarray(w.offset, w.offset + w.numel)); this.W[n] = b; }
    this._plan();
    return this;
  }
  bytesOf(n) { return this.prod(this.graph.tensors[n] || this.W[n].shape) * 2; }
  _plan() {
    const g = this.graph, R = this.R;
    const inSet = new Set(g.inputs.map(i => i.name));
    const cons = {}; for (const nd of g.nodes) for (const i of nd.in) (cons[i] = cons[i] || []).push(nd);
    const skip = new Set(), fuseOut = {};       // fuse a single-consumer LeakyRelu into its InstanceNorm or Add producer
    for (const nd of g.nodes) if (nd.op === 'InstanceNormalization' || nd.op === 'Add') { const c = cons[nd.out[0]]; if (c && c.length === 1 && c[0].op === 'LeakyRelu') { skip.add(c[0]); fuseOut[nd.out[0]] = c[0].out[0]; } }
    const ops = []; for (const nd of g.nodes) { if (skip.has(nd)) continue; const fo = (nd.op === 'InstanceNormalization' || nd.op === 'Add') ? fuseOut[nd.out[0]] : undefined; ops.push({ nd, out: fo || nd.out[0], fused: !!fo }); }
    const lastUse = {}; ops.forEach((o, i) => { for (const t of o.nd.in) lastUse[t] = i; });
    g.outputs.forEach(o => lastUse[o.name] = 1e18);
    const T = {}, pool = []; this.zeroBias = {};
    const acquire = b => { let bi = -1; for (let i = 0; i < pool.length; i++) if (pool[i].size >= b && (bi < 0 || pool[i].size < pool[bi].size)) bi = i; return bi >= 0 ? pool.splice(bi, 1)[0] : this.mk(b); };
    const zB = C => (this.zeroBias[C] = this.zeroBias[C] || this.mk(C * 2));
    this.recs = [];
    ops.forEach((o, i) => {
      const nd = o.nd, tsr = g.tensors;
      if (!inSet.has(o.out)) T[o.out] = acquire(this.bytesOf(o.out));   // graph inputs are external (not pooled)
      const r = { op: nd.op, ins: nd.in, out: o.out };
      if (nd.op === 'Conv') { const os = tsr[o.out], is = tsr[nd.in[0]]; const Nvox = os[2] * os[3] * os[4], gx = Math.min(65535, Math.ceil(Nvox / 64));
        r.bias = nd.bias ? nd.in[2] : null; r.Co = nd.Co; r.cs = [nd.Ci, nd.Co, os[2], os[3], os[4], is[2], is[3], is[4], nd.K, nd.S, nd.pad]; r.u = R.uni([nd.Ci, nd.Co, os[2], os[3], os[4], is[2], is[3], is[4], nd.K, nd.S, nd.pad, gx]); r.wg = [gx, Math.ceil(nd.Co / 64), Math.ceil(Nvox / 64 / gx)]; }
      else if (nd.op === 'InstanceNormalization') { const s = tsr[o.out]; const C = s[1], N = s[2] * s[3] * s[4]; r.C = C; r.stats = this.mk(C * 2 * 4); r.u1 = R.uni([C, N]); const gd = R.grid(C * N); r.u2 = R.uni([C, N, gd.gx]); r.wg = gd.wg; r.applyK = o.fused ? R.IN_APPLY_LEAKY : R.IN_APPLY; }
      else if (nd.op === 'LeakyRelu') { const n = this.prod(tsr[o.out]); const gd = R.grid(n); r.u = R.uni([n, gd.gx]); r.wg = gd.wg; }
      else if (nd.op === 'Add') { const n = this.prod(tsr[o.out]); const gd = R.grid(n); r.u = R.uni([n, gd.gx]); r.wg = gd.wg; r.leaky = o.fused; }
      else if (nd.op === 'Concat') { r.parts = nd.in.map(x => this.bytesOf(x)); }
      else if (nd.op === 'AveragePool' || nd.op === 'MaxPool') { const os = tsr[o.out], is = tsr[nd.in[0]]; const Nout = os[1] * os[2] * os[3] * os[4]; const gd = R.grid(Nout); r.u = R.uni([os[1], os[2], os[3], os[4], is[2], is[3], is[4], nd.kernel, nd.S, gd.gx]); r.wg = gd.wg; r.mx = nd.op === 'MaxPool'; }
      else if (nd.op === 'Resize') { const os = tsr[o.out], is = tsr[nd.in[0]]; const Nout = os[1] * os[2] * os[3] * os[4]; const gd = R.grid(Nout); r.u = R.uni([os[1], os[2], os[3], os[4], is[2], is[3], is[4], gd.gx]); r.wg = gd.wg; }
      // capture buffers NOW (pooling later nulls T[name]); graph inputs stay swappable via {inp} markers
      r.inB = nd.in.map(n => inSet.has(n) ? { inp: n } : (this.W[n] || T[n]));
      r.outB = T[o.out]; if (r.bias) r.biasB = this.W[r.bias];
      this.recs.push(r);
      for (const t of nd.in) if (lastUse[t] === i && !(t in this.W) && !inSet.has(t) && T[t]) { pool.push(T[t]); T[t] = null; }
    });
    this.T = T; this.zB = zB;
  }
  setInputBuffer(name, buf) { this.ext[name] = buf; }
  setInputData(name, f32) { if (!this.inBuf[name]) this.inBuf[name] = this.mk(this.bytesOf(name)); this.dev.queue.writeBuffer(this.inBuf[name], 0, toF16(f32)); this.ext[name] = this.inBuf[name]; }
  outBuf(name) { return this.ext[name] || this.T[name] || this.outBufFor(name); }
  run() {
    const B = x => x && x.inp ? this.ext[x.inp] : x, R = this.R, enc = this.dev.createCommandEncoder();
    for (const r of this.recs) {
      const i = r.inB, out = r.outB;
      if (r.op === 'Conv') R.pass(enc, this.convSrc, [B(i[0]), B(i[1]), r.bias ? r.biasB : this.zB(r.Co), out, r.u], r.wg);
      else if (r.op === 'InstanceNormalization') { R.pass(enc, R.IN_STATS, [B(i[0]), r.stats, r.u1], [r.C, 1, 1]); R.pass(enc, r.applyK, [B(i[0]), r.stats, B(i[1]), B(i[2]), out, r.u2], r.wg); }
      else if (r.op === 'LeakyRelu') R.pass(enc, R.LEAKY, [B(i[0]), out, r.u], r.wg);
      else if (r.op === 'Add') R.pass(enc, r.leaky ? R.ADD_LEAKY : R.ADD, [B(i[0]), B(i[1]), out, r.u], r.wg);
      else if (r.op === 'Concat') { let off = 0; i.forEach((x, k) => { enc.copyBufferToBuffer(B(x), 0, out, off, r.parts[k]); off += r.parts[k]; }); }
      else if (r.op === 'AveragePool' || r.op === 'MaxPool') R.pass(enc, R.POOL(r.mx), [B(i[0]), out, r.u], r.wg);
      else if (r.op === 'Resize') R.pass(enc, R.RESIZE, [B(i[0]), out, r.u], r.wg);
    }
    this.dev.queue.submit([enc.finish()]);
  }
  outBufFor(name) { for (const r of this.recs) if (r.out === name) return r.outB; return this.ext[name]; }
  async read(name, floatOut = true) {
    const buf = this.outBufFor(name), n = this.prod(this.graph.tensors[name]);
    const rb = this.dev.createBuffer({ size: n * 2, usage: U.MAP_READ | U.COPY_DST });
    const e = this.dev.createCommandEncoder(); e.copyBufferToBuffer(buf, 0, rb, 0, n * 2); this.dev.queue.submit([e.finish()]);
    await rb.mapAsync(GPUMapMode.READ); const u = new Uint16Array(rb.getMappedRange().slice(0)); rb.unmap();
    return floatOut ? fromF16(u) : u;
  }
  // GPU argmax of a 2-channel logits tensor -> 1-byte-per-voxel mask (fg = ch1 > ch0), packed 4/u32.
  // Reads back N bytes instead of 2N f16 + a JS fromF16 on 2N values (the big per-click CPU cost).
  async argmaxMask(name) {
    const buf = this.outBufFor(name), N = this.prod(this.graph.tensors[name]) / 2, n4 = Math.ceil(N / 4);
    if (!this._maskBuf || this._maskN !== N) {
      this._maskBuf = this.mk(n4 * 4); this._maskN = N;
      const gd = this.R.grid(n4); this._maskU = this.R.uni([N, gd.gx]); this._maskWg = gd.wg;
    }
    const enc = this.dev.createCommandEncoder();
    this.R.pass(enc, this.R.ARGMAX, [buf, this._maskBuf, this._maskU], this._maskWg);
    const rb = this.dev.createBuffer({ size: n4 * 4, usage: U.MAP_READ | U.COPY_DST });
    enc.copyBufferToBuffer(this._maskBuf, 0, rb, 0, n4 * 4); this.dev.queue.submit([enc.finish()]);
    await rb.mapAsync(GPUMapMode.READ); const u = new Uint8Array(rb.getMappedRange().slice(0, N)); rb.unmap();
    return u;
  }
  // ---- per-GPU conv autotuning ----
  setConvConfig(TM, TN) {   // switch every conv record to a (TM,TN) tiling: kernel src + dispatch(gx,wg) + uniform aux
    this.convTM = TM; this.convTN = TN;
    this.convSrc = (TM === 4 && TN === 4) ? this.R.CONV : genConv(TM, TN);
    const tileR = 16 * TM, tileC = 16 * TN;
    for (const r of this.recs) {
      if (r.op !== 'Conv') continue;
      const cs = r.cs, Nvox = cs[2] * cs[3] * cs[4], gx = Math.min(65535, Math.ceil(Nvox / tileC));
      r.u = this.R.uni([cs[0], cs[1], cs[2], cs[3], cs[4], cs[5], cs[6], cs[7], cs[8], cs[9], cs[10], gx]);
      r.wg = [gx, Math.ceil(cs[1] / tileR), Math.ceil(Nvox / tileC / gx)];
    }
  }
  // verify candidate configs against the reference CONV on a tiny conv (rejects any buggy tiling); fast.
  async _verifyConfigs(configs) {
    const dev = this.dev, R = this.R;
    const Ci = 8, Co = 20, OD = 9, OH = 9, OW = 9, ID = 9, IH = 9, IW = 9, K = 3, S = 1, pad = 1;
    const Nin = Ci * ID * IH * IW, Nw = Co * Ci * K * K * K, Nout = Co * OD * OH * OW, Nvox = OD * OH * OW;
    const rnd = (n, seed) => { const a = new Float32Array(n); let s = seed >>> 0; for (let i = 0; i < n; i++) { s = (s * 1103515245 + 12345) & 0x7fffffff; a[i] = s / 0x7fffffff - 0.5; } return a; };
    const mkb = f => { const b = this.mk(f.length * 2); dev.queue.writeBuffer(b, 0, toF16(f)); return b; };
    const inB = mkb(rnd(Nin, 1)), wB = mkb(rnd(Nw, 2)), biasB = mkb(rnd(Co, 3)), outB = this.mk(Nout * 2);
    const readOut = async () => { const rb = dev.createBuffer({ size: Nout * 2, usage: U.MAP_READ | U.COPY_DST }); const e = dev.createCommandEncoder(); e.copyBufferToBuffer(outB, 0, rb, 0, Nout * 2); dev.queue.submit([e.finish()]); await rb.mapAsync(GPUMapMode.READ); const u = new Uint16Array(rb.getMappedRange().slice(0)); rb.unmap(); return fromF16(u); };
    const runCfg = (src, TM, TN) => { const tileC = 16 * TN, tileR = 16 * TM, gx = Math.min(65535, Math.ceil(Nvox / tileC));
      const u = R.uni([Ci, Co, OD, OH, OW, ID, IH, IW, K, S, pad, gx]); const wg = [gx, Math.ceil(Co / tileR), Math.ceil(Nvox / tileC / gx)];
      const e = dev.createCommandEncoder(); R.pass(e, src, [inB, wB, biasB, outB, u], wg); dev.queue.submit([e.finish()]); };
    runCfg(R.CONV, 4, 4); const ref = await readOut();
    const ok = [];
    for (const [TM, TN] of configs) { const src = (TM === 4 && TN === 4) ? R.CONV : genConv(TM, TN);
      try { runCfg(src, TM, TN); const o = await readOut(); let md = 0; for (let i = 0; i < Nout; i++) { const dd = Math.abs(o[i] - ref[i]); if (dd > md) md = dd; } if (md < 0.05) ok.push([TM, TN]); }
      catch (e) { /* shader/validation error → drop this config */ } }
    return ok.length ? ok : [[4, 4]];
  }
  // pick the fastest verified conv tiling for THIS gpu by timing the real forward. Inputs must be set. Never regresses.
  async autotuneConv(candidates = [[4, 4], [2, 8], [8, 2], [2, 4], [4, 2]], reps = 3) {
    const verified = await this._verifyConfigs(candidates);
    let best = [4, 4], bestMs = Infinity;
    for (const [TM, TN] of verified) {
      this.setConvConfig(TM, TN);
      this.run(); await this.dev.queue.onSubmittedWorkDone();               // warm (compile pipelines)
      const t = performance.now(); for (let i = 0; i < reps; i++) this.run(); await this.dev.queue.onSubmittedWorkDone();
      const ms = (performance.now() - t) / reps;
      if (ms < bestMs - 0.5) { bestMs = ms; best = [TM, TN]; }               // require a real margin
    }
    this.setConvConfig(best[0], best[1]);
    return { TM: best[0], TN: best[1], ms: Math.round(bestMs), verified: verified.length, tried: candidates.length };
  }
}
// argmax(ch1>ch0) over a 2-ch logits buffer -> u8 mask packed 4-per-u32 (avoids fromF16 + JS argmax loop)
const ARGMAX = `struct P{N:u32,gx:u32}; @group(0)@binding(0) var<storage,read> lg:array<f16>;
@group(0)@binding(1) var<storage,read_write> outp:array<u32>; @group(0)@binding(2) var<uniform> p:P;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid:vec3<u32>,@builtin(local_invocation_id) lid:vec3<u32>){
  let q=(wid.y*p.gx+wid.x)*256u+lid.x; let base=q*4u; if(base>=p.N){return;}
  var packed:u32=0u;
  for(var j=0u;j<4u;j++){ let i=base+j; if(i<p.N){
    let fg=select(0u,1u, f32(lg[p.N+i])>f32(lg[i])); packed=packed|(fg<<(j*8u)); } }
  outp[q]=packed; }`;

export function makeRunner(dev) {
  const pc = new Map();
  const pipe = src => { if (pc.has(src)) return pc.get(src); const m = dev.createShaderModule({ code: 'enable f16;\n' + src });
    const p = dev.createComputePipeline({ layout: 'auto', compute: { module: m, entryPoint: 'main' } }); pc.set(src, p); return p; };
  const uni = arr => { const b = dev.createBuffer({ size: Math.max(16, arr.length * 4), usage: U.UNIFORM | U.COPY_DST }); dev.queue.writeBuffer(b, 0, new Uint32Array(arr)); return b; };
  const pass = (enc, src, bufs, wg) => { const p = pipe(src);
    const bg = dev.createBindGroup({ layout: p.getBindGroupLayout(0), entries: bufs.map((b, i) => ({ binding: i, resource: { buffer: b } })) });
    const cp = enc.beginComputePass(); cp.setPipeline(p); cp.setBindGroup(0, bg); cp.dispatchWorkgroups(wg[0], wg[1] || 1, wg[2] || 1); cp.end(); };
  const grid = n => { const nWG = Math.ceil(n / 256), gx = Math.min(65535, nWG); return { gx, wg: [gx, Math.ceil(nWG / gx), 1] }; };
  return { pipe, uni, pass, grid, CONV, CONV_VSM, IN_STATS, IN_APPLY, IN_APPLY_LEAKY, ADD, ADD_LEAKY, LEAKY, POOL, RESIZE, ARGMAX };
}
