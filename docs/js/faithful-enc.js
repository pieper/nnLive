// Faithful nnInteractive interaction encoding for the browser runtime.
// Mirrors the shipped inference session (v1.0): global whole-volume z-score, OOB pad 0,
// EDT point balls radius 4, interaction decay 0.9, prev_seg autoregressive channel, and
// auto-zoom (crop round(zoom*ps) FOV, trilinear image downsample to ps, interactions mapped by /zoom).
// 7-ch inter (injected downstream): [0 prev_seg, 1 bbox+, 2 bbox-, 3 pt+, 4 pt-, 5 scr+, 6 scr-]
import { EDT_BALL, EDT_R, EDT_D } from './edt-ball.js';

export const DECAY = 0.9;
export const CH = { PREV:0, BBOXP:1, BBOXN:2, PTP:3, PTN:4, SCRP:5, SCRN:6 };

// origin (box lower corner, may be negative / OOB) for a centered patch at a given zoom.
export function boxOrigin(center, ps, zoom) {
  const size = Math.round(ps * zoom);
  return [Math.round(center[0]) - (size >> 1), Math.round(center[1]) - (size >> 1), Math.round(center[2]) - (size >> 1)];
}

// Max-stamp the EDT ball (scaled by intensity) into a ps^3 channel at patch coords (cz,cy,cx).
export function placePoint(ch, ps, cz, cy, cx, intensity=1.0) {
  const r = EDT_R;
  cz=Math.round(cz); cy=Math.round(cy); cx=Math.round(cx);
  for (let dz=0; dz<EDT_D; dz++) { const z=cz-r+dz; if (z<0||z>=ps) continue;
    for (let dy=0; dy<EDT_D; dy++) { const y=cy-r+dy; if (y<0||y>=ps) continue;
      const bo=(dz*EDT_D+dy)*EDT_D, po=(z*ps+y)*ps;
      for (let dx=0; dx<EDT_D; dx++) { const x=cx-r+dx; if (x<0||x>=ps) continue;
        const v=EDT_BALL[bo+dx]*intensity; if (v>ch[po+x]) ch[po+x]=v; } } }
}

// Global whole-volume z-score stats (computed once at image load).
export function globalStats(vol) {
  let s=0; const n=vol.length; for (let i=0;i<n;i++) s+=vol[i]; const mean=s/n;
  let q=0; for (let i=0;i<n;i++){ const d=vol[i]-mean; q+=d*d; } const std=Math.sqrt(q/n)+1e-8;
  return { mean, std };
}

// Extract a ps^3 image crop for a centered patch at `zoom`, GLOBAL z-score, OOB pad 0.
// zoom==1 => direct crop; zoom>1 => trilinear downsample of a (zoom*ps)^3 FOV (matches session crop_img path).
export function extractCrop(vol, Z, Y, X, center, ps, zoom, mean, std) {
  const out = new Float32Array(ps*ps*ps); const inv=1/std;
  const [oz,oy,ox] = boxOrigin(center, ps, zoom);
  const sample = (zz,yy,xx) => (zz<0||zz>=Z||yy<0||yy>=Y||xx<0||xx>=X) ? 0 : (vol[(zz*Y+yy)*X+xx]-mean)*inv;
  if (zoom === 1) {
    for (let z=0; z<ps; z++) for (let y=0; y<ps; y++){ const orow=(z*ps+y)*ps;
      for (let x=0; x<ps; x++) out[orow+x]=sample(oz+z, oy+y, ox+x); }
    return out;
  }
  // trilinear downsample: output o maps to crop-local (o+0.5)*zoom-0.5, global = origin + that
  for (let z=0; z<ps; z++){ const gz=oz+(z+0.5)*zoom-0.5, z0=Math.floor(gz), fz=gz-z0;
    for (let y=0; y<ps; y++){ const gy=oy+(y+0.5)*zoom-0.5, y0=Math.floor(gy), fy=gy-y0; const orow=(z*ps+y)*ps;
      for (let x=0; x<ps; x++){ const gx=ox+(x+0.5)*zoom-0.5, x0=Math.floor(gx), fx=gx-x0;
        const c000=sample(z0,y0,x0),   c001=sample(z0,y0,x0+1),   c010=sample(z0,y0+1,x0),   c011=sample(z0,y0+1,x0+1);
        const c100=sample(z0+1,y0,x0), c101=sample(z0+1,y0,x0+1), c110=sample(z0+1,y0+1,x0), c111=sample(z0+1,y0+1,x0+1);
        const c00=c000*(1-fx)+c001*fx, c01=c010*(1-fx)+c011*fx, c10=c100*(1-fx)+c101*fx, c11=c110*(1-fx)+c111*fx;
        const c0=c00*(1-fy)+c01*fy, c1=c10*(1-fy)+c11*fy;
        out[orow+x]=c0*(1-fz)+c1*fz; } } }
  return out;
}

// Interaction memory in FULL-VOLUME coordinates + running prev_seg. One instance per object.
export class Interactions {
  constructor(Z, Y, X) { this.Z=Z; this.Y=Y; this.X=X;
    this.points=[];                 // {z,y,x,sign,age}
    this.scribbles=[];              // {pts:[[z,y,x]...], sign, age}
    this.bboxes=[];                 // {z0,y0,x0,z1,y1,x1, sign, age}
    this.prev=null;                 // Uint8Array(Z*Y*X) running prediction, or null
  }
  addPoint(z, y, x, sign) { for (const p of this.points) p.age++; this.points.push({ z, y, x, sign, age:0 }); }
  addScribble(pts, sign) { for (const s of this.scribbles) s.age++; this.scribbles.push({ pts, sign, age:0 }); }
  addBbox(z0,y0,x0,z1,y1,x1, sign) { for (const b of this.bboxes) b.age++; this.bboxes.push({ z0,y0,x0,z1,y1,x1, sign, age:0 }); }
  clear() { this.points=[]; this.scribbles=[]; this.bboxes=[]; this.prev=null; }

  // 7-channel inter for a centered patch at `zoom`. Interaction coords mapped global->patch by /zoom.
  buildInter(center, ps, zoom) {
    const S=ps*ps*ps; const inter=new Float32Array(7*S);
    const [oz,oy,ox] = boxOrigin(center, ps, zoom);
    const g2p = (g,o) => (g - o) / zoom;                       // global coord -> patch coord
    // prev_seg (ch0): sample running prediction (nearest) at each patch voxel's global location
    if (this.prev) { const pv=inter.subarray(CH.PREV*S, CH.PREV*S+S);
      for (let z=0; z<ps; z++){ const gz=Math.round(oz+(z+0.5)*zoom-0.5); if(gz<0||gz>=this.Z) continue;
        for (let y=0; y<ps; y++){ const gy=Math.round(oy+(y+0.5)*zoom-0.5); if(gy<0||gy>=this.Y) continue; const orow=(z*ps+y)*ps, vrow=(gz*this.Y+gy)*this.X;
          for (let x=0; x<ps; x++){ const gx=Math.round(ox+(x+0.5)*zoom-0.5); if(gx<0||gx>=this.X) continue; if(this.prev[vrow+gx]) pv[orow+x]=1.0; } } } }
    // points (ch3/4): EDT ball with decay, mapped to patch coords
    for (const p of this.points) {
      const cz=g2p(p.z,oz), cy=g2p(p.y,oy), cx=g2p(p.x,ox);
      if (cz< -EDT_R||cz>=ps+EDT_R||cy< -EDT_R||cy>=ps+EDT_R||cx< -EDT_R||cx>=ps+EDT_R) continue;
      const ch=(p.sign>0?CH.PTP:CH.PTN); placePoint(inter.subarray(ch*S, ch*S+S), ps, cz, cy, cx, Math.pow(DECAY, p.age));
    }
    // scribbles (ch5/6): thin polyline, mapped to patch coords, decay
    for (const s of this.scribbles) { const ch=(s.sign>0?CH.SCRP:CH.SCRN)*S, a=inter.subarray(ch, ch+S), v=Math.pow(DECAY,s.age);
      for (const [gz,gy,gx] of s.pts){ const z=Math.round(g2p(gz,oz)), y=Math.round(g2p(gy,oy)), x=Math.round(g2p(gx,ox));
        for(let dz=-1;dz<=1;dz++)for(let dy=-1;dy<=1;dy++)for(let dx=-1;dx<=1;dx++){ const zz=z+dz,yy=y+dy,xx=x+dx; if(zz<0||zz>=ps||yy<0||yy>=ps||xx<0||xx>=ps)continue; const o=(zz*ps+yy)*ps+xx; if(v>a[o])a[o]=v; } } }
    // bbox (ch1/2): filled box region, mapped to patch coords, decay
    for (const b of this.bboxes) { const ch=(b.sign>0?CH.BBOXP:CH.BBOXN)*S, a=inter.subarray(ch, ch+S), v=Math.pow(DECAY,b.age);
      const pz0=Math.max(0,Math.round(g2p(b.z0,oz))), pz1=Math.min(ps,Math.round(g2p(b.z1,oz)));
      const py0=Math.max(0,Math.round(g2p(b.y0,oy))), py1=Math.min(ps,Math.round(g2p(b.y1,oy)));
      const px0=Math.max(0,Math.round(g2p(b.x0,ox))), px1=Math.min(ps,Math.round(g2p(b.x1,ox)));
      for(let z=pz0;z<pz1;z++)for(let y=py0;y<py1;y++){ const row=(z*ps+y)*ps; for(let x=px0;x<px1;x++) if(v>a[row+x])a[row+x]=v; } }
    return inter;
  }
}
