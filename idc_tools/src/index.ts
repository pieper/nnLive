// idc_tools — browser utilities to stream & reconstruct imaging series from the NCI Imaging Data
// Commons (IDC) public buckets, plus SEG Roulette (spin a random annotated series). No server.
//
//   import { spinRandom } from 'idc_tools';
//   const { ct, seg, entry } = await spinRandom(
//     { onProgress: p => console.log(p.msg), onCT: ct => renderFirst(ct) },
//     { manifestUrl: 'segroulette.json', filter: e => e.m === 'CT' });
//   // ct.vol (Int16 HU), ct.dims [nx,ny,nz], ct.ijkToRAS, ct.win/lev ; seg?.lab on the same grid.
//
// Code: Apache-2.0.  Data streamed at runtime from IDC is governed by each collection's own license.
export type {
  CTVolume, SegLabelmap, SeriesEntry, RouletteManifest,
  LoadProgress, LoadHandlers, LoadResult, LoaderOptions,
} from './types.js';
export { idcS3, fetchRetry, s3ListKeys, ohifViewerURL } from './s3.js';
export { loadManifest, pickRandom } from './roulette.js';
export { loadSeries, spinRandom } from './loader.js';
