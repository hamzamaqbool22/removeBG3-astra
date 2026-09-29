import {refineMask,estimateGeometry} from './geometry.js';
import {recoverShadow} from './source-shadow.js';
import {renderShadows} from './shadows.js';
import {normalize,parking,transparent} from './composite.js';
import {warp} from './cv.js';
import {clip} from './numeric.js';

export function processWithMask(rgb,rawMask,background=null,enhancement=false,progress=()=>{}){
  progress('Refining mask and estimating tire contacts');const alpha=refineMask(rawMask),geometry=estimateGeometry(rgb,alpha);
  progress('Recovering the source shadow');const source=recoverShadow(rgb,alpha),placed=normalize(rgb,alpha,geometry);
  progress('Rendering ground shadows');const shadow=source?Float32Array.from(warp(source,placed.matrix,1024,768,cv.INTER_LINEAR).data,v=>clip(v,0,.97)):renderShadows(1024,768,placed.geometry).combined;
  progress('Compositing the final image');const image=background?parking(placed.rgb,placed.alpha,shadow,placed.geometry,background,enhancement):transparent(placed.rgb,placed.alpha,shadow,enhancement);
  return {image,geometry,shadowMode:source?'Recovered source shadow':'Procedural contact / chassis / ground',shadow};
}
