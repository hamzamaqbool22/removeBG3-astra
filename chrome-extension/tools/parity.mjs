import {readFile,writeFile} from 'node:fs/promises';
import {createRequire} from 'node:module';
const require=createRequire(import.meta.url);
globalThis.cv=await require('@techstark/opencv-js');
const {raster,warp}=await import('../src/cv.js');
const {estimateGeometry}=await import('../src/geometry.js');
const {recoverShadow}=await import('../src/source-shadow.js');
const {renderShadows}=await import('../src/shadows.js');
const {normalize,parking}=await import('../src/composite.js');
const root=new URL('../validation/',import.meta.url),meta=JSON.parse(await readFile(new URL('reference.json',root)));
async function array(name){const {shape,dtype}=meta.arrays[name],buf=await readFile(new URL(name+'.bin',root)),data=dtype==='uint8'?new Uint8Array(buf):new Float32Array(buf.buffer.slice(buf.byteOffset,buf.byteOffset+buf.byteLength));return raster(shape[1],shape[0],shape[2]??1,data);}
function difference(a,b){if(a.length!==b.length)return {error:'shape'};let sum=0,max=0,nonzero=0;for(let i=0;i<a.length;i++){const e=Math.abs(a[i]-b[i]);sum+=e;max=Math.max(max,e);if(e>1e-5)nonzero++;}return {mae:sum/a.length,max,different:nonzero/a.length};}
const rgb=await array('rgb'),alpha=await array('alpha'),bg=await array('background');
const start=performance.now(),geometry=estimateGeometry(rgb,alpha),source=recoverShadow(rgb,alpha),p=normalize(rgb,alpha,geometry),procedural=renderShadows(1024,768,p.geometry).combined;
const shadow=source?Float32Array.from(warp(source,p.matrix,1024,768,cv.INTER_LINEAR).data,v=>Math.max(0,Math.min(.97,v))):procedural;
const final=parking(p.rgb,p.alpha,shadow,p.geometry,bg,true),report={seconds:(performance.now()-start)/1000,source:!!source,geometry:geometry.view_cues,pythonGeometry:meta.geometry.view_cues};
for(const [name,result]of Object.entries({placed_rgb:p.rgb.data,placed_alpha:p.alpha.data,procedural,shadow,final:final.data,...(source?{source:source.data}:{})})){
  report[name]=meta.arrays[name]?difference(result,(await array(name)).data):'No Python source';
}
await writeFile(new URL('js-final.bin',root),final.data);
await writeFile(new URL('parity.json',root),JSON.stringify(report,null,2));
console.log(JSON.stringify(report,null,2));
