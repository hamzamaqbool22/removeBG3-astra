import {createRequire} from 'node:module';
import {readFile,writeFile,mkdir} from 'node:fs/promises';
const require=createRequire(import.meta.url);globalThis.cv=await require('@techstark/opencv-js');
const {processWithMask}=await import('../src/pipeline.js');
const root=new URL('../validation/eight-python/',import.meta.url),out=new URL('../validation/eight-js-same-mask/',import.meta.url);
await mkdir(out,{recursive:true});
const meta=JSON.parse(await readFile(new URL('reference.json',root)));
const bytes=async file=>new Uint8Array(await readFile(new URL(file,root)));
const bg={w:meta.background.width,h:meta.background.height,c:3,data:await bytes('background.bin')};
for(const [name,{width:w,height:h}]of Object.entries(meta.views)){
  const start=performance.now(),rgb={w,h,c:3,data:await bytes(name+'-rgb.bin')},mask={w,h,c:1,data:await bytes(name+'-mask.bin')};
  const result=processWithMask(rgb,mask,bg,true);
  await writeFile(new URL(name+'.bin',out),result.image.data);
  console.log(name,((performance.now()-start)/1000).toFixed(2),result.shadowMode);
}
