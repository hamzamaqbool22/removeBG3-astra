// Optional developer model copies for comparisons. The release downloads pinned
// model data itself; these local copies are not bundled. No photos are uploaded.
import {mkdir,readFile,writeFile,rename,rm} from 'node:fs/promises';
import {createWriteStream,createReadStream} from 'node:fs';
import {pipeline} from 'node:stream/promises';
import {createHash} from 'node:crypto';
const revision='4a3c40c36c94093cc1e724d9ea428b8fa4b57dc7';
const repository='studioludens/birefnet-lite-512';
const root=new URL('../models/',import.meta.url);
const models=[
  {file:'birefnet-lite-512.onnx',remote:'model.onnx',sha256:'1cb0fb360dadd15af77c639085d77a9df67db0c64315560c3de005f676345ac2'},
  {file:'birefnet-lite-512-fp16.onnx',remote:'model_fp16.onnx',sha256:'eff9216bb2f9d3f023d9c2b7196845a7485739ab1f231593633e4d2344ffc516'},
];
async function hash(file){const h=createHash('sha256');for await(const part of createReadStream(file))h.update(part);return h.digest('hex');}
await mkdir(root,{recursive:true});
for(const model of models){
  const file=new URL(model.file,root);
  if(await hash(file).then(h=>h===model.sha256).catch(()=>false)){console.log('Verified',model.file);continue;}
  const temporary=new URL(model.file+'.download',root);
  try{
    const response=await fetch(`https://huggingface.co/${repository}/resolve/${revision}/onnx/${model.remote}`);
    if(!response.ok)throw Error(`Model download failed: HTTP ${response.status}`);
    await pipeline(response.body,createWriteStream(temporary));
    if(await hash(temporary)!==model.sha256)throw Error('Model checksum mismatch');
    await rename(temporary,file);console.log('Downloaded and verified',model.file);
  }finally{await rm(temporary,{force:true});}
}
await writeFile(new URL('model-info.json',root),JSON.stringify({repository,revision,models,inputSize:512},null,2));
