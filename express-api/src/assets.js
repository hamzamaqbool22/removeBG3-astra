import {readFile,mkdir,rename,rm} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import path from 'node:path';
import {config} from './config.js';
export const catalog=JSON.parse(await readFile(new URL('../assets/catalog.json',import.meta.url)));
export async function verifiedAsset(spec,file,timeout=120000){
  const valid=bytes=>bytes.length===spec.bytes&&createHash('sha256').update(bytes).digest('hex')===spec.sha256;
  try{if(valid(await readFile(file)))return file;}catch{}
  const response=await fetch(spec.url,{signal:AbortSignal.timeout(timeout)});
  if(!response.ok)throw Error(`Asset download failed HTTP ${response.status}`);
  const data=Buffer.alloc(spec.bytes);let offset=0;
  for await(const chunk of response.body){if(offset+chunk.length>data.length)throw Error('Asset too large');data.set(chunk,offset);offset+=chunk.length;}
  if(offset!==data.length||!valid(data))throw Error('Asset checksum mismatch');
  await mkdir(path.dirname(file),{recursive:true});
  const {writeFile}=await import('node:fs/promises');
  try{await writeFile(file+'.tmp',data);await rename(file+'.tmp',file);}finally{await rm(file+'.tmp',{force:true});}
  return file;
}
export const modelPath=()=>verifiedAsset(catalog.models['birefnet-lite-512.onnx'],path.join(config.dir,'models/model.onnx'));
export async function backgroundPath(options){
  if(!options.isBackgroundWant)return {file:null};
  const key=`${options.folder}/${options.background}`;
  try{return {file:await verifiedAsset(catalog.backgrounds[key],path.join(config.dir,'backgrounds',key),15000)};}
  catch(e){return {file:new URL(`../assets/backgrounds/${options.folder}.png`,import.meta.url),warning:`Used default ${options.folder} background: ${e.message}`};}
}
