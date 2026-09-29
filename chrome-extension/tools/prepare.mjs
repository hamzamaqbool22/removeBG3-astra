// Copies only into chrome-extension; never edits the Python project.
import {mkdir, copyFile, readdir, readFile, writeFile, unlink} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import path from 'node:path';
import {createHash} from 'node:crypto';
import {makeCspCompatible} from './csp-opencv.mjs';
const root = fileURLToPath(new URL('../', import.meta.url));
await mkdir(path.join(root,'vendor'), {recursive:true});
await mkdir(path.join(root,'models'), {recursive:true});
await writeFile(path.join(root,'vendor/opencv.js'),makeCspCompatible(await readFile(path.join(root,'node_modules/@techstark/opencv-js/dist/opencv.js'),'utf8')));
const dist = path.join(root,'node_modules/onnxruntime-web/dist');
for (const file of await readdir(dist)) {
  if (file==='ort.webgpu.min.js'||file.startsWith('ort-wasm'))
    await copyFile(path.join(dist,file), path.join(root,'vendor',file));
}
const catalog = {};
const assets=JSON.parse(await readFile(path.join(root,'assets.json'),'utf8'));
assets.fallbacks={};
delete assets.backgroundsBundled;
for (const folder of ['parking-lots','backgrounds']) {
  const src = path.join(root,'../backgrounds',folder), dst=path.join(root,'backgrounds',folder);
  await mkdir(dst,{recursive:true});
  catalog[folder] = (await readdir(src)).filter(x=>/^\d+\.png$/.test(x)).sort((a,b)=>parseInt(a)-parseInt(b));
  for (const file of catalog[folder]) {
    if(!assets.backgrounds[`${folder}/${file}`]?.url)throw Error(`Missing Cloudinary URL for ${folder}/${file}`);
  }
  await copyFile(path.join(src,'1.png'),path.join(dst,'default.png'));
  const bytes=await readFile(path.join(dst,'default.png'));
  assets.fallbacks[folder]={file:`${folder}/default.png`,sha256:createHash('sha256').update(bytes).digest('hex'),bytes:bytes.length};
  // These are generated copies; the original Python backgrounds stay untouched.
  for(const file of await readdir(dst))if(/^\d+\.png$/.test(file))await unlink(path.join(dst,file));
}
await writeFile(path.join(root,'backgrounds/catalog.json'),JSON.stringify(catalog,null,2));
await writeFile(path.join(root,'assets.json'),JSON.stringify(assets,null,2)+'\n');
console.log('Prepared runtime and two default backgrounds. Models and numbered backgrounds download at first use.');
