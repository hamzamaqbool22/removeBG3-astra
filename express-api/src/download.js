import https from 'node:https';
import {lookup} from 'node:dns/promises';
import {createWriteStream} from 'node:fs';
import {rm} from 'node:fs/promises';
import {Transform} from 'node:stream';
import {pipeline} from 'node:stream/promises';
import ipaddr from 'ipaddr.js';
// Resolve once and pin the connection to that public IP. Redirects are rejected.
export async function downloadImage(raw,destination,maxBytes,hosts) {
  const url=new URL(raw);
  if(url.protocol!=='https:'||url.username||url.password||(url.port&&url.port!=='443')||!hosts.includes(url.hostname))throw Error('imageurl must use HTTPS on an IMAGE_URL_HOSTS allowlisted host');
  const addresses=await lookup(url.hostname,{all:true});
  if(!addresses.length||addresses.some(a=>ipaddr.process(a.address).range()!=='unicast'))throw Error('Private or reserved imageurl addresses are not allowed');
  const ip=addresses[0];
  const controller=new AbortController(),timer=setTimeout(()=>controller.abort(),30000);
  try {
    const response=await new Promise((resolve,reject)=>{
      const req=https.get(url,{signal:controller.signal,lookup:(_host,opts,cb)=>opts?.all?cb(null,[ip]):cb(null,ip.address,ip.family)},resolve);
      req.on('error',reject);
    });
    if(response.statusCode!==200){response.destroy();throw Error(`Image URL returned HTTP ${response.statusCode}; use a direct URL without redirects`);}
    let count=0;
    await pipeline(response,new Transform({transform(chunk,_encoding,callback){count+=chunk.length;callback(count>maxBytes?Error('Image URL exceeds upload limit'):null,chunk);}}),createWriteStream(destination,{flags:'wx'}));
    if(!count)throw Error('Empty image download');
  }catch(e){await rm(destination,{force:true});throw e;}finally{clearTimeout(timer);}
}
