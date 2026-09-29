import test from 'node:test';
import assert from 'node:assert/strict';
import {webcrypto,createHash} from 'node:crypto';
import {verifiedBytes,cachedAsset} from '../src/assets.js';
globalThis.crypto ??= webcrypto;
const bytes=new TextEncoder().encode('known pinned model bytes');
const spec={bytes:bytes.length,sha256:createHash('sha256').update(bytes).digest('hex')};
test('accepts only pinned bytes',async()=>assert.equal(await verifiedBytes(bytes.buffer,spec),bytes.buffer));
test('rejects truncated downloads',async()=>assert.rejects(verifiedBytes(bytes.slice(1).buffer,spec),/size mismatch/));
test('rejects a same-length corrupt download',async()=>{const changed=bytes.slice();changed[0]^=1;await assert.rejects(verifiedBytes(changed.buffer,spec),/checksum mismatch/);});
test('cache recovers corrupt entries and never stores partial or failed downloads',async()=>{
  const previousFetch=globalThis.fetch,previousCaches=globalThis.caches;
  const entries=new Map();let downloads=0;
  const asset={...spec,url:'https://example.com/model.onnx'};
  const key=`https://huggingface.co/_vehicle-local-cache/${spec.sha256}`;
  globalThis.caches={open:async()=>({match:async k=>entries.get(k)?.clone(),delete:async k=>entries.delete(k),put:async(k,r)=>entries.set(k,r.clone())})};
  try{
    globalThis.fetch=async()=>{downloads++;return new Response(bytes);};
    await cachedAsset(asset,'test');
    await cachedAsset(asset,'test');
    assert.equal(downloads,1,'saved bytes should avoid a second download');
    entries.set(key,new Response(new Uint8Array(bytes.length)));
    await cachedAsset(asset,'test');
    assert.equal(downloads,2,'corrupt saved data should be replaced');
    entries.clear();globalThis.fetch=async()=>new Response(bytes.slice(1));
    await assert.rejects(cachedAsset(asset,'test'),/interrupted/);assert.equal(entries.size,0);
    globalThis.fetch=async()=>new Response('',{status:503});
    await assert.rejects(cachedAsset(asset,'test'),/HTTP 503/);assert.equal(entries.size,0);
    globalThis.fetch=async()=>new Response(new Uint8Array(bytes.length));
    await assert.rejects(cachedAsset(asset,'test'),/checksum/);assert.equal(entries.size,0);
  }finally{globalThis.fetch=previousFetch;globalThis.caches=previousCaches;}
});
