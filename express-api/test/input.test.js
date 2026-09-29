import test from 'node:test';
import assert from 'node:assert/strict';
import {options} from '../src/app.js';
import {downloadImage} from '../src/download.js';
import {createApp} from '../src/app.js';
test('accepts documented options and rejects invalid background paths and booleans',()=>{
  assert.deepEqual(options({}),{isBackgroundWant:false,enhancement:false,folder:'parking-lots',background:'1.png'});
  assert.equal(options({background:3,enhancment:'true',isBackgroundWant:'true'}).enhancement,true);
  assert.throws(()=>options({folder:'../../'}));assert.throws(()=>options({background:'../../../1'}));assert.throws(()=>options({isBackgroundWant:'yes'}));
});
test('URL input rejects forbidden schemes, credentials and unapproved hosts before network access',async()=>{
  for(const url of ['file:///etc/passwd','http://example.com/a','https://user:pass@example.com/a','https://other.example/a','https://example.com:8443/a'])
    await assert.rejects(downloadImage(url,'/tmp/not-written-image',100,['example.com']),/HTTPS/);
  await assert.rejects(downloadImage('https://127.0.0.1/a','/tmp/not-written-image',100,['127.0.0.1']),/Private or reserved/);
});
test('service authentication protects job endpoints and health remains available',async()=>{
  const original=process.env.API_KEY;process.env.API_KEY='test-secret';
  const app=createApp({dir:'/tmp'}),server=app.listen(0,'127.0.0.1');await new Promise(r=>server.once('listening',r));
  const base=`http://127.0.0.1:${server.address().port}`;
  try{
    assert.equal((await fetch(base+'/backgrounds')).status,401);
    assert.equal((await fetch(base+'/backgrounds',{headers:{Authorization:'Bearer test-secret'}})).status,200);
    assert.equal((await fetch(base+'/health')).status,200);
  }finally{await new Promise(r=>server.close(r));if(original===undefined)delete process.env.API_KEY;else process.env.API_KEY=original;}
});
