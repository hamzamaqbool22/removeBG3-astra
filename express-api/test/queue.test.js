import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,rm,writeFile} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import path from 'node:path';
import {randomUUID} from 'node:crypto';
import {Queue} from '../src/queue.js';
import {createApp} from '../src/app.js';
const worker=new URL('./fixtures/worker.js',import.meta.url);
async function until(fn,ms=15000){const start=Date.now();while(!fn()){if(Date.now()-start>ms)throw Error('Timed out');await new Promise(r=>setTimeout(r,20));}}
test('100 simultaneous uploads are persisted, processed serially, and downloadable; excess gets 429',async()=>{
  const dir=await mkdtemp(path.join(tmpdir(),'vehicle-queue-'));const q=new Queue({dir,maxJobs:100,worker});
  const app=createApp(q),server=app.listen(0,'127.0.0.1');await new Promise(r=>server.once('listening',r));
  const base=`http://127.0.0.1:${server.address().port}`;
  try{
    // Hold processing so admissions cannot race with completions during the burst.
    await until(()=>q.ready);q.ready=false;
    const replies=await Promise.all(Array.from({length:100},async()=>{const f=new FormData();f.append('image',new Blob(['photo']),'photo.png');return fetch(base+'/jobs',{method:'POST',body:f});}));
    assert.ok(replies.every(r=>r.status===202));
    const jobs=await Promise.all(replies.map(r=>r.json()));assert.equal(new Set(jobs.map(j=>j.jobId)).size,100);
    const extra=await fetch(base+'/jobs',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});assert.equal(extra.status,429);
    assert.equal((await fetch(base+'/health')).status,200);
    assert.equal(q.count(),100);q.ready=true;q.dispatch();await until(()=>q.count()===0);
    for(const job of jobs)assert.equal(q.get(job.jobId).status,'done');
    assert.equal((await fetch(base+jobs[0].resultUrl)).status,200);
    assert.equal((await fetch(base+'/jobs/not-valid')).status,404);
  }finally{await new Promise(r=>server.close(r));await q.close();await rm(dir,{recursive:true,force:true});}
});
test('restart recovers an interrupted job; crashed and timed-out workers do not lose waiting jobs',async()=>{
  const dir=await mkdtemp(path.join(tmpdir(),'vehicle-recovery-'));let q=new Queue({dir,worker,timeout:1000});
  const add=async opts=>{const id=randomUUID();await writeFile(path.join(dir,'inputs',id),'photo');q.add(id,opts);return id;};
  try{
    await until(()=>q.ready);const interrupted=await add({delay:10000});await q.close();
    q=new Queue({dir,worker,timeout:20000});await until(()=>q.get(interrupted).status==='done');
    q.timeout=1000;
    const crash=await add({crash:true}),next=await add({});await until(()=>q.get(next).status==='done');assert.equal(q.get(crash).status,'failed');
    const hung=await add({hang:true}),last=await add({});await until(()=>q.get(last).status==='done');assert.equal(q.get(hung).status,'failed');
  }finally{await q.close();await rm(dir,{recursive:true,force:true});}
});
