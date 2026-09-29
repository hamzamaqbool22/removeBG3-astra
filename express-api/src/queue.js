import {DatabaseSync} from 'node:sqlite';
import {mkdirSync,unlinkSync,readdirSync,statSync} from 'node:fs';
import {fork} from 'node:child_process';
import path from 'node:path';
import {config} from './config.js';
const log=(event,details={})=>console.log(JSON.stringify({time:new Date().toISOString(),event,...details}));
export class Queue {
  constructor({dir=config.dir,maxJobs=config.maxJobs,worker=new URL('./worker.js',import.meta.url),timeout=config.timeout,ttl=config.ttl}={}){
    Object.assign(this,{dir,maxJobs,worker,timeout,ttl});this.ready=false;this.active=null;this.stopping=false;this.reservations=0;
    for(const sub of ['inputs','results','tmp'])mkdirSync(path.join(dir,sub),{recursive:true});
    this.db=new DatabaseSync(path.join(dir,'jobs.sqlite'));
    this.db.exec(`PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL; CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY,status TEXT,created INTEGER,updated INTEGER,options TEXT,stage TEXT,error TEXT,result TEXT); CREATE INDEX IF NOT EXISTS jobs_status ON jobs(status,created);`);
    this.db.prepare("UPDATE jobs SET status='queued',stage='recovered after restart' WHERE status='running'").run();
    this.cleanup();this.interval=setInterval(()=>this.cleanup(),60000);this.interval.unref();
    this.spawn();
  }
  count(){return this.db.prepare("SELECT count(*) n FROM jobs WHERE status IN ('queued','running')").get().n;}
  reserve(){if(this.stopping||this.count()+this.reservations>=this.maxJobs)return false;this.reservations++;return true;}
  release(){this.reservations--;}
  add(id,options){const now=Date.now();this.db.prepare("INSERT INTO jobs VALUES(?, 'queued', ?, ?, ?, 'queued', NULL, NULL)").run(id,now,now,JSON.stringify(options));this.dispatch();}
  get(id){const row=this.db.prepare('SELECT * FROM jobs WHERE id=?').get(id);if(!row)return null;return {id,status:row.status,stage:row.stage,createdAt:new Date(row.created).toISOString(),updatedAt:new Date(row.updated).toISOString(),error:row.error,...JSON.parse(row.result??'{}'),queuePosition:row.status==='queued'?this.db.prepare("SELECT count(*) n FROM jobs WHERE status='running' OR (status='queued' AND rowid<=(SELECT rowid FROM jobs WHERE id=?))").get(id).n:0};}
  removeFiles(id){for(const file of [path.join(this.dir,'inputs',id),path.join(this.dir,'results',id+'.png'),path.join(this.dir,'results',id+'.tmp.png')])try{unlinkSync(file);}catch(e){if(e.code!=='ENOENT')log('cleanup_error',{error:e.message});}}
  cleanup(){
    for(const {id} of this.db.prepare("SELECT id FROM jobs WHERE status IN ('done','failed') AND updated<?").all(Date.now()-this.ttl)){this.removeFiles(id);this.db.prepare('DELETE FROM jobs WHERE id=?').run(id);}
    for(const folder of ['tmp','inputs'])for(const file of readdirSync(path.join(this.dir,folder))){const p=path.join(this.dir,folder,file);if(Date.now()-statSync(p).mtimeMs>3600000&&(folder==='tmp'||!this.db.prepare('SELECT id FROM jobs WHERE id=?').get(file)))try{unlinkSync(p);}catch{}}
  }
  spawn(){
    if(this.stopping)return;
    this.ready=false;
    this.child=fork(this.worker,[],{env:{...process.env,DATA_DIR:this.dir},stdio:['ignore','inherit','inherit','ipc']});
    this.bootTimer=setTimeout(()=>this.child?.kill('SIGKILL'),240000);
    this.child.on('message',m=>{
      if(m.type==='ready'){clearTimeout(this.bootTimer);this.ready=true;log('worker_ready');this.dispatch();return;}
      if(m.id!==this.active)return;
      if(m.type==='progress')this.db.prepare('UPDATE jobs SET stage=?,updated=? WHERE id=?').run(m.stage,Date.now(),m.id);
      if(['done','failed'].includes(m.type)){
        clearTimeout(this.jobTimer);
        this.db.prepare('UPDATE jobs SET status=?,stage=?,error=?,result=?,updated=? WHERE id=?').run(m.type,m.type,m.error??null,JSON.stringify(m.result??{}),Date.now(),m.id);
        try{unlinkSync(path.join(this.dir,'inputs',m.id));}catch{}
        if(m.type==='failed')this.removeFiles(m.id);
        log('job_'+m.type,{id:m.id,error:m.error});this.active=null;this.dispatch();
      }
    });
    this.child.on('error',e=>log('worker_error',{error:e.message}));
    this.child.on('exit',(code,signal)=>{
      clearTimeout(this.bootTimer);clearTimeout(this.jobTimer);this.ready=false;
      if(this.active){this.db.prepare("UPDATE jobs SET status='failed',stage='failed',error=?,updated=? WHERE id=?").run('Processing worker stopped or timed out. Please retry this image.',Date.now(),this.active);this.removeFiles(this.active);this.active=null;}
      log('worker_exit',{code,signal});if(!this.stopping)this.restartTimer=setTimeout(()=>this.spawn(),3000);
    });
  }
  dispatch(){
    if(this.stopping||!this.ready||this.active)return;
    const row=this.db.prepare("SELECT * FROM jobs WHERE status='queued' ORDER BY rowid LIMIT 1").get();if(!row)return;
    this.active=row.id;this.db.prepare("UPDATE jobs SET status='running',stage='starting',updated=? WHERE id=?").run(Date.now(),row.id);
    this.jobTimer=setTimeout(()=>this.child.kill('SIGKILL'),this.timeout);
    log('job_started',{id:row.id});
    this.child.send({type:'job',id:row.id,input:path.join(this.dir,'inputs',row.id),output:path.join(this.dir,'results',row.id+'.png'),options:JSON.parse(row.options)});
  }
  async close(){
    this.stopping=true;clearInterval(this.interval);clearTimeout(this.restartTimer);clearTimeout(this.bootTimer);clearTimeout(this.jobTimer);
    if(this.active){this.db.prepare("UPDATE jobs SET status='queued',stage='waiting for restart' WHERE id=?").run(this.active);this.active=null;}
    if(this.child?.exitCode===null&&this.child?.signalCode===null)await new Promise(resolve=>{this.child.once('exit',resolve);this.child.kill('SIGKILL');});
    this.db.close();
  }
}
