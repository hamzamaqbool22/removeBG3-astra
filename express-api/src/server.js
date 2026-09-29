import {mkdirSync,readFileSync,writeFileSync,unlinkSync} from 'node:fs';
import path from 'node:path';
import {config,integer} from './config.js';
import {Queue} from './queue.js';
import {createApp} from './app.js';
const host=process.env.HOST??'127.0.0.1',port=integer('PORT',8002,1,65535);
if(host!=='127.0.0.1'&&host!=='localhost'&&!process.env.API_KEY)throw Error('Set API_KEY before exposing this service outside localhost');
mkdirSync(config.dir,{recursive:true});
const lock=path.join(config.dir,'server.pid');
try{const pid=Number(readFileSync(lock,'utf8'));let alive=false;try{process.kill(pid,0);alive=true;}catch(e){if(e.code!=='ESRCH')alive=true;}if(alive)throw Error('Another server owns DATA_DIR. Run exactly one API process per data directory.');unlinkSync(lock);}catch(e){if(e.code!=='ENOENT')throw e;}
writeFileSync(lock,String(process.pid),{flag:'wx'});
const queue=new Queue(),app=createApp(queue);
const server=app.listen(port,host,()=>console.log(`Vehicle API listening on http://${host}:${port}`));
server.requestTimeout=120000;server.headersTimeout=15000;
let closing=false;
async function close(){if(closing)return;closing=true;server.close();await queue.close();unlinkSync(lock);process.exit(0);}
process.on('SIGINT',close);process.on('SIGTERM',close);
server.on('error',async e=>{console.error(e);await close();});
