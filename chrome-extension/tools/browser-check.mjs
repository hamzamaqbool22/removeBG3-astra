import {chromium} from 'playwright';
import {createServer} from 'node:http';
import {createReadStream} from 'node:fs';
import {stat,mkdir,writeFile,readdir} from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const root=fileURLToPath(new URL('../',import.meta.url));
const server=createServer(async(req,res)=>{try{const file=path.resolve(root,'.'+decodeURIComponent(req.url.split('?')[0]));if(!file.startsWith(root))throw Error();const info=await stat(file);if(!info.isFile())throw Error();res.setHeader('Content-Type',({'.js':'application/javascript','.mjs':'application/javascript','.html':'text/html','.css':'text/css','.json':'application/json','.wasm':'application/wasm','.png':'image/png','.jpg':'image/jpeg'})[path.extname(file)]??'application/octet-stream');res.setHeader('Content-Security-Policy',"script-src 'self' 'wasm-unsafe-eval'; object-src 'self'; worker-src 'self'; connect-src 'self'; img-src 'self' blob: data:;");createReadStream(file).pipe(res);}catch{res.writeHead(404);res.end();}});
await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
const origin=`http://127.0.0.1:${server.address().port}`;
let browser;
try{
  browser=await chromium.launch({channel:'chrome',headless:true,args:['--enable-unsafe-webgpu']});
  const page=await browser.newPage();page.on('console',msg=>console.log('browser:',msg.type(),msg.text()));page.on('pageerror',error=>console.log('pageerror:',error.message));
  await page.addInitScript(()=>{
    window.__masks=[];
    const OriginalWorker=window.Worker;
    window.Worker=class extends OriginalWorker{
      constructor(...args){super(...args);this.addEventListener('message',({data})=>{if(data.type==='diagnostic-mask')window.__masks.push({...data,mask:Array.from(data.mask)});});}
      postMessage(data,...args){super.postMessage(data?.type==='generate'?{...data,diagnostics:true}:data,...args);}
    };
  });
  await page.goto(origin+'/panel.html');
  console.log('UI',await page.title());
  console.log('GPU',await page.evaluate(async()=>{const a=await navigator.gpu?.requestAdapter();return a?{info:a.info,buffer:a.limits.maxBufferSize,storage:a.limits.maxStorageBufferBindingSize}:null;}));
  const check=await page.evaluate(()=>new Promise(resolve=>{const w=new Worker('tools/probe-worker.js');w.onmessage=e=>{resolve(e.data);w.terminate();};w.onerror=e=>{resolve({error:e.message});w.terminate();};}));
  console.log('Runtime',JSON.stringify(check));
  if(check.error)throw Error(check.error);
  if(process.argv.includes('--infer')){
    if(process.argv.includes('--cpu'))await page.locator('#backend').selectOption('wasm');
    const value=flag=>{const i=process.argv.indexOf(flag);return i<0?null:process.argv[i+1];};
    const input=value('--input');
    const files=input?(await readdir(input)).filter(f=>/\.(png|jpg|jpeg)$/i.test(f)).sort().map(f=>path.join(input,f)):[path.join(root,'images/car1.jpg')];
    if(value('--background')){await page.locator('#folder').selectOption('parking-lots');await page.locator('#number').selectOption(value('--background'));}
    if(process.argv.includes('--enhance'))await page.locator('#enhance').check();
    await page.locator('#photos').setInputFiles(files);
    await page.locator('#generate').click();
    await page.waitForFunction(()=>document.querySelector('#status').textContent.startsWith('Finished.')||document.querySelector('#status').textContent.startsWith('Local processing failed'),null,{timeout:600000});
    console.log('Outcome',await page.locator('#status').innerText());
    const output=path.join(root,'validation',value('--output')??(process.argv.includes('--cpu')?'lite-cpu':'lite-gpu'));await mkdir(output,{recursive:true});
    await writeFile(path.join(output,'details.txt'),await page.locator('#details').textContent());
    await writeFile(path.join(output,'log.txt'),await page.locator('#log').textContent());
    const count=await page.locator('#results a').count();
    for(const m of await page.evaluate(()=>window.__masks))await writeFile(path.join(output,m.name.replace(/\.[^.]+$/,'')+'-mask.bin'),Buffer.from(m.mask));
    for(let i=0;i<count;i++){const download=page.waitForEvent('download');await page.locator('#results a').nth(i).click();const d=await download;await d.saveAs(path.join(output,d.suggestedFilename()));}
    console.log('Downloaded',count,'of',files.length,'to',output);
    if(count!==files.length)throw Error(await page.locator('#status').innerText());
  }
}finally{await browser?.close();server.close();}
