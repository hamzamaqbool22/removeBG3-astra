import {chromium} from 'playwright';
import {fileURLToPath} from 'node:url';
import {mkdtemp,rm,mkdir,writeFile} from 'node:fs/promises';
import path from 'node:path';
const root=fileURLToPath(new URL('../',import.meta.url));
const extensionRoot=process.env.EXTENSION_ROOT?path.resolve(process.env.EXTENSION_ROOT):root;
await mkdir(path.join(root,'validation'),{recursive:true});
const profile=await mkdtemp(path.join(root,'validation/profile-'));
let context;
try{
  context=await chromium.launchPersistentContext(profile,{channel:'chromium',headless:true,args:[`--disable-extensions-except=${extensionRoot}`,`--load-extension=${extensionRoot}`]});
  const sw=context.serviceWorkers()[0]??await context.waitForEvent('serviceworker',{timeout:30000});
  const id=new URL(sw.url()).hostname;
  const page=await context.newPage();
  const errors=[];page.on('pageerror',e=>errors.push(e.message));
  await page.goto(`chrome-extension://${id}/panel.html`);
  console.log('Loaded actual extension',id,await page.title());
  await page.locator('#backend').selectOption(process.argv.includes('--gpu')?'webgpu':'wasm');
  await page.locator('#folder').selectOption('white');
  if(process.argv.includes('--cloud-check')){
    await page.locator('#folder').selectOption('parking-lots');
    await page.locator('#number').selectOption('3.png');
  }
  await page.locator('#photos').setInputFiles([path.join(root,'images/car1.jpg'),path.join(root,'images/car2.jpg')]);
  await page.locator('#generate').click();
  await page.waitForFunction(()=>/^(Finished\.|Local processing failed)/.test(document.querySelector('#status').textContent),null,{timeout:600000});
  const log=await page.locator('#log').textContent();console.log(log);
  const output=path.join(root,'validation',process.argv.includes('--gpu')?'installed-gpu':'installed-cpu');await mkdir(output,{recursive:true});
  await writeFile(path.join(output,'log.txt'),log);
  const count=await page.locator('#results a').count();
  if(count!==2)throw Error('Expected two successful local photos; '+await page.locator('#status').textContent());
  for(let i=0;i<count;i++){const event=page.waitForEvent('download');await page.locator('#results a').nth(i).click();const download=await event;await download.saveAs(path.join(output,download.suggestedFilename()));}
  await page.locator('#cancel').click();
  if(!await page.locator('#generate').isEnabled())throw Error('Stop/release did not reset UI');
  if(errors.length)throw Error(errors.join('\n'));
  await page.screenshot({path:path.join(output,'panel.png'),fullPage:true});
  console.log('PASS: installed extension, two images, white output, downloads, release memory');
  if(process.argv.includes('--cache-check')){
    await context.setOffline(true);
    await page.reload();
    await page.locator('#backend').selectOption(process.argv.includes('--gpu')?'webgpu':'wasm');
    await page.locator('#folder').selectOption('parking-lots');
    await page.locator('#number').selectOption(process.argv.includes('--cloud-check')?'3.png':'1.png');
    await page.locator('#photos').setInputFiles(path.join(root,'images/car1.jpg'));
    await page.locator('#generate').click();
    await page.waitForFunction(()=>/^(Finished\.|Local processing failed)/.test(document.querySelector('#status').textContent),null,{timeout:180000});
    const cachedLog=await page.locator('#log').textContent();console.log(cachedLog);
    if(await page.locator('#results a').count()!==1||!cachedLog.includes('using saved download'))throw Error('Offline cache processing failed');
    if(process.argv.includes('--cloud-check')){
      if(!cachedLog.includes('Background parking-lots/3.png: using saved download'))throw Error('Cloudinary image was not cached');
      await page.locator('#folder').selectOption('backgrounds');
      await page.locator('#number').selectOption('1.png');
      await page.locator('#generate').click();
      await page.waitForFunction(()=>/^(Finished\.|Local processing failed)/.test(document.querySelector('#status').textContent),null,{timeout:180000});
      if(await page.locator('#results a').count()!==1||!(await page.locator('#notice').textContent()).includes('backgrounds/default.png'))throw Error('Offline default compositing failed');
      // Exercise both families independently, checking exact selected/default bytes.
      for(const [key,fallback] of [['parking-lots/4.png','parking-lots'],['backgrounds/2.png','backgrounds']]){
        const result=await page.evaluate(async({key,fallback})=>{
          const assets=await import('./src/assets.js'),config=await assets.assetConfig(),messages=[];
          const blob=await assets.backgroundBlob(key,m=>messages.push(m));
          await assets.verifiedBytes(await blob.arrayBuffer(),config.fallbacks[fallback]);
          return messages;
        },{key,fallback});
        if(!result.some(m=>m.startsWith('Using fallback')))throw Error('Expected offline fallback');
      }
      await context.setOffline(false);
      await page.evaluate(async()=>{
        const assets=await import('./src/assets.js'),config=await assets.assetConfig();
        await assets.verifiedBytes(await (await assets.backgroundBlob('backgrounds/2.png')).arrayBuffer(),config.backgrounds['backgrounds/2.png']);
      });
      await context.setOffline(true);
      const reuse=await page.evaluate(async()=>{
        const assets=await import('./src/assets.js'),messages=[];
        await assets.backgroundBlob('backgrounds/2.png',m=>messages.push(m));return messages;
      });
      if(!reuse.some(m=>m.includes('using saved download')))throw Error('Second folder cache failed');
      console.log('PASS: Cloudinary download, both-folder cache reuse, both-folder offline defaults, visible fallback notice and compositing');
    }
    await page.locator('#clearCache').click();
    await page.waitForFunction(()=>document.querySelector('#status').textContent.startsWith('Saved downloads cleared'));
    const saved=await page.evaluate(()=>caches.keys());
    if(saved.includes('vehicle-assets-v1'))throw Error('Clear cache failed');
    console.log('PASS: panel reload, offline processing, clear saved downloads');
  }
}finally{await context?.close();await rm(profile,{recursive:true,force:true});}
