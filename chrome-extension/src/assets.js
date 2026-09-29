// Only fixed, checksum-verified model/image data is downloaded. Code stays bundled.
const CACHE = 'vehicle-assets-v1';
let configuration;
export function assetConfig() {
  return configuration ??= fetch(new URL('../assets.json', import.meta.url))
    .then(r => { if (!r.ok) throw Error('Asset configuration is missing.'); return r.json(); });
}
export async function verifiedBytes(buffer, spec) {
  if (buffer.byteLength !== spec.bytes) throw Error('Download size mismatch. Please retry.');
  const hash = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', buffer)), b => b.toString(16).padStart(2, '0')).join('');
  if (hash !== spec.sha256) throw Error('Download checksum mismatch. Please retry.');
  return buffer;
}
export async function cachedAsset(spec, label, report = () => {}) {
  const cache = await caches.open(CACHE);
  // Cache.put requires an HTTP(S) key, even inside a chrome-extension origin.
  // This is only a storage key; no request is sent to this synthetic URL.
  const key = `https://huggingface.co/_vehicle-local-cache/${spec.sha256}`;
  const existing = await cache.match(key);
  if (existing) {
    try {
      const data = await verifiedBytes(await existing.arrayBuffer(), spec);
      report(`${label}: using saved download`);
      return data;
    } catch { await cache.delete(key); report(`${label}: saved file damaged; downloading again`); }
  }
  if (!spec.url) throw Error('Numbered backgrounds need a hosting URL. Use White, Transparent or Upload a background for now.');
  if (new URL(spec.url).protocol !== 'https:') throw Error('Asset downloads require HTTPS.');
  report(`Downloading ${label} — 0% (first use)`);
  const controller = new AbortController();
  let timer;
  const timeoutMs=spec.timeoutMs??60000;
  const resetTimeout = () => { clearTimeout(timer); timer = setTimeout(() => controller.abort(), timeoutMs); };
  const deadline=spec.timeoutMs?setTimeout(()=>controller.abort(),timeoutMs):null;
  resetTimeout();
  try {
    const response = await fetch(spec.url, {signal: controller.signal, credentials:'omit', cache:'no-store', referrerPolicy:'no-referrer'});
    if (!response.ok) throw Error(`Download failed (HTTP ${response.status}). Please retry when connected.`);
    const reader = response.body.getReader();
    const data = new Uint8Array(spec.bytes);
    let offset = 0, lastPercent = -1;
    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      resetTimeout();
      if (offset + value.length > data.length) { await reader.cancel(); throw Error('Download exceeds expected size.'); }
      data.set(value, offset); offset += value.length;
      const percent = Math.floor(offset / spec.bytes * 100);
      if (percent >= lastPercent + 5) { lastPercent = percent; report(`Downloading ${label} — ${percent}%`); }
    }
    if (offset !== spec.bytes) throw Error('Download interrupted. Please retry.');
    report(`Verifying ${label}…`);
    await verifiedBytes(data.buffer, spec);
    try { await cache.put(key, new Response(data, {headers:{'Content-Type':'application/octet-stream'}})); }
    catch (error) { throw Error(`Could not save ${label} in browser storage (${error.message}). Check free disk space and retry.`); }
    report(`${label}: saved for future use`);
    return data.buffer;
  } catch (error) {
    if (error.name === 'AbortError') throw Error('Download timed out. Check your connection and retry.');
    throw error;
  } finally { clearTimeout(timer); clearTimeout(deadline); }
}
export async function modelBytes(filename, report) {
  const config = await assetConfig();
  const spec = config.models[filename];
  if (!spec) throw Error('Unknown model.');
  return cachedAsset(spec, filename.includes('fp16') ? 'GPU model' : 'CPU model', report);
}
export async function backgroundBlob(key, report) {
  const config = await assetConfig(), entry = config.backgrounds[key];
  if (!entry) throw Error('Unknown background.');
  return backgroundWithFallback(key,entry,config.fallbacks[key.split('/')[0]],report);
}
export async function backgroundWithFallback(key,entry,fallback,report=()=>{}) {
  try {
    return new Blob([await cachedAsset({...entry,timeoutMs:15000},`Background ${key}`,report)],{type:'image/png'});
  } catch(error) {
    report(`Background ${key} download unavailable: ${error.message}`);
  }
  if(!fallback)throw Error('Default background is not configured.');
  const response = await fetch(new URL(`../backgrounds/${fallback.file}`, import.meta.url));
  if (!response.ok) throw Error('Bundled background is missing. Reinstall the extension.');
  const bytes = await verifiedBytes(await response.arrayBuffer(), fallback);
  report(`Using fallback ${fallback.file} because ${key} could not be downloaded. Retry online to use your selected background.`);
  return new Blob([bytes], {type:'image/png'});
}
export async function clearSavedAssets() { await caches.delete(CACHE); }
