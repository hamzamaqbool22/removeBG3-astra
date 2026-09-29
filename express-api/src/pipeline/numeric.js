// Numerical counterparts of NumPy/SciPy operations used by the Python pipeline.
export const clip=(x,a=0,b=1)=>Math.max(a,Math.min(b,x));
export const mean=a=>a.reduce((s,x)=>s+x,0)/a.length;
export function quantile(a,q){const s=Array.from(a).sort((x,y)=>x-y);if(!s.length)return NaN;const x=(s.length-1)*q,i=Math.floor(x);return s[i]+(s[Math.min(i+1,s.length-1)]-s[i])*(x-i);}
export const median=a=>quantile(a,.5);
export function round(x){const f=Math.floor(x),r=x-f;return r===.5?(f%2===0?f:f+1):Math.round(x);}
export const u8=x=>clip(round(x),0,255);
export function reflect(i,n){while(i<0||i>=n)i=i<0?-i-1:2*n-i-1;return i;}
export function smooth1d(a,sigma){const r=Math.floor(4*sigma+.5),k=[];let sum=0;for(let i=-r;i<=r;i++){const v=Math.exp(-.5*(i/sigma)**2);k.push(v);sum+=v;}return Float64Array.from(a,(_,i)=>k.reduce((s,v,j)=>s+v*a[reflect(i+j-r,a.length)],0)/sum);}
export const median1d=(a,size)=>Float64Array.from(a,(_,i)=>median(Array.from({length:size},(_,j)=>a[reflect(i+j-(size>>1),a.length)])));
export function interp(x,xs,ys){if(x<=xs[0])return ys[0];if(x>=xs.at(-1))return ys.at(-1);let lo=0,hi=xs.length-1;while(hi-lo>1){const m=(lo+hi)>>1;if(xs[m]<=x)lo=m;else hi=m;}return ys[lo]+(ys[hi]-ys[lo])*(x-xs[lo])/(xs[hi]-xs[lo]);}
export function peaks(a,prominence,distance){
  let candidates=[];
  for(let i=1;i<a.length-1;i++){if(a[i]<=a[i-1])continue;let end=i;while(end+1<a.length&&a[end+1]===a[i])end++;if(end<a.length-1&&a[end]>a[end+1])candidates.push(Math.floor((i+end)/2));i=end;}
  const kept=[];for(const i of candidates.sort((i,j)=>a[j]-a[i]))if(!kept.some(j=>Math.abs(i-j)<distance))kept.push(i);
  return kept.filter(i=>{let l=a[i],r=a[i];for(let j=i-1;j>=0&&a[j]<=a[i];j--)l=Math.min(l,a[j]);for(let j=i+1;j<a.length&&a[j]<=a[i];j++)r=Math.min(r,a[j]);return a[i]-Math.max(l,r)>=prominence;}).sort((a,b)=>a-b);
}
// Small least-squares system, solved with pivoting. Inputs are normalized x/y.
export function leastSquares(rows,values,valid){const n=rows[0].length,A=Array.from({length:n},()=>new Float64Array(n+1));for(let r=0;r<rows.length;r++){if(!valid[r])continue;for(let i=0;i<n;i++){for(let j=0;j<n;j++)A[i][j]+=rows[r][i]*rows[r][j];A[i][n]+=rows[r][i]*values[r];}}for(let i=0;i<n;i++){let pivot=i;for(let k=i+1;k<n;k++)if(Math.abs(A[k][i])>Math.abs(A[pivot][i]))pivot=k;[A[pivot],A[i]]=[A[i],A[pivot]];if(Math.abs(A[i][i])<1e-10)return null;const d=A[i][i];for(let j=i;j<=n;j++)A[i][j]/=d;for(let k=0;k<n;k++)if(k!==i){const f=A[k][i];for(let j=i;j<=n;j++)A[k][j]-=f*A[i][j];}}return A.map(r=>r[n]);}

// Pillow LANCZOS uses support=3 and a widened antialiasing kernel when shrinking.
// Each pass quantizes to uint8, just like Pillow's RGB/L paths.
export function lanczos(image,w,h,kernel='lanczos'){
  function weights(src,dst){const scale=src/dst,filter=Math.max(1,scale),support=(kernel==='bilinear'?1:3)*filter;return Array.from({length:dst},(_,i)=>{const center=(i+.5)*scale,lo=Math.max(0,Math.floor(center-support+.5)),hi=Math.min(src,Math.floor(center+support+.5));let sum=0;const k=[];for(let j=lo;j<hi;j++){const x=(j-center+.5)/filter;const sinc=t=>Math.abs(t)<1e-12?1:Math.sin(Math.PI*t)/(Math.PI*t);const v=kernel==='bilinear'?Math.max(0,1-Math.abs(x)):Math.abs(x)<3?sinc(x)*sinc(x/3):0;k.push(v);sum+=v;}return {lo,k:k.map(v=>v/sum)};});}
  const c=image.c;let current=image;
  for(const axis of [0,1]){const src=axis?current.h:current.w,dst=axis?h:w;if(src===dst)continue;const ks=weights(src,dst),nw=axis?current.w:w,nh=axis?h:current.h,out=new Uint8Array(nw*nh*c);for(let y=0;y<nh;y++)for(let x=0;x<nw;x++){const {lo,k}=ks[axis?y:x];for(let ch=0;ch<c;ch++){let v=0;for(let j=0;j<k.length;j++){const xx=axis?x:lo+j,yy=axis?lo+j:y;v+=current.data[(yy*current.w+xx)*c+ch]*k[j];}out[(y*nw+x)*c+ch]=clip(Math.floor(v+.5),0,255);}}current={w:nw,h:nh,c,data:out};}
  return current;
}
