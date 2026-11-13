import fs from 'node:fs';
const BASE='https://tools.infinitensor.com/compute/scheduler';
const TOK=fs.readFileSync(new URL('./token.txt',import.meta.url),'utf8').trim();
const HDR={Authorization:'Bearer '+TOK};
async function call(body){
  const r=await fetch(BASE+'/api/persistent-storage/files',{method:'POST',headers:{...HDR,'Content-Type':'application/json'},body:JSON.stringify({request:body})});
  const t=await r.text(); let j=null; try{j=JSON.parse(t);}catch{}
  if(!r.ok) throw new Error(body.op+' '+(j&&j.error||t.slice(0,200)));
  return j;
}
export async function list(path){
  const j=await call({op:'list',path});
  return j.entries||[];
}
export async function stat(path){
  const p=path.split('/'); const name=p.pop();
  const es=await list(p.join('/'));
  return es.find(e=>e.name===name)||null;
}
export async function download(path,out){
  const id=Array.from({length:32},()=>'0123456789abcdef'[Math.floor(Math.random()*16)]).join('');
  const chunks=[]; let offset=0,size=null;
  for(;;){
    const r=await call({op:'download',path,offset,download_id:id});
    if(r.download_state==='preparing'){await new Promise(s=>setTimeout(s,500));continue;}
    const buf=Buffer.from(r.data,'base64');
    chunks.push(buf); offset+=buf.length; size=r.size;
    if(r.eof) break;
  }
  const all=Buffer.concat(chunks);
  if(out) fs.writeFileSync(out,all);
  return {size:all.length,sha:null};
}
export async function upload(local,path){
  const buf=fs.readFileSync(local);
  const crypto=await import('node:crypto');
  const sha=crypto.createHash('sha256').update(buf).digest('hex');
  const id=crypto.randomBytes(16).toString('hex');
  let r=await call({op:'begin',id,path,size:buf.length,sha256:sha});
  let chunk=r.raw_upload_bytes===1048576?1048576:(r.http_block_bytes===262144?262144:(r.chunk_bytes===65536?65536:32768));
  let offset=Number.isSafeInteger(r.offset)?r.offset:0;
  while(offset<buf.length&&!r.committed){
    const bytes=buf.subarray(offset,offset+chunk);
    const up=await fetch(BASE+'/api/persistent-storage/upload-chunk',{method:'POST',headers:{...HDR,'Content-Type':'application/octet-stream','X-Upload-ID':id,'X-Upload-Offset':String(offset)},body:bytes});
    const ut=await up.text(); let uj=null; try{uj=JSON.parse(ut);}catch{}
    if(!up.ok) throw new Error('chunk '+up.status+' '+(uj&&uj.error||ut.slice(0,120)));
    r=uj; offset=r.offset;
  }
  const c=await call({op:'commit',id});
  if(c.committed!==true) throw new Error('commit unconfirmed');
  return {bytes:buf.length,sha};
}
export async function mkdir(path){ return await call({op:'mkdir',path}); }
export async function trash(path,version){ return await call({op:'trash',path,version,id:'0123456789abcdef0123456789abcdef'}); }
