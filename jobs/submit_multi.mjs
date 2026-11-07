import fs from 'node:fs';
import { saveText, saveConfig, project, snapshot, preflight, submit } from './batch.mjs';
// generic multi-file submitter
//   FILES="remote1=local1,remote2=local2"  ENTRY=cmp1.sh  NAME=... CNT=.. TMO=..
const FILES = (process.env.FILES||'').split(',').filter(Boolean);
const ENTRY = process.env.ENTRY || 'cmp1.sh';
const NAME = process.env.NAME || 'job';
const p0 = await project();
const get = n => { const f=(p0.files||[]).find(x=>x.path===n); return f ? f.revision : 0; };
for (const spec of FILES) {
  const i = spec.indexOf('=');
  const name = spec.slice(0,i), local = spec.slice(i+1);
  const text = fs.readFileSync(local,'utf8');
  const r = await saveText(name, text, get(name));
  console.log('save', name, '<-', local, text.length, r.status);
  if (r.status !== 200) { console.log(r.text); process.exit(1); }
}
const p = await project();
console.log('FILES_NOW', (p.files||[]).map(f=>f.path).join(','));
const ARGS = (process.env.ARGS||'').split(/\s+/).filter(Boolean);
console.log('cfg', (await saveConfig({ entrypoint: ENTRY, args: ARGS, kind: 'batch' }, p.config?.revision)).status, JSON.stringify(ARGS));
const s = await snapshot();
const CNT = Number(process.env.CNT||8), TMO = Number(process.env.TMO||900);
const pf = await preflight(s.json?.id, TMO, CNT);
console.log('pf', pf.status, pf.json?.allowed);
if (pf.status === 200 && pf.json?.allowed) {
  const sub = await submit(s.json?.id, TMO, NAME, CNT);
  console.log('sub', sub.status, JSON.stringify(sub.json).slice(0,160));
} else console.log('PF_BLOCKED', JSON.stringify(pf.json).slice(0,200));
