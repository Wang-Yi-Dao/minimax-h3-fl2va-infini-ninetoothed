import fs from 'node:fs';
import { saveText, saveConfig, project, snapshot, preflight, submit } from './batch.mjs';
const SH = process.env.JOBSH, PY = process.env.PYFILE, NAME = process.env.NAME||'job';
const p0 = await project();
const get = n => { const f=(p0.files||[]).find(x=>x.path===n); return f ? f.revision : 0; };
const plan = [];
if (PY) plan.push(['probe.py', PY]);
plan.push(['cmp1.sh', SH]);
for (const [name, local] of plan) {
  const text = fs.readFileSync(local,'utf8');
  const r = await saveText(name, text, get(name));
  console.log('save', name, '<-', local, text.length, r.status);
  if (r.status !== 200) { console.log(r.text); process.exit(1); }
}
const p = await project();
console.log('cfg', (await saveConfig({ entrypoint: 'cmp1.sh', args: [], kind: 'batch' }, p.config?.revision)).status);
const s = await snapshot();
const CNT = Number(process.env.CNT||8), TMO = Number(process.env.TMO||900);
const pf = await preflight(s.json?.id, TMO, CNT);
console.log('pf', pf.status, pf.json?.allowed);
const FORCE = process.env.FORCE === '1';
if ((pf.status === 200 && pf.json?.allowed) || FORCE) {
  const sub = await submit(s.json?.id, TMO, NAME, CNT);
  console.log('sub', sub.status, JSON.stringify(sub.json).slice(0,160));
} else console.log('PF_BLOCKED', JSON.stringify(pf.json).slice(0,200));
