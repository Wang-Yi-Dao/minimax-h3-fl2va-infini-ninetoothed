import fs from 'node:fs';
const BASE = 'https://tools.infinitensor.com/compute/scheduler';
const TOK = fs.readFileSync(new URL('./token.txt', import.meta.url), 'utf8').trim();
const STATE = new URL('./gpu_state.json', import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, '$1');
export function loadGpu() { try { return JSON.parse(fs.readFileSync(STATE, 'utf8')); } catch { return null; } }
export function saveGpu(s) { fs.writeFileSync(STATE, JSON.stringify(s)); }
export async function api(path, body, timeoutMs = 60000) {
  const ac = new AbortController(); const t = setTimeout(() => ac.abort(), timeoutMs);
  try {
    const r = await fetch(BASE + path, { method: body === undefined ? 'GET' : 'POST', headers: { Authorization: 'Bearer ' + TOK, ...(body === undefined ? {} : { 'Content-Type': 'application/json' }) }, body: body === undefined ? undefined : JSON.stringify(body), signal: ac.signal });
    const text = await r.text(); let json = null; try { json = JSON.parse(text); } catch { }
    return { status: r.status, json, text };
  } finally { clearTimeout(t); }
}
export async function allocRequest(count = 8, timeout = 1800) {
  const rid = Math.random().toString(16).slice(2).padEnd(32, '0').slice(0, 32);
  const r = await api('/api/resources/request', { pool: 'scnet-dcu', mode: 'cards', count, timeout, request_id: rid });
  return { status: r.status, json: r.json, text: r.text };
}
export async function allocInfo(id) {
  const r = await api('/api/resources');
  const a = (r.json?.allocations || []).find(x => x.id === id);
  const nodes = (r.json?.pools || []).flatMap(p => p.nodes || []);
  return { alloc: a, nodes };
}
export async function termOpen(allocation, node) {
  const r = await api('/api/resource-terminal/open', { allocation, node, cols: 200, rows: 50 });
  if (r.status !== 200 && r.status !== 201) throw new Error('term open ' + r.status + ' ' + r.text.slice(0, 200));
  const st = { alloc: allocation, node, id: r.json.id, sid: r.json.id, cursor: 0, nextSeq: 0 };
  saveGpu(st); return st;
}
export async function termPoll(st) {
  const r = await api('/api/resource-terminal/poll', { id: st.id, cursor: st.cursor }, 30000);
  if (r.status !== 200) return { text: '', state: 'ERR' + r.status, raw: r.json };
  const j = r.json; let text = '';
  for (const b of (j.frames || [])) text += Buffer.from(b, 'base64').toString('utf8');
  st.cursor = j.cursor ?? st.cursor; if (Number.isInteger(j.seq)) st.nextSeq = Math.max(st.nextSeq, j.seq);
  saveGpu(st); return { text, state: j.state, raw: j };
}
export async function termWrite(st, text) {
  const buf = Buffer.from(text, 'utf8');
  for (let i = 0; i < buf.length; i += 2048) {
    const data = buf.slice(i, i + 2048).toString('base64');
    let ok = false;
    for (let n = 0; n < 5; n++) {
      const r = await api('/api/resource-terminal/input', { id: st.id, seq: st.nextSeq + 1, data }, 20000);
      if (r.status === 200 && r.json && r.json.accepted) { ok = true; break; }
      await new Promise(res => setTimeout(res, 250));
    }
    if (!ok) throw new Error('term input not accepted');
    st.nextSeq++; saveGpu(st);
  }
}
export async function termRun(cmd, opts = {}) {
  const st = loadGpu(); const tag = 'G' + Math.random().toString(36).slice(2, 8).toUpperCase();
  const deadline = Date.now() + (opts.timeoutMs || 240000);
  await termWrite(st, 'MK=' + tag + '; echo "__BE""GIN_$MK"; ' + cmd + ' ; echo "__EN""D_$MK=$?"\n');
  let out = '';
  while (Date.now() < deadline) {
    const p = await termPoll(loadGpu());
    out += p.text;
    const re = new RegExp('__END_' + tag + '=(-?\\d+)');
    const m = out.match(re);
    if (m) return { out, code: Number(m[1]), state: p.state };
    if (p.state && /Closed|Exited|Cancelled/i.test(String(p.state))) return { out, code: null, state: p.state };
    await new Promise(res => setTimeout(res, 1500));
  }
  return { out, code: null, state: 'TIMEOUT' };
}