// watch.mjs <jobId> — poll a batch job until terminal state; survives transient network errors.
import { api } from './gpu.mjs';
const ID = process.argv[2];
async function once() {
  try {
    const r = await api('/api/batch?id=' + ID);
    const list = (r.json && (r.json.jobs || [])) || [];
    return list.find(x => x && x.id === ID) || list[0] || null;
  } catch (e) { return { __err: String(e).slice(0, 80) }; }
}
if (!ID) console.log('USAGE watch.mjs <jobId>');
else {
  let prev = '', errs = 0;
  for (;;) {
    const j = await once();
    if (j && j.__err) { errs++; if (errs % 10 === 1) console.log(new Date().toISOString().slice(11, 19), 'net-retry', j.__err); }
    else if (j) {
      errs = 0;
      const bytes = (j.output_capture || {}).received_bytes || 0;
      const key = j.state + '/' + bytes;
      if (key !== prev) { console.log(new Date().toISOString().slice(11, 19), 'state=' + j.state, 'exit=' + j.exit_code, 'bytes=' + bytes); prev = key; }
      if (j.state && j.state !== 'Running' && j.state !== 'Queued') {
        console.log('--- final ---');
        console.log((j.output || '').slice(-6000));
        break;
      }
    }
    await new Promise(res => setTimeout(res, 30000));
  }
}