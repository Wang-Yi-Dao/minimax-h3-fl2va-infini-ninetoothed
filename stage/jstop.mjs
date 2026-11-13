import { cancel } from './batch.mjs';
const r = await cancel(process.argv[2]);
console.log('cancel', r.status, JSON.stringify(r.json).slice(0,300));
