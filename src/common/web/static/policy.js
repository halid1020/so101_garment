// The rollout page: what the policy was shown, what it planned, what the arms
// did with it, and whether it arrived in time. One poll drives all of it; the
// pictures are separate streams the browser keeps open by itself.

const $ = (s) => document.querySelector(s);
const JOINTS = ['pan', 'lift', 'elbow', 'wrist_flex', 'wrist_roll'];
const SIDES = ['left', 'right'];
let status = {};
let twinWhat = 'plan';

async function j(url, opts) {
  const r = await fetch(url, opts);
  if (!r.ok) throw new Error((await r.text()) || r.statusText);
  return r.json();
}

// -- the throttle -----------------------------------------------------
async function setMode(mode) {
  try {
    await j('/api/mode', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({mode}),
    });
    $('#p-err').hidden = true;
  } catch (e) {
    $('#p-err').textContent = e.message;
    $('#p-err').hidden = false;
  }
  poll();
}

$('#p-arm').onclick = () => {
  // The same sentence the terminal used to print, in the one place that can
  // still stop it being true.
  if (!confirm('The follower arms will MOVE: they ramp to the policy\u2019s '
               + 'first action, then follow it. Is the workspace clear?')) return;
  setMode('arm');
};
$('#p-hold').onclick = () => setMode('hold');
$('#p-preview').onclick = () => setMode('preview');
$('#p-step').onclick = () => setMode('step');
$('#p-run').onclick = () => setMode('run');
$('#p-stop').onclick = () => setMode('stop');
$('#p-twin-what').onclick = () => {
  twinWhat = twinWhat === 'plan' ? 'measured' : 'plan';
  $('#p-twin-what').textContent =
    `showing: ${twinWhat === 'plan' ? 'the plan' : 'the arms'}`;
  // A new src restarts the stream, which is how the twin switches subject.
  $('#p-twin').src = twinWhat === 'plan' ? '/twin.mjpg'
                                         : '/twin.mjpg?what=measured';
};

// -- the pictures -----------------------------------------------------
// Built once, when the status first names the cameras: an <img> on an mjpeg
// stream is left alone by the browser, and rebuilding it would restart it.
function buildTiles(cameras) {
  if ($('#p-live').children.length) return;
  const tile = (src, caption) =>
    `<figure class="tile"><img src="${src}" alt="${caption}">
     <figcaption>${caption}</figcaption></figure>`;
  $('#p-live').innerHTML = cameras.map(
    (c) => tile(`/stream/${c}.mjpg`, c)).join('');
  $('#p-shown').innerHTML = cameras.map(
    (c) => tile('', c)).join('');
  $('#p-twin').src = '/twin.mjpg';
}

// The frames the last request carried are still, so they are fetched rather
// than streamed -- once per chunk, which is when they change.
let shownSeq = -1;
function refreshShown(cameras, seq) {
  if (seq === shownSeq) return;
  shownSeq = seq;
  const imgs = $('#p-shown').querySelectorAll('img');
  cameras.forEach((c, i) => {
    if (imgs[i]) imgs[i].src = `/shown/${c}.jpg?seq=${seq}`;
  });
}

// -- the numbers ------------------------------------------------------
function jointsTable(state, commanded) {
  const rows = [];
  rows.push(`<tr><th></th><th>measured</th><th>commanded</th><th>error</th></tr>`);
  SIDES.forEach((side, s) => {
    JOINTS.forEach((name, k) => {
      const i = s * 6 + k;
      const m = state ? state[i] : null;
      const c = commanded ? commanded[i] : null;
      const e = (m === null || c === null) ? null : c - m;
      const far = e !== null && Math.abs(e) > 5 ? ' class="far"' : '';
      rows.push(`<tr><td>${side} ${name}</td><td>${fmt(m)}</td>
        <td>${fmt(c)}</td><td${far}>${fmt(e)}</td></tr>`);
    });
  });
  $('#p-joints').innerHTML = rows.join('');
}

// The grippers get their own panel because they are where a grasp is won or
// lost, and the whole range they were taught spans a few hundredths.
function grippers(state, commanded) {
  $('#p-grip').innerHTML = SIDES.map((side, s) => {
    const i = s * 6 + 5;
    const m = state ? state[i] : null;
    const c = commanded ? commanded[i] : null;
    const gap = (m === null || c === null) ? null : c - m;
    return `<div class="grip"><span class="muted">${side}</span>
      <b>${fmt(m, 3)}</b>
      <span class="cmd">commanded ${fmt(c, 3)}</span>
      <span class="gap">gap ${fmt(gap, 3)}</span></div>`;
  }).join('');
}

function fmt(v, digits = 2) {
  return (v === null || v === undefined) ? '—' : v.toFixed(digits);
}

// -- the vitals -------------------------------------------------------
// The four numbers that decide whether a rollout is healthy, sized so they
// read across the room, and coloured only when something is actually wrong.
function vital(id, value, note, level) {
  const el = $(`#${id}`);
  el.querySelector('b').textContent = value;
  if (note !== undefined) el.querySelector('span').textContent = note;
  el.className = `vital${level ? ' ' + level : ''}`;
}

function vitals(s) {
  const queue = s.queue === undefined ? null : s.queue;
  const need = s.threshold || 0;
  const remote = s.chunked !== false;
  // An empty queue means the arms are holding for want of a plan; a queue at
  // or below the threshold with nothing in flight is about to be.
  const level = queue === 0 ? 'bad' : (queue !== null && queue <= need ? 'warn' : '');
  vital('v-queue', queue === null ? '—' : String(queue),
        remote ? `queue · asks at ${need}` : 'queue · local', level);
  const rtt = (s.round_trip_s || 0) * 1000;
  vital('v-rtt', remote && rtt ? `${rtt.toFixed(0)} ms` : '—', 'round trip');
  const infer = (s.server_infer_s || 0) * 1000;
  vital('v-infer', remote && infer ? `${infer.toFixed(0)} ms` : '—',
        'inference, of that');
  const holds = s.holds || 0;
  const ticks = s.tick || 0;
  vital('v-holds', `${holds}`,
        ticks ? `held · ${((holds / ticks) * 100).toFixed(0)}% of ticks` : 'held',
        holds && ticks && holds / ticks > 0.4 ? 'warn' : '');
  vital('v-chunk',
        s.pending ? `${s.pending.n}` : (s.chunk ? `${s.chunk.n}` : '—'),
        s.pending ? 'queued to execute' : 'plan, actions');
  vital('v-splice', s.strategy || '—', 'splice');
  vital('v-progress', `${ticks}/${s.ticks_total || 0}`,
        `${(s.t || 0).toFixed(0)} s at ${s.hz || 0} Hz`);
}

function timing(s) {
  const rtt = (s.round_trip_s || 0) * 1000;
  const need = (s.threshold || 0);
  const queue = s.queue === undefined ? null : s.queue;
  const starving = queue !== null && queue === 0;
  const row = (k, v, note = '') =>
    `<tr><td>${k}</td><td>${v}</td><td class="muted">${note}</td></tr>`;
  $('#p-timing').innerHTML = [
    row('round trip', `${rtt.toFixed(0)} ms`,
        `inference ${((s.server_infer_s || 0) * 1000).toFixed(0)} ms of it`),
    row('queue', queue === null ? '—' : `${queue} action(s)`,
        starving ? 'EMPTY — the arms are waiting for a plan'
                 : `next chunk requested at ${need}`),
    row('chunk', s.actions_per_chunk ? `${s.actions_per_chunk} actions`
                                     : '—',
        s.chunk ? `plan #${s.chunk.seq}` : 'none yet'),
    row('holds', `${s.holds || 0} tick(s)`, s.last_error || ''),
    row('progress', `${s.tick || 0} / ${s.ticks_total || 0} ticks`,
        `${(s.t || 0).toFixed(1)} s at ${s.hz || 0} Hz`),
  ].join('');
}

// -- the plan ---------------------------------------------------------
// One line per body joint over the chunk's horizon, with the executed part
// shaded: the plan and how far into it the arms have got, in one picture.
function drawChunk(chunk, executed) {
  const cv = $('#p-chunk'), ctx = cv.getContext('2d');
  ctx.clearRect(0, 0, cv.width, cv.height);
  if (!chunk) {
    $('#p-chunk-note').textContent = 'no plan yet';
    return;
  }
  const a = chunk.actions, n = a.length;
  const cols = [];
  for (let i = 0; i < 12; i++) if (i % 6 !== 5) cols.push(i);
  let lo = Infinity, hi = -Infinity;
  a.forEach((row) => cols.forEach((c) => {
    lo = Math.min(lo, row[c]); hi = Math.max(hi, row[c]);
  }));
  if (!(hi > lo)) { hi = lo + 1; }
  const x = (i) => (i / Math.max(1, n - 1)) * (cv.width - 8) + 4;
  const y = (v) => cv.height - 6 - ((v - lo) / (hi - lo)) * (cv.height - 12);

  if (executed > 0) {
    ctx.fillStyle = '#8882';
    ctx.fillRect(4, 0, x(Math.min(executed, n - 1)) - 4, cv.height);
  }
  cols.forEach((c, k) => {
    ctx.beginPath();
    ctx.strokeStyle = `hsl(${(k * 36) % 360} 65% 50%)`;
    ctx.lineWidth = 1.2;
    a.forEach((row, i) => (i ? ctx.lineTo(x(i), y(row[c]))
                             : ctx.moveTo(x(i), y(row[c]))));
    ctx.stroke();
  });
  $('#p-chunk-note').textContent =
    `plan #${chunk.seq}: ${n} actions, ${lo.toFixed(0)}° to ${hi.toFixed(0)}°`;
}

// -- the loop ---------------------------------------------------------
function render(s) {
  const mode = s.mode || '…';
  $('#p-mode').textContent = s.stopping ? 'stopping' : mode;
  $('#p-mode').className = `mode ${mode}`;
  $('#p-clock').textContent = s.task ? `“${s.task}”` : '';
  $('#p-torque').hidden = !!s.dry_run;
  ['hold', 'preview', 'step', 'run'].forEach((m) => {
    $(`#p-${m}`).classList.toggle('sel', m === mode);
  });
  $('#p-step').disabled = s.chunked === false;
  $('#p-preview').disabled = s.chunked === false;
  // Only while this run is waiting for consent it delegated to us.
  $('#p-arm').hidden = !(s.arm_from_view && !s.armed && !s.dry_run);

  buildTiles(s.cameras || []);
  refreshShown(s.cameras || [], s.chunk ? s.chunk.seq : -1);
  const twinBtn = $('#p-twin-what');
  if (twinWhat === 'plan') {
    // Under every splice but 'append' the queue is not the returned chunk:
    // say which one is on screen, or the picture is quietly misleading.
    twinBtn.textContent = s.pending
      ? 'showing: what will execute' : 'showing: the plan';
  }
  $('#p-shown-note').textContent = s.chunk
    ? `the window of plan #${s.chunk.seq}, ${((Date.now() / 1000) - s.chunk.at).toFixed(1)} s ago`
    : 'nothing sent yet';
  jointsTable(s.state, s.commanded);
  grippers(s.state, s.commanded);
  vitals(s);
  timing(s);
  // How much of THIS chunk has gone. The queue can hold more than one
  // chunk's worth under the 'append' splice, so it is clamped rather than
  // subtracted blind -- otherwise the shading runs backwards.
  const queued = Math.min(s.queue || 0, s.chunk ? s.chunk.n : 0);
  drawChunk(s.chunk, s.chunk ? s.chunk.n - queued : 0);
}

async function poll() {
  try {
    status = await j('/api/status');
    render(status);
  } catch (e) {
    $('#p-err').textContent = `the rollout is not answering: ${e.message}`;
    $('#p-err').hidden = false;
  }
}

poll();
setInterval(poll, 400);
