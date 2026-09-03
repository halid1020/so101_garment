// The progress panel: what every machine is training, and how it is going.
//
// The Runs column used to list what THIS console had launched and offer one
// button that returned one line. Almost nothing was in that list: nineteen of
// the twenty-one run directories on these machines were started from a
// terminal. So the list is discovered, and a card can be opened into the curve
// the driver's log carries.
//
// Two costs shape everything here. Asking a machine anything is a whole SSH,
// and a filtered log is a couple of hundred kilobytes. So the LIST is cheap
// (one call per machine, cached on the server) and carries only what a
// directory listing knows; the CURVE is fetched for the runs being watched --
// the ones opened, the ones ticked for comparison, and whatever was written to
// in the last few minutes, which is how a live run is recognised without
// reading it. Nothing polls while the tab is off screen.

const RUN_LIST_MS = 45000;
const RUN_WATCH_MS = 20000;
// A log written to this recently is being written to now. Discovery reports the
// age against the machine's OWN clock, so this holds across timezones.
const LIVE_S = 300;

let runList = [];
let runProblems = {};
const runDetail = new Map(); // key -> the last /progress answer
const runOpen = new Set();
const runPicked = new Set();
let runTimer = null;
let runBusy = false;

const runKey = (r) => `${r.dest}|${r.run}|${r.policy}`;

// A canvas is transparent: with nothing painted into it a line is drawn onto
// whatever the page happens to be, which in a dark browser is black on black.
// Same rule, and the same explicit two-theme palette, as the motion plots.
function chartPalette() {
  const dark = window.matchMedia
    && matchMedia('(prefers-color-scheme: dark)').matches;
  return dark
    ? {bg: '#151a21', grid: '#2b323c', axis: '#7c8797', ink: '#d6dbe3',
       series: ['#60a5fa', '#fbbf24', '#4ade80', '#f472b6', '#a78bfa', '#22d3ee']}
    : {bg: '#fbfcfe', grid: '#e6e9ef', axis: '#8b93a1', ink: '#31363f',
       series: ['#1d4ed8', '#c2410c', '#15803d', '#be185d', '#6d28d9', '#0e7490']};
}

// Device-pixel-ratio aware, so a line is one pixel and not a grey smear. The
// rollout page's version, which is the better of the two in this console.
function fitCanvas(cv) {
  const dpr = window.devicePixelRatio || 1;
  const w = Math.max(1, Math.round(cv.clientWidth * dpr));
  const h = Math.max(1, Math.round(cv.clientHeight * dpr));
  if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; }
  const ctx = cv.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cv.clientWidth, cv.clientHeight);
  return [ctx, cv.clientWidth, cv.clientHeight];
}

function fmtSeconds(s) {
  if (s === null || s === undefined) return '–';
  if (s < 90) return `${Math.round(s)}s`;
  if (s < 5400) return `${Math.round(s / 60)}m`;
  const h = Math.floor(s / 3600);
  return `${h}h${String(Math.round((s % 3600) / 60)).padStart(2, '0')}`;
}

const fmtInt = (n) => (n === null || n === undefined ? '–' : n.toLocaleString('en-GB'));
const fmtNum = (n, dp = 3) => (n === null || n === undefined ? '–' : n.toFixed(dp));

// ── Drawing ─────────────────────────────────────────────────────────────────

// series: [{label, points: [{x, y}]}]. Log-y by default, because this rig's
// runs go 10.1 -> 0.08: on a linear axis everything after the first few hundred
// steps is one flat line along the floor, which is exactly the part worth
// seeing.
function drawCurves(canvas, series, opts = {}) {
  const p = chartPalette();
  const [ctx, W, H] = fitCanvas(canvas);
  const logY = opts.logY !== false;
  const padL = 46, padR = 8, padT = 8, padB = 18;

  ctx.fillStyle = p.bg;
  ctx.fillRect(0, 0, W, H);

  const all = series.flatMap((s) => s.points).filter((q) => Number.isFinite(q.y));
  const usable = logY ? all.filter((q) => q.y > 0) : all;
  if (!usable.length) {
    ctx.fillStyle = p.axis;
    ctx.font = '12px system-ui, sans-serif';
    ctx.fillText(opts.empty || 'nothing logged yet', padL, H / 2);
    return;
  }

  const xs = usable.map((q) => q.x), ys = usable.map((q) => q.y);
  const x0 = Math.min(...xs), x1 = Math.max(...xs) || 1;
  let y0 = Math.min(...ys), y1 = Math.max(...ys);
  if (y1 === y0) { y1 = y0 + Math.abs(y0 || 1) * 0.1; }
  const tf = logY ? Math.log10 : (v) => v;
  const ty0 = tf(y0), ty1 = tf(y1);

  const X = (v) => padL + ((v - x0) / (x1 - x0 || 1)) * (W - padL - padR);
  const Y = (v) => H - padB - ((tf(v) - ty0) / (ty1 - ty0 || 1)) * (H - padT - padB);

  // Grid and the two labels that make the axis readable. Three lines, not ten:
  // this is a shape to read at a glance, not a figure for a paper.
  ctx.strokeStyle = p.grid;
  ctx.lineWidth = 1;
  ctx.fillStyle = p.axis;
  ctx.font = '10px system-ui, sans-serif';
  ctx.textBaseline = 'middle';
  for (const frac of [0, 0.5, 1]) {
    const value = logY ? 10 ** (ty0 + frac * (ty1 - ty0)) : y0 + frac * (y1 - y0);
    const y = Math.round(Y(value)) + 0.5;
    ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(W - padR, y); ctx.stroke();
    const text = value >= 100 ? value.toFixed(0)
      : value >= 1 ? value.toFixed(2) : value.toPrecision(2);
    ctx.fillText(text, 3, y);
  }
  ctx.textBaseline = 'alphabetic';
  ctx.fillText(opts.xLabel || '', padL, H - 4);
  const right = opts.xMax !== undefined ? opts.xMax : x1;
  const rightText = opts.xUnit === 'h' ? `${right.toFixed(1)} h` : fmtInt(Math.round(right));
  ctx.fillText(rightText, W - padR - ctx.measureText(rightText).width, H - 4);

  series.forEach((s, i) => {
    const points = (logY ? s.points.filter((q) => q.y > 0) : s.points)
      .filter((q) => Number.isFinite(q.y));
    if (!points.length) return;
    ctx.strokeStyle = s.colour || p.series[i % p.series.length];
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    points.forEach((q, k) => (k ? ctx.lineTo(X(q.x), Y(q.y)) : ctx.moveTo(X(q.x), Y(q.y))));
    ctx.stroke();
  });
}

function xOf(point, first) {
  if ($('#t-xaxis').value === 'hours' && point.time && first && first.time) {
    return (point.time - first.time) / 3600;
  }
  return point.step;
}

function seriesFor(detail, field = 'loss') {
  const points = detail.points || [];
  const first = points[0];
  return points
    .filter((p) => p[field] !== undefined && p[field] !== null)
    .map((p) => ({x: xOf(p, first), y: p[field]}));
}

const xUnit = () => ($('#t-xaxis').value === 'hours' ? 'h' : '');
const xLabel = () => ($('#t-xaxis').value === 'hours' ? 'hours' : 'step');

// ── The cards ───────────────────────────────────────────────────────────────

function stateOf(entry) {
  const detail = runDetail.get(runKey(entry));
  if (detail && detail.ok && detail.summary) return detail.summary.state;
  // Nothing has been read yet, so only the listing is known: a log written to
  // in the last few minutes is being written to now, and anything else is a
  // run that is not currently going. Which of "done", "failed" or "stalled" it
  // was needs the log, and the log is only fetched for what is being watched.
  return entry.age_s !== null && entry.age_s < LIVE_S ? 'running' : 'idle';
}

function isLive(entry) {
  return entry.age_s !== null && entry.age_s !== undefined && entry.age_s < LIVE_S;
}

function watched() {
  return runList.filter(
    (r) => runOpen.has(runKey(r)) || runPicked.has(runKey(r)) || isLive(r)
  );
}

function cardBody(entry) {
  const key = runKey(entry);
  const detail = runDetail.get(key);
  const s = (detail && detail.ok && detail.summary) || null;
  const state = stateOf(entry);
  const pct = s && s.fraction !== null ? Math.min(1, s.fraction) : 0;

  const nums = s
    ? `${fmtInt(s.step)} / ${fmtInt(s.total_steps)} · ${(pct * 100).toFixed(1)}%`
      + (state === 'running' ? ` · ${fmtSeconds(s.eta_s)} left` : '')
      + ` · loss ${fmtNum(s.loss_last)} (best ${fmtNum(s.loss_min)})`
      + (s.gpu_mem_gb_max ? ` · ${s.gpu_mem_gb_max.toFixed(1)} GB` : '')
    : entry.checkpoint
      ? `checkpoint ${fmtInt(entry.checkpoint)} · open to read the log`
      : 'open to read the log';

  return `
    <div class="bar wrap">
      <input type="checkbox" class="pick" ${runPicked.has(key) ? 'checked' : ''}
             title="compare this curve with the others">
      <b>${entry.run}</b>
      <span class="chip on">${entry.policy}</span>
      <span class="runstate ${state}">${state}</span>
      <span class="grow muted">${entry.dest}${entry.episodes ? ` · ${entry.episodes} ep` : ''}
        · written ${fmtSeconds(entry.age_s)} ago</span>
      <button class="link toggle">${runOpen.has(key) ? 'hide' : 'open'}</button>
      ${entry.id ? '<button class="danger stop">Stop</button>' : ''}
    </div>
    <div class="runbar ${state}"><span style="width:${(pct * 100).toFixed(1)}%"></span></div>
    <div class="muted runnums">${nums}</div>
    <div class="detail" ${runOpen.has(key) ? '' : 'hidden'}>
      <figure class="plot"><canvas class="loss"></canvas></figure>
      <div class="pair">
        <figure class="plot small">
          <figcaption class="muted runnums">GPU memory (GB)</figcaption>
          <canvas class="mem"></canvas>
        </figure>
        <figure class="plot small">
          <figcaption class="muted runnums">samples / s</figcaption>
          <canvas class="rate"></canvas>
        </figure>
      </div>
      <p class="err detail-err"></p>
      <pre class="logtail"></pre>
    </div>`;
}

function paintCard(node, entry) {
  const detail = runDetail.get(runKey(entry));
  if (!runOpen.has(runKey(entry))) return;
  const err = node.querySelector('.detail-err');
  err.textContent = detail && !detail.ok ? detail.problem : '';
  if (detail && detail.problems && detail.problems.length) {
    err.textContent = detail.problems.join('\n');
  }
  node.querySelector('.logtail').textContent = (detail && detail.tail) || '';
  const logY = $('#t-logy').checked;
  drawCurves(node.querySelector('canvas.loss'),
    detail ? [{label: 'loss', points: seriesFor(detail, 'loss')}] : [],
    {logY, xLabel: xLabel(), xUnit: xUnit(),
     xMax: detail && detail.summary ? detail.summary.total_steps : undefined,
     empty: detail ? 'nothing logged yet' : 'reading…'});
  // One axis each. Memory sits near 11 and throughput near 29, so a shared
  // scale would draw the memory trace as a flat line along the bottom -- and
  // memory is the number a machine's ceiling is read off.
  for (const [selector, field] of [['canvas.mem', 'gpu_mem_gb'],
                                   ['canvas.rate', 'samples_per_s']]) {
    drawCurves(node.querySelector(selector),
      detail ? [{label: field, points: seriesFor(detail, field)}] : [],
      {logY: false, xLabel: '', empty: ''});
  }
}

function renderRuns() {
  const box = $('#t-runs');
  const seen = new Set();
  for (const entry of runList) {
    const key = runKey(entry);
    seen.add(key);
    let node = box.querySelector(`[data-key="${CSS.escape(key)}"]`);
    if (!node) {
      node = document.createElement('div');
      node.className = 'runcard';
      node.dataset.key = key;
      box.appendChild(node);
      wireCard(node, entry);
    }
    // Rebuilt rather than patched field by field: a card is small, and the
    // canvases are redrawn from the detail afterwards anyway.
    node.innerHTML = cardBody(entry);
    node.classList.toggle('open', runOpen.has(key));
    paintCard(node, entry);
  }
  for (const node of [...box.children]) {
    if (!seen.has(node.dataset.key)) node.remove();
  }
  const live = runList.filter(isLive).length;
  $('#t-runstate').textContent = runList.length
    ? `${runList.length} run(s) on ${new Set(runList.map((r) => r.dest)).size} machine(s)`
      + (live ? `, ${live} being written to now` : '')
    : 'no runs found on any machine';
  $('#t-runs-err').textContent = Object.entries(runProblems)
    .map(([name, problem]) => `${name}: ${problem}`).join('\n');
  renderCompare();
}

function wireCard(node, entry) {
  const key = runKey(entry);
  node.addEventListener('click', async (event) => {
    const target = event.target;
    if (target.classList.contains('toggle')) {
      if (runOpen.has(key)) runOpen.delete(key); else runOpen.add(key);
      renderRuns();
      if (runOpen.has(key)) await refreshWatched();
    } else if (target.classList.contains('pick')) {
      if (target.checked) runPicked.add(key); else runPicked.delete(key);
      renderCompare();
      if (target.checked) await refreshWatched();
    } else if (target.classList.contains('stop')) {
      const record = runList.find((r) => runKey(r) === key);
      const yes = await confirmDialog({
        title: `Stop ${record.run}?`,
        body: `This cancels the run on ${record.dest}. Checkpoints already `
          + `written are kept, but training does not resume by itself.`,
        confirmLabel: 'Stop it',
      });
      if (!yes) return;
      try {
        await j(`/api/training/runs/${record.id}/stop`, {method: 'POST'});
      } catch (e) { $('#t-runs-err').textContent = e.message; }
    }
  });
}

function renderCompare() {
  const picked = runList.filter((r) => runPicked.has(runKey(r)));
  const figure = $('#t-compare');
  figure.hidden = picked.length < 2;
  if (figure.hidden) return;
  const p = chartPalette();
  const series = picked.map((entry, i) => {
    const detail = runDetail.get(runKey(entry));
    return {
      label: `${entry.dest}/${entry.run}/${entry.policy}`,
      colour: p.series[i % p.series.length],
      points: detail ? seriesFor(detail, 'loss') : [],
    };
  });
  $('#t-compare-legend').innerHTML = series
    .map((s) => `<span class="muted"><span class="swatch"
        style="background:${s.colour}"></span>${s.label}</span>`)
    .join('');
  drawCurves($('#t-compare-canvas'), series,
    {logY: $('#t-logy').checked, xLabel: xLabel(), xUnit: xUnit(),
     empty: 'reading…'});
}

// ── Talking to the machines ─────────────────────────────────────────────────

async function loadRunList(fresh = false) {
  const body = await j(`/api/training/discovered${fresh ? '?refresh=1' : ''}`);
  runList = body.runs || [];
  runProblems = body.problems || {};
  renderRuns();
}

async function refreshWatched(fresh = false) {
  // Sequential on purpose: each of these is an SSH, and firing six at a machine
  // at once to draw six lines is not a trade worth making.
  for (const entry of watched()) {
    const key = runKey(entry);
    const query = new URLSearchParams({
      dest: entry.dest, run: entry.run, policy: entry.policy,
    });
    if (fresh) query.set('refresh', '1');
    try {
      runDetail.set(key, await j(`/api/training/progress?${query}`));
    } catch (e) {
      runDetail.set(key, {ok: false, problem: e.message});
    }
    renderRuns();
  }
}

async function runsTick() {
  if (trainingVisible() && !runBusy) {
    runBusy = true;
    try {
      await loadRunList();
      await refreshWatched();
    } catch (e) {
      $('#t-runs-err').textContent = e.message;
    } finally { runBusy = false; }
  }
  clearTimeout(runTimer);
  runTimer = setTimeout(runsTick, watched().length ? RUN_WATCH_MS : RUN_LIST_MS);
}

$('#t-refresh').onclick = async () => {
  $('#t-runs-err').textContent = 'asking every machine…';
  try {
    await loadRunList(true);
    await refreshWatched(true);
    $('#t-runs-err').textContent = '';
  } catch (e) { $('#t-runs-err').textContent = e.message; }
};
$('#t-logy').onchange = renderRuns;
$('#t-xaxis').onchange = renderRuns;
window.addEventListener('resize', () => renderRuns());

const _chartPaneShown = window.onPaneShown;
window.onPaneShown = (name) => {
  if (_chartPaneShown) _chartPaneShown(name);
  if (name === 'training') runsTick();
};

window.addEventListener('load', () => {
  if (trainingVisible()) runsTick();
  else runTimer = setTimeout(runsTick, RUN_LIST_MS);
});
