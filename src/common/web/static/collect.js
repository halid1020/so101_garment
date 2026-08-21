// The Collect tab: rig readiness, the live view, and one collection session.
// The console never opens a device while a session runs — every tile here is
// either the session's own frames, proxied, or the console's idle preview.

let sessionState = null;
let liveStreams = [];
let controlsByMode = {};
let collectTimer = null;

function collectVisible() {
  return !document.querySelector('#pane-collect').hidden;
}

async function loadCollectConfig() {
  const cfg = await j('/api/collect/config');
  controlsByMode = cfg.controls || {};
  renderControls(controlsByMode[$('#c-input').value]);
  const box = $('#c-streams');
  box.innerHTML = '<legend>Camera streams</legend>';
  for (const cam of cfg.cameras) {
    const id = 'cam-' + cam.name;
    const label = document.createElement('label');
    label.className = 'row';
    label.innerHTML = `<input type="checkbox" id="${id}" value="${cam.name}"`
      + `${cam.enabled ? ' checked' : ''}> ${cam.name}`;
    box.appendChild(label);
  }
}

function pickedStreams() {
  return [...document.querySelectorAll('#c-streams input:checked')].map(c => c.value);
}

function sessionRequest() {
  return {
    name: $('#c-name').value.trim(),
    task: $('#c-task').value.trim(),
    streams: pickedStreams(),
    depth: $('#c-depth').checked,
    ee: $('#c-ee').checked,
    input: $('#c-input').value,
    sensor_view: $('#c-view').checked,
  };
}

function describePlan(plan) {
  const bits = [
    plan.resuming ? 'resume' : 'new dataset',
    `cameras ${plan.cameras.join('+') || 'none'}`,
    `depth ${plan.depth ? 'on' : 'off'}`,
    `ee ${plan.ee ? 'on' : 'off'}`,
    `fps ${plan.fps || 'default'}`,
  ];
  return bits.join(' · ') + (plan.warnings.length ? '\n' + plan.warnings.join('\n') : '');
}

async function checkPlan() {
  $('#c-err').textContent = ''; $('#c-plan').textContent = '';
  let plan;
  try {
    plan = await j('/api/session/plan', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(sessionRequest()),
    });
  } catch (e) { $('#c-err').textContent = e.message; return null; }
  $('#c-plan').textContent = describePlan(plan);
  $('#c-err').textContent = plan.refusals.join('\n');
  // A resumed dataset dictates its own streams; showing them ticked keeps the
  // form honest about what will actually be recorded.
  if (plan.resuming) {
    document.querySelectorAll('#c-streams input').forEach(c => {
      c.checked = plan.cameras.includes(c.value);
      c.disabled = true;
    });
    $('#c-depth').checked = plan.depth; $('#c-ee').checked = plan.ee;
  } else {
    document.querySelectorAll('#c-streams input').forEach(c => { c.disabled = false; });
  }
  return plan;
}

$('#c-check').onclick = checkPlan;
$('#c-name').onchange = checkPlan;
$('#c-input').onchange = () => {
  renderControls(controlsByMode[$('#c-input').value]);
  if ($('#c-name').value.trim()) checkPlan();
};

$('#c-start').onclick = async () => {
  const plan = await checkPlan();
  if (!plan || plan.refusals.length) return;
  $('#c-start').disabled = true;
  try {
    await j('/api/session/start', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(sessionRequest()),
    });
  } catch (e) { $('#c-err').textContent = e.message; }
  $('#c-start').disabled = false;
  await pollSession();
};

$('#c-episode').onclick = async () => {
  try {
    await j('/api/session/episode',
            {method: 'POST', headers: {'Content-Type': 'application/json'}, body: '{}'});
  } catch (e) { alert('episode key failed: ' + e.message); }
  await pollSession();
};

// Enabling the arms moves them, so it lives where the operator can see them --
// the headset, in Quest mode. A leader session has no headset, and when the
// console started it there is no keyboard either, so there the page is the only
// surface and this button is the way in. The session decides: a mode that does
// not allow the key answers with its own refusal.
$('#c-enable').onclick = async () => {
  const ok = await confirmDialog({
    title: 'Enable both arms?',
    body: 'Torque goes on and both followers MOVE to the ready pose and hold '
      + 'it. Stand clear of the arms before confirming.',
    confirmLabel: 'Enable arms',
  });
  if (!ok) return;
  try {
    await j('/api/session/key', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({key: 'y'}),
    });
  } catch (e) { alert('enable failed: ' + e.message); }
  await pollSession();
};

$('#c-stop').onclick = async () => {
  const ok = await confirmDialog({
    title: 'End the collection session?',
    body: 'An episode in progress is saved and the arms are parked before it '
      + 'exits.',
    confirmLabel: 'Stop session',
  });
  if (!ok) return;
  try {
    await j('/api/session/stop',
            {method: 'POST', headers: {'Content-Type': 'application/json'}, body: '{}'});
  } catch (e) { alert('stop failed: ' + e.message); }
  await pollSession();
};

$('#c-arms').onclick = async () => {
  const on = $('#c-arms').dataset.on === '1';
  $('#c-arms').disabled = true;
  try {
    await j(on ? '/api/preview/arms/stop' : '/api/preview/arms/start',
            {method: 'POST', headers: {'Content-Type': 'application/json'}, body: '{}'});
  } catch (e) { alert(e.message); }
  $('#c-arms').disabled = false;
  await pollSession();
};

$('#c-preview').onclick = async () => {
  const on = $('#c-preview').dataset.on === '1';
  try {
    await j(on ? '/api/preview/stop' : '/api/preview/start',
            {method: 'POST', headers: {'Content-Type': 'application/json'}, body: '{}'});
  } catch (e) { alert(e.message); }
  await refreshTiles();
};

// ── Controls and signals ────────────────────────────────────────────────────

// The steps come from the session (common/recording/controls.py), which is also
// what the terminal prints, so this pane cannot describe a rig it is not
// driving. Anything that moves an arm names the surface it lives on.
function renderControls(steps) {
  if (!steps || !steps.length) return;
  $('#c-controls').innerHTML = steps.map(s => {
    const here = s.where.includes('this page');
    return `<tr class="${here ? 'lvl-OK' : ''}">
      <td>${s.key || '·'}</td><td>${s.what}</td>
      <td class="muted">${s.where}</td></tr>`;
  }).join('');
}

const JOINT_ROWS = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex',
                    'wrist_roll', 'gripper'];

// The table is drawn whether or not a session is running: an empty one saying
// where the numbers come from is honest, while a pane that simply omits the
// arms reads as a console that cannot see them.
function renderJoints(monitor) {
  const sig = monitor && monitor.joints;  // 'j' is the fetch helper here
  const preview = !!(monitor && monitor.source === 'preview');
  const num = (v) => (v === null || v === undefined) ? '--' : v.toFixed(2);
  // A command older than a frame is what makes the recorded action fall back to
  // the measured state, so a stale column is dimmed rather than left to look
  // like a live one.
  const cmdClass = (side) => (sig && sig[side].fresh) ? '' : ' class="rate"';
  const at = (side, half, name) => sig ? sig[side][half][name] : null;
  let html = '<tr><th></th><th colspan="2">left</th><th colspan="2">right</th></tr>'
    + '<tr><th></th><th>state</th><th>cmd</th><th>state</th><th>cmd</th></tr>';
  for (const name of JOINT_ROWS) {
    html += `<tr><th>${name}</th>` + ['left', 'right'].map(side =>
      `<td>${num(at(side, 'state', name))}</td>`
      + `<td${cmdClass(side)}>${num(at(side, 'command', name))}</td>`).join('')
      + '</tr>';
  }
  $('#c-joints').innerHTML = html;

  if (!sig) {
    $('#c-teleop').textContent =
      'no session — start one, or read the arms from here';
    $('#c-drift').textContent = '';
    return;
  }
  if (preview) {
    $('#c-teleop').textContent = 'reading the arms — torque off, nothing commanded';
    $('#c-drift').textContent =
      `last read ${((monitor.joint_drift_s || 0) * 1000).toFixed(0)} ms ago`;
    return;
  }
  const leader = sessionState && sessionState.input === 'leader';
  $('#c-teleop').textContent = monitor.teleop_active
    ? (leader ? 'following the leaders' : 'teleoperating')
    : (leader ? 'not following — enable the arms first'
              : 'not teleoperating — hold both grips');
  const drift = monitor.joint_drift_s;
  const stale = ['left', 'right'].filter(side => !sig[side].fresh);
  $('#c-drift').textContent =
    (drift === null || drift === undefined
      ? 'joints --' : `joints ${(drift * 1000).toFixed(0)} ms`)
    + (stale.length ? ` · no fresh command: ${stale.join(', ')}` : '');
}

// ── Live tiles ──────────────────────────────────────────────────────────────

async function refreshTiles() {
  let body;
  try { body = await j('/api/live/streams'); }
  catch (e) { return; }
  const names = body.streams || [];
  const preview = body.source === 'preview';
  $('#c-preview').dataset.on = (preview && names.length) ? '1' : '0';
  $('#c-preview').textContent = (preview && names.length)
    ? 'Stop preview' : 'Start preview';
  $('#c-preview').disabled = !preview;
  $('#live-source').textContent = names.length
    ? (preview ? 'preview (no session running)' : 'live from the session')
    : 'no live view';
  if (names.join() === liveStreams.join()) return;
  liveStreams = names;
  const tiles = $('#live-tiles'); tiles.innerHTML = '';
  for (const name of names) {
    const fig = document.createElement('figure');
    fig.className = 'tile';
    // The query parameter is a cache-buster: an <img> pointed at a multipart
    // stream keeps the connection open, so a stale one must not be reused.
    fig.innerHTML = `<img src="/api/live/${encodeURIComponent(name)}.mjpg?t=${Date.now()}">
                     <figcaption>${name}</figcaption>`;
    tiles.appendChild(fig);
  }
}

// ── Session polling ─────────────────────────────────────────────────────────

function describeStatus(s) {
  if (!s.running) return 'no session';
  const m = s.monitor;
  if (!m) return `session running (pid ${s.pid}) — waiting for its live monitor`;
  const rec = m.recorder;
  const bits = [`arms ${m.arms}`];
  if (rec) {
    bits.push(rec.state);
    bits.push(rec.episodes_goal
      ? `episodes ${rec.episodes_done}/${rec.episodes_goal}`
      : `episodes ${rec.episodes_done}`);
    if (rec.recording) bits.push(`${rec.current_frames} frames`);
  }
  const stale = (m.streams || []).filter(x => x.age_s === null || x.age_s > 1.0);
  if (stale.length) bits.push(`stale: ${stale.map(x => x.name).join(', ')}`);
  return bits.join(' · ');
}

async function pollSession() {
  let s;
  try { s = await j('/api/session'); } catch (e) { return; }
  sessionState = s;
  $('#session-state').textContent = describeStatus(s);
  $('#session-form').hidden = s.running;
  $('#session-live').hidden = !s.running;
  if (s.running) {
    $('#c-running').textContent =
      `${s.resuming ? 'resuming' : 'recording into'} '${s.name}' — ${s.task}\n`
      + `cameras ${(s.cameras || []).join('+')} · depth ${s.depth ? 'on' : 'off'}`
      + ` · ee ${s.ee ? 'on' : 'off'}`;
    const m = s.monitor, rec = m && m.recorder;
    const armed = !!m && m.arms === 'ENABLED';
    const leading = s.input === 'leader';
    $('#c-enable').hidden = !leading;
    $('#c-enable').disabled = armed;
    $('#c-enable').title = armed ? 'the arms are already enabled' : '';
    $('#c-surface').textContent = leading
      ? 'This session is driven by the leader arms; its control keys are read '
        + 'from the terminal that started it, so the ones this page may press '
        + 'are here. Stand clear before enabling.'
      : 'Everything below that moves an arm stays on the headset or the '
        + 'keyboard at the rig — stand clear before it does.';
    $('#c-episode').disabled = !armed;
    $('#c-episode').title = armed ? ''
      : (leading ? 'enable the arms first'
                 : 'enable the arms first (button Y on the headset)');
    $('#c-episode').textContent = (rec && rec.recording)
      ? 'Stop episode and save' : 'Start episode';
    $('#c-stop').disabled = !!s.stopping;
    $('#c-stop').textContent = s.stopping ? 'Stopping…' : 'Stop session';
  }
  renderJoints(s.monitor || s.preview_joints);
  $('#c-arms').hidden = !!s.running;
  $('#c-arms').dataset.on = s.preview_joints ? '1' : '0';
  $('#c-arms').textContent = s.preview_joints ? 'Stop reading' : 'Read the arms';
  if (s.running) renderControls(s.controls);
  $('#session-log').textContent = (s.tail || []).slice(-200).join('\n');
  $('#session-log').scrollTop = $('#session-log').scrollHeight;
  await refreshTiles();
}

async function loadPreflight() {
  $('#preflight-body').textContent = 'checking…';
  let body;
  try { body = await j('/api/preflight'); }
  catch (e) { $('#preflight-body').textContent = e.message; return; }
  const rows = body.checks.map(c =>
    `<tr class="lvl-${c.level}"><td>${c.level}</td><td>${c.name}</td>`
    + `<td>${c.detail}</td></tr>`).join('');
  $('#preflight-body').innerHTML =
    (body.hardware ? '' : '<p class="muted">devices are in use — file checks only</p>')
    + `<table class="checks">${rows}</table>`;
}

$('#preflight').ontoggle = () => { if ($('#preflight').open) loadPreflight(); };

// Poll only while the tab is on screen: the live tiles are streams, and the
// status is only interesting to someone looking at it. A running session is
// polled four times as often, because the joint table is worth watching move;
// one request is in flight at a time, so a slow monitor cannot queue them up.
let collectBusy = false;

async function collectTick() {
  if (collectVisible() && !collectBusy) {
    collectBusy = true;
    try { await pollSession(); }
    catch (e) { /* the next tick tries again */ }
    finally { collectBusy = false; }
  }
  const period = (sessionState && sessionState.running && collectVisible())
    ? 400 : 1500;
  clearTimeout(collectTimer);
  collectTimer = setTimeout(collectTick, period);
}

window.addEventListener('load', () => {
  loadCollectConfig().catch(e => { $('#c-err').textContent = e.message; });
  collectTimer = setTimeout(collectTick, 300);
  // app.js chooses the pane before this file is parsed, so a console opened
  // straight on #collect has to be told once, here.
  if (collectVisible()) window.onPaneShown('collect');
});

window.onPaneShown = (name) => {
  if (name === 'collect') {
    // The dataset names are the same list the Datasets tab shows.
    const list = $('#c-names');
    list.innerHTML = datasets.map(d => `<option value="${d.name}">`).join('');
    pollSession();
  }
};
