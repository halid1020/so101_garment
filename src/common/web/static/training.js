// The Training tab: send a collected dataset to a GPU machine, and watch it.
//
// Nothing here decides whether a run can work. Every refusal comes from
// /api/training/plan, which is the same rule the terminal launcher applies, so
// the page cannot be more permissive than the thing that actually submits.

let trainingConfig = null;
let trainingRuns = [];
let trainingTimer = null;
let trainingBusy = false;

function trainingVisible() {
  return !document.querySelector('#pane-training').hidden;
}

async function loadTrainingConfig() {
  const cfg = await j('/api/training/config');
  trainingConfig = cfg;

  const ds = $('#t-dataset');
  const chosen = ds.value;
  ds.innerHTML = cfg.datasets
    .map(d => `<option value="${d.name}">${d.name} (${d.episodes} ep)</option>`)
    .join('');
  if (chosen) ds.value = chosen;

  const dest = $('#t-dest');
  const chosenDest = dest.value;
  dest.innerHTML = cfg.destinations
    .map(d => `<option value="${d.name}">${d.name} — ${d.kind}</option>`)
    .join('');
  if (chosenDest) dest.value = chosenDest;

  const box = $('#t-policies');
  box.innerHTML = '<legend>Policies</legend>';
  for (const p of cfg.policies) {
    const label = document.createElement('label');
    label.className = 'row';
    // A policy the installed LeRobot has never heard of is SHOWN and left
    // untickable: hiding it would make the gate look like a missing feature,
    // and the refusal names the two files to change.
    label.innerHTML = `<input type="checkbox" value="${p.name}"`
      + `${p.available ? '' : ' disabled'}> ${p.name}`
      + (p.available ? '' : ' <span class="stream-absent">— not in this LeRobot</span>');
    if (!p.available) label.title = p.problem;
    box.appendChild(label);
  }
  describeDataset();
  describeDest();
}

function describeDataset() {
  const d = (trainingConfig.datasets || []).find(x => x.name === $('#t-dataset').value);
  $('#t-dataset-detail').textContent = d
    ? `${d.episodes} episodes, ${d.frames} frames — ${d.cameras.join(', ')}`
    : '';
  const composites = Object.entries(trainingConfig.composites || {});
  $('#t-cameras-help').textContent =
    "'all' is the dataset's own cameras. "
    + composites.map(([n, parts]) => `${n} tiles ${parts.join(' + ')} into one`).join('; ');
}

function describeDest() {
  const d = (trainingConfig.destinations || []).find(x => x.name === $('#t-dest').value);
  if (!d) { $('#t-dest-detail').textContent = ''; return; }
  const limits = Object.entries(d.limits || {})
    .map(([p, caps]) => `${p} ${Object.entries(caps).map(([k, v]) => `${k}≤${v}`).join(' ')}`);
  $('#t-dest-detail').textContent =
    `${d.ssh} — stages to ${d.stage}`
    + (limits.length ? ` — measured ceilings: ${limits.join(', ')}` : '');
}

function trainingRequest() {
  return {
    dataset: $('#t-dataset').value,
    dest: $('#t-dest').value,
    policies: [...document.querySelectorAll('#t-policies input:checked')].map(c => c.value),
    cameras: $('#t-cameras').value.trim() || 'all',
    slots: $('#t-slots').value.trim() || '-',
    steps: $('#t-steps').value.trim() || '-',
    batch: $('#t-batch').value.trim() || '-',
    hours: $('#t-hours').value.trim() || null,
    restage: $('#t-restage').checked,
  };
}

function renderPlan(plan) {
  const table = $('#t-rows');
  table.innerHTML = '';
  for (const row of plan.rows) {
    const tr = document.createElement('tr');
    tr.className = 'lvl-OK';
    tr.innerHTML = `<td>${row.policy}</td><td>${row.cameras}</td>`
      + `<td>${row.resolved_steps} steps</td><td>batch ${row.resolved_batch}</td>`
      + `<td>${row.hours} h</td>`;
    table.appendChild(tr);
  }
  $('#t-err').textContent = (plan.refusals || []).join('\n');
  return (plan.refusals || []).length === 0;
}

async function checkTraining() {
  $('#t-err').textContent = 'checking…';
  const plan = await j('/api/training/plan', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(trainingRequest()),
  });
  return renderPlan(plan);
}

async function startTraining() {
  const request = trainingRequest();
  const ok = await checkTraining();
  if (!ok) return;
  const yes = await confirmDialog({
    title: `Train on ${request.dest}?`,
    body: `${request.policies.join(', ')} on ${request.dataset}. The dataset is `
      + `copied to ${request.dest} first, which can take several minutes, and `
      + `the run then holds that machine's GPU for hours.`,
    confirmLabel: 'Start training',
  });
  if (!yes) return;
  await j('/api/training/start', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(request),
  });
  // Staging outlives this request, so the dock is where it is watched.
  if (window.pollJobs) pollJobs();
  loadRuns();
}

async function testReach() {
  $('#t-err').textContent = 'connecting…';
  const out = await j('/api/training/reach', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({dest: $('#t-dest').value}),
  });
  $('#t-err').textContent = out.ok ? '' : out.problem;
  $('#t-dest-detail').textContent = out.ok
    ? `${$('#t-dest').value} answered.`
    : $('#t-dest-detail').textContent;
}

async function loadRuns() {
  trainingRuns = await j('/api/training/runs');
  const box = $('#t-runs');
  box.innerHTML = '';
  $('#t-state').textContent = trainingRuns.length
    ? `${trainingRuns.length} run(s) launched from this console`
    : 'nothing launched';
  for (const run of trainingRuns) {
    const div = document.createElement('div');
    div.className = 'pad';
    div.innerHTML = `<b>${run.id}</b> — ${run.dest} — ${run.dataset}`
      + ` (${run.episodes} ep)<br><span class="muted">`
      + `${(run.policies || []).join(', ')} · ${run.started}</span>`;
    const bar = document.createElement('div');
    bar.className = 'bar';
    const status = document.createElement('span');
    status.className = 'muted';
    const ask = document.createElement('button');
    ask.textContent = 'Status';
    ask.onclick = async () => {
      status.textContent = 'asking…';
      try {
        const out = await j(`/api/training/runs/${run.id}/status`);
        status.textContent = out.text;
        status.className = out.ok ? 'muted' : 'err';
      } catch (e) { status.textContent = e.message; status.className = 'err'; }
    };
    const stop = document.createElement('button');
    stop.textContent = 'Stop';
    stop.className = 'danger';
    stop.onclick = async () => {
      const yes = await confirmDialog({
        title: `Stop ${run.id}?`,
        body: `This cancels the run on ${run.dest}. Checkpoints already written `
          + `are kept, but the training itself does not resume by itself.`,
        confirmLabel: 'Stop it',
      });
      if (!yes) return;
      try {
        const out = await j(`/api/training/runs/${run.id}/stop`, {method: 'POST'});
        status.textContent = out.stopped;
      } catch (e) { status.textContent = e.message; status.className = 'err'; }
    };
    bar.append(ask, stop, status);
    div.appendChild(bar);
    box.appendChild(div);
  }
}

$('#t-check').onclick = () =>
  checkTraining().catch(e => { $('#t-err').textContent = e.message; });
$('#t-reach').onclick = () =>
  testReach().catch(e => { $('#t-err').textContent = e.message; });
$('#t-start').onclick = () =>
  startTraining().catch(e => { $('#t-err').textContent = e.message; });
$('#t-refresh').onclick = () =>
  loadRuns().catch(e => { $('#t-err').textContent = e.message; });
$('#t-dataset').onchange = describeDataset;
$('#t-dest').onchange = describeDest;

// Nothing here streams, and asking a machine anything costs an SSH — so the
// list is refreshed slowly and only while the tab is on screen. A launch in
// progress is watched in the job dock, which polls on its own.
async function trainingTick() {
  if (trainingVisible() && !trainingBusy) {
    trainingBusy = true;
    try { await loadRuns(); }
    catch (e) { /* the next tick tries again */ }
    finally { trainingBusy = false; }
  }
  clearTimeout(trainingTimer);
  trainingTimer = setTimeout(trainingTick, 10000);
}

const _trainingPaneShown = window.onPaneShown;
window.onPaneShown = (name) => {
  if (_trainingPaneShown) _trainingPaneShown(name);
  if (name === 'training') {
    loadTrainingConfig().catch(e => { $('#t-err').textContent = e.message; });
    loadRuns().catch(() => {});
  }
};

window.addEventListener('load', () => {
  trainingTimer = setTimeout(trainingTick, 1200);
  if (trainingVisible()) window.onPaneShown('training');
});
