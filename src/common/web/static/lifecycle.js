// Whole-dataset operations: new (name check), rename, merge, delete.
// The episode pane in datasets.js works inside one dataset; this file works on
// the collection directory itself.

let selectedDataset = null;

function dlg(id) { return document.querySelector('#' + id); }

function openDialog(id) { dlg(id).showModal(); }

document.querySelectorAll('[data-close]').forEach(b => {
  b.onclick = () => dlg(b.dataset.close).close();
});

function currentMeta() {
  return datasets.find(d => d.name === selectedDataset) || null;
}

function refreshButtons() {
  const has = !!selectedDataset;
  $('#ds-rename').disabled = !has;
  $('#ds-remove').disabled = !(has && allowDelete);
  $('#ds-remove').title = allowDelete ? ''
    : 'restart the console with --allow-delete to delete datasets';
}

window.onDatasetSelected = (name) => { selectedDataset = name; refreshButtons(); };
window.onDatasetsLoaded = () => {
  if (selectedDataset && !datasets.some(d => d.name === selectedDataset)) {
    selectedDataset = null;
  }
  refreshButtons();
};

// ── New ─────────────────────────────────────────────────────────────────────

$('#ds-new').onclick = () => {
  $('#new-err').textContent = ''; $('#new-cmd').hidden = true;
  openDialog('dlg-new');
  $('#new-name').focus();
};

$('#new-go').onclick = async () => {
  const name = $('#new-name').value.trim();
  const task = $('#new-task').value.trim();
  $('#new-cmd').hidden = true;
  let check;
  try {
    check = await j('/api/datasets/check-name', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({name}),
    });
  } catch (e) { $('#new-err').textContent = e.message; return; }
  if (!check.ok) { $('#new-err').textContent = check.problem; return; }
  if (!task) { $('#new-err').textContent = 'give the dataset an instruction'; return; }
  $('#new-err').textContent = '';
  const root = $('#rootdir').textContent;
  $('#new-cmd').textContent =
    `venv/bin/python tool/collect_dataset.py \\\n`
    + `    --dir ${root} --name ${name} --task ${JSON.stringify(task)}`;
  $('#new-cmd').hidden = false;
};

// ── Rename ──────────────────────────────────────────────────────────────────

$('#ds-rename').onclick = () => {
  if (!selectedDataset) return;
  $('#rename-err').textContent = '';
  $('#rename-to').value = selectedDataset;
  openDialog('dlg-rename');
  $('#rename-to').focus();
};

$('#rename-go').onclick = async () => {
  const from = selectedDataset;
  const to = $('#rename-to').value.trim();
  if (!from || to === from) { dlg('dlg-rename').close(); return; }
  try {
    await j(`/api/datasets/${encodeURIComponent(from)}/rename`, {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({new: to}),
    });
  } catch (e) { $('#rename-err').textContent = e.message; return; }
  dlg('dlg-rename').close();
  selectedDataset = to;
  await loadDatasets();
  resetSelection(to);
  document.querySelectorAll('#ds-list li').forEach(li => {
    if (li.querySelector('.dsname') &&
        li.querySelector('.dsname').textContent === to) li.classList.add('sel');
  });
};

// ── Delete ──────────────────────────────────────────────────────────────────

$('#ds-remove').onclick = async () => {
  const meta = currentMeta();
  if (!meta) return;
  const size = meta.bytes ? ` (${humanSize(meta.bytes)})` : '';
  const what = meta.stillborn ? 'this empty dataset'
    : `${meta.episodes} recording(s)`;
  if (!confirm(`Delete '${meta.name}'${size} for good?\n\n`
      + `${what} will be removed from the drive. This cannot be undone.`)) return;
  try {
    await j(`/api/datasets/${encodeURIComponent(meta.name)}/remove`,
            {method: 'POST', headers: {'Content-Type': 'application/json'}, body: '{}'});
  } catch (e) { alert('delete failed: ' + e.message); return; }
  selectedDataset = null;
  resetSelection(null);
  await loadDatasets();
};

// ── Merge ───────────────────────────────────────────────────────────────────

function mergePicked() {
  return [...document.querySelectorAll('#merge-list input:checked')]
    .map(c => c.value);
}

$('#ds-merge').onclick = () => {
  const ul = $('#merge-list'); ul.innerHTML = '';
  for (const d of datasets) {
    if (d.stillborn) continue;
    const li = document.createElement('li');
    li.innerHTML = `<label><input type="checkbox" value="${d.name}"> `
      + `${d.name} <span class="muted">${datasetLine(d)}</span></label>`;
    ul.appendChild(li);
  }
  $('#merge-err').textContent = '';
  $('#merge-progress').hidden = true;
  $('#merge-go').disabled = false;
  openDialog('dlg-merge');
  ul.querySelectorAll('input').forEach(c => { c.onchange = checkMerge; });
  $('#merge-name').oninput = checkMerge;
};

async function checkMerge() {
  const names = mergePicked();
  const name = $('#merge-name').value.trim();
  if (names.length < 2 || !name) { $('#merge-err').textContent = ''; return; }
  try {
    const r = await j('/api/datasets/merge-check', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({names, name}),
    });
    $('#merge-err').textContent = r.ok ? '' : r.reasons.join('\n');
  } catch (e) { $('#merge-err').textContent = e.message; }
}

$('#merge-go').onclick = async () => {
  const names = mergePicked();
  const name = $('#merge-name').value.trim();
  let job;
  try {
    job = await j('/api/datasets/merge', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({names, name}),
    });
  } catch (e) { $('#merge-err').textContent = e.message; return; }
  $('#merge-go').disabled = true;
  $('#merge-progress').hidden = false;
  pollMerge(job.id);
};

async function pollMerge(id) {
  // Merging re-encodes every episode of every source, so this runs for as long
  // as it takes; the dialog stays open and says where it is.
  let job;
  try { job = await j(`/api/datasets/jobs/${id}`); }
  catch (e) { $('#merge-err').textContent = e.message; return; }
  $('#merge-progress').textContent = `${job.state}: ${job.message}`;
  if (job.state === 'running') { setTimeout(() => pollMerge(id), 2000); return; }
  $('#merge-go').disabled = false;
  if (job.state === 'failed') { $('#merge-err').textContent = job.message; return; }
  await loadDatasets();
}
