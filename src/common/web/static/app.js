// App shell: the tab bar and the header's collection-directory readout. Each
// pane keeps its own script; this file only decides which one is on screen.
const panes = () => document.querySelectorAll('.pane');

function showPane(name) {
  panes().forEach(p => { p.hidden = (p.id !== 'pane-' + name); });
  document.querySelectorAll('.tab').forEach(t => {
    t.classList.toggle('sel', t.dataset.pane === name);
  });
  location.hash = name;
  if (window.onPaneShown) window.onPaneShown(name);
}

document.querySelectorAll('.tab').forEach(t => {
  t.onclick = () => showPane(t.dataset.pane);
});

// '#sensors' is what the tab used to be called; keep an old link working.
const _hash = (location.hash || '#datasets').slice(1);
showPane(_hash === 'sensors' ? 'signals' : _hash);

// One confirmation for the deletes that cannot be undone. Native confirm() is
// blocking and a browser may suppress it after a few; this one is neither, and
// it says the same things every time: what goes, how much of it, and that it
// does not come back. Marking an episode is NOT confirmed -- Restore undoes it.
function confirmDialog({title, body, confirmLabel = 'Delete'}) {
  const d = document.querySelector('#dlg-confirm');
  document.querySelector('#confirm-title').textContent = title;
  document.querySelector('#confirm-body').textContent = body;
  document.querySelector('#confirm-yes').textContent = confirmLabel;
  return new Promise(resolve => {
    d.returnValue = 'no';
    d.onclose = () => resolve(d.returnValue === 'yes');
    d.showModal();
  });
}
