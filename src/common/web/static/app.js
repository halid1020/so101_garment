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

showPane((location.hash || '#datasets').slice(1));

fetch('/api/console').then(r => r.json()).then(c => {
  document.querySelector('#rootdir').textContent = c.root;
}).catch(() => {});
