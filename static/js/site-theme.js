'use strict';
try {
  if (localStorage.getItem('penvmap-theme') === 'light') document.documentElement.dataset.theme = 'light';
} catch { /* Use the default when browser storage is unavailable. */ }

document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('themeToggle').onclick = () => {
    const root = document.documentElement;
    root.dataset.theme = root.dataset.theme === 'light' ? 'dark' : 'light';
    try {
      localStorage.setItem('penvmap-theme', root.dataset.theme);
    } catch { /* The theme still applies for this visit. */ }
  };
});
window.addEventListener('pageshow', event => {
  if (event.persisted) {
    try { document.documentElement.dataset.theme = localStorage.getItem('penvmap-theme') === 'light' ? 'light' : 'dark'; } catch {}
  }
});
