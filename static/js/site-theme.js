'use strict';

(() => {
  const key = 'penvmap-theme';
  const system = window.matchMedia('(prefers-color-scheme: dark)');
  const root = document.documentElement;
  let override;

  function applyTheme() {
    root.dataset.theme = override || (system.matches ? 'dark' : 'light');
  }

  function restoreTheme() {
    try {
      const value = localStorage.getItem(key);
      override = value === 'light' || value === 'dark' ? value : null;
    } catch { /* Keep the current page preference if storage is unavailable. */ }
    applyTheme();
  }

  restoreTheme();
  system.addEventListener('change', applyTheme);
  window.addEventListener('pageshow', event => {
    if (event.persisted) restoreTheme();
  });
  window.addEventListener('storage', event => {
    if (event.key === key || event.key === null) restoreTheme();
  });
  document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('themeToggle').onclick = () => {
      override = root.dataset.theme === 'light' ? 'dark' : 'light';
      applyTheme();
      try { localStorage.setItem(key, override); } catch { /* Apply for this visit. */ }
    };
  });
})();
