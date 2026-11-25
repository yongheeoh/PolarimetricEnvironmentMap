'use strict';
function element(tag, value, cls) {
  const node = document.createElement(tag);
  if (value) node.textContent = value;
  if (cls) node.className = cls;
  return node;
}
function luminancePreview(source) {
  return source + '?v=luma1';
}

function polarizationImage(lightSource) {
  return luminancePreview(document.documentElement.dataset.theme === 'light'
    ? lightSource : lightSource.replace(/\.jpg$/, '-dark.jpg'));
}

function observeTheme(callback) {
  const observer = new MutationObserver(callback);
  observer.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
  return observer;
}

function makeCard(scene) {
  const card = element('a', '', 'card');
  card.href = 'scene.html?id=' + encodeURIComponent(scene.id);
  const image = new Image();
  image.decoding = 'async';
  image.alt = '';
  image.dataset.src = scene.card;
  const preview = document.createElement('div');
  preview.className = 'preview';
  const overlay = new Image();
  overlay.decoding = 'async';
  overlay.alt = '';
  overlay.dataset.src = polarizationImage(scene.hover);
  overlay.onload = () => overlay.classList.add('ready');
  preview.append(image, overlay);
  card.append(preview, element('strong', scene.title), element('span',
    [scene.setting, scene.collection, '#' + scene.number].join(' · ')));
  return card;
}

function loadCardImage(image, priority) {
  if (image.hasAttribute('src')) {
    if (priority === 'high') image.fetchPriority = priority;
    return;
  }
  image.fetchPriority = priority;
  image.src = image.dataset.src;
}

let stopCardObservers;
function renderCards(container, scenes) {
  stopCardObservers?.();
  const cards = scenes.map(makeCard);
  container.replaceChildren(...cards);
  const visible = new Set();
  const nearby = new Set();
  let frame = 0;
  let stopped = false;
  let backgroundLoads = 0;
  function schedule() {
    if (!stopped && !frame) frame = requestAnimationFrame(update);
  }
  function update() {
    frame = 0;
    let foregroundLoading = false;
    for (const card of visible) {
      const image = card.querySelector('img');
      loadCardImage(image, 'high');
      if (!image.complete) foregroundLoading = true;
    }
    if (foregroundLoading) return;
    // Warm hovered views first, then the ordinary thumbnails just beyond the viewport.
    for (const [group, selector] of [[visible, 'img:last-child'], [nearby, 'img:first-child']]) {
      for (const card of group) {
        if (backgroundLoads >= 2) return;
        const image = card.querySelector(selector);
        if (image.hasAttribute('src')) continue;
        backgroundLoads++;
        loadCardImage(image, 'low');
        image.decode().catch(() => {}).finally(() => {
          backgroundLoads--;
          schedule();
        });
      }
    }
  }
  function observe(group, rootMargin) {
    const observer = new IntersectionObserver(entries => {
      for (const entry of entries) {
        if (entry.isIntersecting) group.add(entry.target);
        else group.delete(entry.target);
      }
      schedule();
    }, { rootMargin });
    cards.forEach(card => observer.observe(card));
    return observer;
  }
  for (const card of cards) {
    const [image, overlay] = card.querySelectorAll('img');
    image.onload = image.onerror = schedule;
    const prepareHover = () => loadCardImage(overlay, 'high');
    card.addEventListener('mouseenter', prepareHover);
    card.addEventListener('focus', prepareHover);
  }
  const viewport = observe(visible, '0px');
  const lookahead = observe(nearby, '300px 0px');
  const themeObserver = observeTheme(() => {
    cards.forEach((card, i) => {
      const overlay = card.querySelector('img:last-child');
      const source = polarizationImage(scenes[i].hover);
      if (overlay.dataset.src === source) return;
      overlay.classList.remove('ready');
      overlay.removeAttribute('src');
      overlay.dataset.src = source;
      if (card.matches(':hover, :focus-visible')) loadCardImage(overlay, 'high');
    });
    schedule();
  });
  stopCardObservers = () => {
    themeObserver.disconnect();
    stopped = true;
    viewport.disconnect();
    lookahead.disconnect();
    cancelAnimationFrame(frame);
  };
}
