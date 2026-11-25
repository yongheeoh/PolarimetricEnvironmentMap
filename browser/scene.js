'use strict';
const byId = id => document.getElementById(id);
const id = new URLSearchParams(location.search).get('id');
const active = window.atlasScenes.find(scene => scene.id === id);
function showLegend(name) {
  const legend = byId('channelLegend');
  legend.hidden = name === 's0';
  if (legend.hidden) return;
  const degrees = ['DoLP', 'DoP', 'DoCP'];
  let key, title, ticks;
  if (degrees.includes(name)) {
    key = 'degree'; title = name; ticks = ['0', '0.5', '1'];
  } else if (name.endsWith('/s0')) {
    key = 'signed'; title = name; ticks = ['−1', '0', '+1'];

  } else if (name === 'CoP') {
    key = 'cop-angle'; title = 'CoP · ellipticity angle'; ticks = ['−45°', '0°', '+45°'];
  } else if (name === 'ToP') {
    key = 'top-angle'; title = 'ToP · absolute ellipticity angle'; ticks = ['0°', '22.5°', '45°'];
  } else {
    key = 'angle';
    title = 'AoLP'; ticks = ['0°', '45°', '90°', '135°', '180°'];
  }
  legend.setAttribute('aria-label', title);
  byId('legendRamp').src = 'assets/legends/' + key + '.png';
  byId('legendTicks').replaceChildren(...ticks.map(value => {
    const tick = document.createElement('span');
    tick.textContent = value;
    return tick;
  }));
}
function renderPreviews(scene) {
  const base = 'assets/' + scene.id;
  const previews = [
    ['s0', scene.card, scene.detail],
    ['AoLP × DoLP', scene.hover, scene.aolp],
    ['AoLP', base + '-angle-thumb.jpg', base + '-angle.jpg'],
    ['DoLP', base + '-dolp-thumb.webp', scene.dolp],
  ];
  const fullStokes = scene.collection === 'Full Stokes';
  const components = [];
  if (fullStokes) components.push(['DoP', 'dop'], ['DoCP', 'docp'], ['CoP', 'cop'], ['ToP', 'top']);
  const channels = fullStokes ? [1, 2, 3] : [1, 2];
  for (const i of channels) components.push(['s' + i + '/s0', 's' + i + 'n']);
  for (const [name, suffix] of components) {
    previews.push([name, base + '-' + suffix + '-thumb.webp', base + '-' + suffix + '.webp']);
  }
  const strip = byId('previews');
  const flat = byId('flat');
  const panorama = byId('panorama');
  const status = byId('panoramaStatus');
  const modeButtons = byId('viewMode').querySelectorAll('button');
  function setPanorama(enabled) {
    flat.hidden = enabled;
    panorama.hidden = !enabled;
    status.hidden = true;
    modeButtons[0].setAttribute('aria-pressed', String(!enabled));
    modeButtons[1].setAttribute('aria-pressed', String(enabled));
    if (enabled) window.scenePanorama.show(panorama, flat.src, flat.alt);
    else window.scenePanorama.hide();
  }
  modeButtons[0].onclick = () => setPanorama(false);
  modeButtons[1].onclick = () => setPanorama(true);
  panorama.addEventListener('panoramaerror', () => {
    setPanorama(false);
    status.textContent = 'Panorama unavailable. Showing the 2D preview.';
    status.hidden = false;
  });
  for (const [name, thumbnail, source] of previews) {
    const button = document.createElement('button');
    button.type = 'button';
    button.setAttribute('aria-pressed', 'false');
    const image = new Image();
    const isModulated = name === 'AoLP × DoLP';
    function previewUrl(path) {
      if (name === 's0') return path;
      return isModulated ? polarizationImage(path) : luminancePreview(path);
    }
    image.src = previewUrl(thumbnail);
    image.alt = '';
    image.width = 128;
    image.height = 64;
    const label = document.createElement('span');
    label.textContent = name;
    button.append(image, label);
    button.onclick = () => {
      for (const item of strip.children) item.setAttribute('aria-pressed', String(item === button));
      showLegend(name);
      flat.src = previewUrl(source);
      flat.alt = scene.title + ' · ' + name;
      status.hidden = true;
      if (!panorama.hidden) window.scenePanorama.show(panorama, flat.src, flat.alt);
    };
    if (isModulated) {
      observeTheme(() => {
        image.src = polarizationImage(thumbnail);
        if (button.getAttribute('aria-pressed') === 'true') button.click();
      });
    }
    strip.append(button);
    if (name === 's0') button.click();
  }
}
function renderScene(s) {
  document.title = s.title + ' · ' + s.id + ' · penvmap';
  byId('sceneTitle').textContent = s.title;
  function filterLink(label, filters) {
    const link = document.createElement('a');
    link.href = './?' + new URLSearchParams(filters);
    link.textContent = label;
    return link;
  }
  const fields = {
    Resolution: s.width + ' × ' + s.height,
    Date: s.captured,
    Environment: filterLink(s.setting, { setting: s.setting }),
    Tags: (s.tags.length ? s.tags : ['Untagged']).map(tag => filterLink(tag, { setting: s.setting, tag })),
    'Polarization': filterLink(s.collection, { collection: s.collection })
  };
  if (s.setting !== 'Indoor' && s.weather) {
    fields.Weather = filterLink(s.weather, { weather: s.weather.toLowerCase() });
  }
  if (s.time) fields['Time of day'] = filterLink(s.time, { time: s.time.toLowerCase() });
  byId('metadata').replaceChildren(...Object.entries(fields).map(([key,val]) => {
    const pair = document.createElement('div');
    const term = document.createElement('dt');
    term.textContent = key;
    const description = document.createElement('dd');
    for (const item of Array.isArray(val) ? val : [val]) {
      if (description.childNodes.length) description.append(', ');
      description.append(item);
    }
    pair.append(term, description);
    return pair;
  }));
  const exr = s.dataset + '/resolve/' + s.revision + '/' + s.file;
  const renderer = new URL('https://elerac.github.io/plenoview/app/');
  const duffFile = s.file.replace('/thetaphi/', '/duff/');
  const viewerSource = s.dataset + '/resolve/' + s.revision + '/' + duffFile;
  renderer.search = new URLSearchParams({ src: viewerSource, view: 'panorama', name: s.title });
  byId('renderer').href = renderer.href;
  const basis = byId('downloadBasis');
  function updateDownload() {
    const duff = basis.value === 'duff';
    const bytes = duff ? s.duffBytes : s.bytes;
    byId('download').href = (duff ? viewerSource : exr) + '?download=true';
    byId('downloadSize').textContent = (bytes / 1e6).toFixed(1) + ' MB';
    byId('source').href = s.dataset + '/blob/' + s.revision + '/' + (duff ? duffFile : s.file);
  }
  basis.onchange = updateDownload;
  updateDownload();
  byId('sceneContent').hidden = false;
  renderPreviews(s);
}
if (active) {
  renderScene(active);
  const related = (window.similarScenes?.[id] || [])
    .map(sceneId => window.atlasScenes.find(scene => scene.id === sceneId))
    .filter(Boolean);
  if (related.length) {
    renderCards(byId('similarMaps').querySelector('.similar-grid'), related);
    byId('similarMaps').hidden = false;
  }
  const preview = byId('sceneContent');
  const fullscreen = byId('fullscreenToggle');
  function setExpanded(expanded) {
    preview.classList.toggle('expanded', expanded);
    const label = expanded ? 'Exit fullscreen' : 'Enter fullscreen';
    fullscreen.setAttribute('aria-label', label);
    fullscreen.title = label;
    fullscreen.querySelector('path').setAttribute('d', expanded
      ? 'M3 9h6V3m6 0v6h6m0 6h-6v6m-6 0v-6H3'
      : 'M3 9V3h6m6 0h6v6m0 6v6h-6m-6 0H3v-6');
  }
  fullscreen.onclick = async () => {
    if (document.fullscreenElement === preview) {
      await document.exitFullscreen();
    } else if (preview.classList.contains('expanded')) {
      setExpanded(false);
    } else {
      setExpanded(true);
      if (document.fullscreenEnabled) {
        try { await preview.requestFullscreen(); }
        catch { /* Keep the viewport-filling layout when native fullscreen is unavailable. */ }
      }
    }
  };
  document.addEventListener('fullscreenchange', () => setExpanded(document.fullscreenElement === preview));
  document.addEventListener('keydown', event => {
    if (event.key === 'Escape' && !document.fullscreenElement && preview.classList.contains('expanded')) {
      setExpanded(false);
      fullscreen.focus();
    }
  });
  const toggle = byId('detailsToggle');
  toggle.hidden = false;
  toggle.onclick = () => toggle.setAttribute('aria-expanded', String(toggle.getAttribute('aria-expanded') !== 'true'));
  document.addEventListener('keydown', event => {
    if (event.key === 'Escape' && toggle.offsetWidth && toggle.getAttribute('aria-expanded') === 'true') {
      toggle.setAttribute('aria-expanded', 'false');
      toggle.focus();
    }
  });
} else {
  document.title = 'Scene not found · penvmap';
  byId('sceneTitle').textContent = 'Scene not found';
}
