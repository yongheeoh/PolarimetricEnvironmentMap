'use strict';
const byId = id => document.getElementById(id);
const collectionOrder = ['Linear Stokes', 'Full Stokes'];
const scenes = [...window.atlasScenes].sort((a, b) =>
  collectionOrder.indexOf(a.collection) - collectionOrder.indexOf(b.collection) || a.number - b.number
);
const params = new URLSearchParams(location.search);
const filters = {
  weather: params.get('weather') || '',
  time: params.get('time') || '',
  setting: params.get('setting') || '',
  collection: params.get('collection') || '',
  tag: params.get('tag') || ''
};
const untagged = 'Untagged';
function hasTag(scene, tag) {
  return !tag || (tag === untagged ? scene.tags.length === 0 : scene.tags.includes(tag));
}
function choice(label, count, selected, select) {
  const button = element('button', label);
  button.append(element('span', String(count)));
  button.setAttribute('aria-pressed', String(selected));
  button.onclick = () => {
    select();
    filter();
    byId(button.dataset.group).querySelector('[aria-pressed="true"]').focus();
  };
  return button;
}
function renderChoices(container, rows, key, choices, label, onSelect, matches) {
  container.replaceChildren(...choices.map(value => {
    const count = rows.filter(scene => matches(scene, value)).length;
    const button = choice(label(value), count, filters[key] === value, () => onSelect(value));
    button.dataset.group = container.id;
    return button;
  }));
}
function filter() {
  if (filters.weather.toLowerCase() === 'indoor') {
    filters.setting ||= 'Indoor';
    filters.weather = '';
  }
  if (filters.setting === 'Indoor') filters.weather = '';
  byId('weatherGroup').hidden = filters.setting === 'Indoor';
  const url = new URL(location.href);
  for (const [key, value] of Object.entries(filters)) {
    if (value) url.searchParams.set(key, value);
    else url.searchParams.delete(key);
  }
  history.replaceState(history.state, '', url);
  const {setting} = filters;
  function matchingRows(...ignored) {
    return scenes.filter(scene => Object.entries(filters).every(([key, value]) => {
      if (!value || ignored.includes(key)) return true;
      if (key === 'tag') return hasTag(scene, value);
      return scene[key].toLowerCase() === value.toLowerCase();
    }));
  }
  const collectionRows = matchingRows('collection');
  const visible = matchingRows();
  const settings = byId('settings');
  renderChoices(settings, matchingRows('setting', 'tag', 'weather'), 'setting', ['', 'Outdoor', 'Indoor'],
    v => v || 'All environments', v => { filters.setting = v; filters.tag = ''; },
    (s, v) => (!v || s.setting === v) && (v === 'Indoor' || !filters.weather || s.weather.toLowerCase() === filters.weather));
  if (setting) {
    const rows = scenes.filter(s => s.setting === setting);
    const tags = [...new Set(rows.flatMap(s => s.tags))].sort();
    if (rows.some(s => !s.tags.length)) tags.push(untagged);
    const group = element('div');
    group.id = 'subcategories';
    group.setAttribute('role', 'group');
    group.setAttribute('aria-label', setting + ' scene content');
    settings.querySelector('[aria-pressed="true"]').after(group);
    renderChoices(group, matchingRows('tag'),
      'tag', ['', ...tags], v => v ? v[0].toUpperCase() + v.slice(1) : 'All content',
      v => { filters.tag = v; }, hasTag);
  }
  renderChoices(byId('collections'), collectionRows, 'collection',
    ['', ...collectionOrder], v => v || 'All',
    v => { filters.collection = v; }, (s, v) => !v || s.collection === v);
  for (const key of ['weather', 'time']) {
    const rows = key === 'weather' ? scenes.filter(s => s.setting === 'Outdoor') : scenes;
    const values = [...new Set(rows.map(s => s[key].toLowerCase()).filter(Boolean))].sort();
    renderChoices(byId(key), matchingRows(key), key, ['', ...values],
      v => v ? v[0].toUpperCase() + v.slice(1) : (key === 'weather' ? 'All weather' : 'All times'),
      v => { filters[key] = v; }, (s, v) => !v || s[key].toLowerCase() === v);
  }
  renderCards(byId('cards'), visible);
  byId('count').textContent = visible.length + (visible.length === 1 ? ' scene' : ' scenes');
  byId('empty').hidden = !!visible.length;
}
byId('reset').onclick = () => { for (const key of Object.keys(filters)) filters[key] = ''; filter(); byId('settings').firstElementChild.focus(); };

filter();
