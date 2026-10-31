// One side-by-side video keeps both lighting conditions on the same decoded frame.
function initializeRenderingComparison() {
  var viewer = document.querySelector('.render-viewer');
  if (!viewer) { return; }
  var comparison = viewer.querySelector('.render-comparison');
  var handle = viewer.querySelector('.render-handle');
  var play = viewer.querySelector('#render-play');
  var viewNumber = viewer.querySelector('#render-view-number');
  var video = viewer.querySelector('.render-motion');
  var canvas = viewer.querySelector('.render-canvas');
  var context = canvas.getContext('2d', { alpha: false });
  var images = [viewer.querySelector('.render-penvmap'), viewer.querySelector('.render-envmap')];
  var buttons = Array.from(viewer.querySelectorAll('[data-component]'));
  var error = viewer.querySelector('.render-error');
  var component = 's0';
  var frame = 0;
  var desiredFrame = 0;
  var frameCount = 360;
  var split = 50;
  var drag = null;
  var playing = false;
  var ready = false;
  var failed = false;
  var fallbackVersion = 0;
  var seekScheduled = false;
  var lastSeekFrame = -1;
  video.playbackRate = 0.8;

  function paint() {
    if (!ready || video.readyState < 2 || video.seeking) { return; }
    var w = canvas.width, h = canvas.height, x = Math.round(w * split / 100);
    if (x > 0) { context.drawImage(video, 0, 0, x, h, 0, 0, x, h); }
    if (x < w) { context.drawImage(video, w + x, 0, w - x, h, x, 0, w - x, h); }
    frame = Math.min(frameCount - 1, Math.floor(video.currentTime * 30 + 0.001));
    if (playing) { desiredFrame = frame; }
    canvas.hidden = false;
    comparison.dataset.frame = String(frame);
    comparison.dataset.component = component;
    comparison.setAttribute('aria-busy', 'false');
    viewNumber.textContent = (frame + 1) + ' / ' + frameCount;
  }
  function setSplit(value) {
    split = Math.max(0, Math.min(100, value));
    comparison.style.setProperty('--split', split + '%');
    handle.setAttribute('aria-valuenow', split.toFixed(1));
    handle.setAttribute('aria-valuetext', split.toFixed(1) + '% pEnvmap, ' +
      (100 - split).toFixed(1) + '% Envmap');
    paint();
  }
  function setPlaying(value) {
    playing = value;
    play.setAttribute('aria-pressed', String(value));
    play.setAttribute('aria-label', value ? 'Pause rotation' : 'Auto rotate');
    play.setAttribute('title', value ? 'Pause rotation' : 'Auto rotate');
    play.textContent = value ? '\u23F8' : '\u25B6';
    if (!value) { video.pause(); }
    else if (ready) {
      video.play().catch(function() { setPlaying(false); });
    }
  }
  async function fallback() {
    var version = ++fallbackVersion;
    var targetComponent = component, targetFrame = desiredFrame;
    try {
      var pair = await Promise.all(['penvmap', 'envmap'].map(function(side) {
        return new Promise(function(resolve, reject) {
          var image = new Image();
          image.onload = function() { resolve(image); };
          image.onerror = reject;
          image.src = './static/images/render-comparison/' + targetComponent + '/' +
            String(targetFrame).padStart(3, '0') + '-' + side + '.jpg';
        });
      }));
      if (version !== fallbackVersion || component !== targetComponent || (ready && !canvas.hidden)) { return; }
      images[0].src = pair[0].src; images[1].src = pair[1].src;
      canvas.hidden = true;
      frame = targetFrame;
      comparison.dataset.frame = String(frame);
      comparison.dataset.component = component;
      viewNumber.textContent = (frame + 1) + ' / ' + frameCount;
      error.hidden = true;
    } catch (loadError) { if (version === fallbackVersion) { error.hidden = false; } }
    if (version === fallbackVersion) { comparison.setAttribute('aria-busy', 'false'); }
  }
  function seek() {
    seekScheduled = false;
    if (failed) { fallback(); return; }
    if (!ready || video.seeking || playing) { return; }
    if (frame === desiredFrame && !canvas.hidden) { paint(); return; }
    comparison.setAttribute('aria-busy', 'true');
    lastSeekFrame = desiredFrame;
    video.currentTime = (desiredFrame + 0.1) / 30;
  }
  function requestFrame(value) {
    desiredFrame = ((value % frameCount) + frameCount) % frameCount;
    if (!seekScheduled) {
      seekScheduled = true;
      requestAnimationFrame(seek);
    }
  }
  function selectComponent(value) {
    component = value;
    ready = false; failed = false;
    play.disabled = false;
    ++fallbackVersion;
    canvas.hidden = true;
    error.hidden = true;
    comparison.setAttribute('aria-busy', 'true');
    video.pause();
    buttons.forEach(function(button) {
      button.setAttribute('aria-pressed', String(button.dataset.component === component));
    });
    fallback();
    video.src = './static/videos/render-comparison/' + component + '.mp4';
    video.load();
  }
  video.addEventListener('loadeddata', function() {
    ready = true;
    canvas.width = video.videoWidth / 2;
    canvas.height = video.videoHeight;
    video.playbackRate = 0.8;
    lastSeekFrame = desiredFrame;
    video.currentTime = (desiredFrame + 0.1) / 30;
  });
  video.addEventListener('seeked', function() {
    paint();
    if (desiredFrame !== lastSeekFrame && !playing) { seek(); }
    else if (playing) { video.play().catch(function() { setPlaying(false); }); }
  });
  video.addEventListener('error', function() {
    ready = false; failed = true; setPlaying(false); play.disabled = true; fallback();
  });
  video.addEventListener('waiting', function() { comparison.setAttribute('aria-busy', 'true'); });
  if ('requestVideoFrameCallback' in video) {
    var nextVideoFrame = function() { paint(); video.requestVideoFrameCallback(nextVideoFrame); };
    video.requestVideoFrameCallback(nextVideoFrame);
  } else {
    var nextAnimationFrame = function() {
      if (playing) { paint(); }
      requestAnimationFrame(nextAnimationFrame);
    };
    requestAnimationFrame(nextAnimationFrame);
  }
  comparison.addEventListener('pointerdown', function(event) {
    if (event.button !== 0 || drag) { return; }
    setPlaying(false);
    var isDivider = event.target === handle;
    drag = { id: event.pointerId, x: event.clientX, frame: desiredFrame,
      divider: isDivider, width: comparison.getBoundingClientRect().width };
    comparison.setPointerCapture(event.pointerId);
    comparison.classList.add('is-dragging');
    if (isDivider) { event.preventDefault(); handle.focus(); }
  });
  comparison.addEventListener('pointermove', function(event) {
    if (!drag || drag.id !== event.pointerId) { return; }
    if (drag.divider) {
      var bounds = comparison.getBoundingClientRect();
      setSplit((event.clientX - bounds.left) / bounds.width * 100);
    } else {
      // A full-width drag rotates only 120 views, one third of the former sensitivity.
      requestFrame(drag.frame + Math.round((event.clientX - drag.x) / drag.width * 120));
    }
  });
  function endDrag(event) {
    if (!drag || drag.id !== event.pointerId) { return; }
    drag = null;
    comparison.classList.remove('is-dragging');
    seek();
  }
  comparison.addEventListener('pointerup', endDrag);
  comparison.addEventListener('pointercancel', endDrag);
  comparison.addEventListener('lostpointercapture', endDrag);
  handle.addEventListener('keydown', function(event) {
    var values = { ArrowLeft: split - 1, ArrowRight: split + 1, Home: 0, End: 100 };
    if (!(event.key in values)) { return; }
    event.preventDefault(); event.stopPropagation(); setSplit(values[event.key]);
  });
  comparison.addEventListener('keydown', function(event) {
    if (event.key !== 'ArrowLeft' && event.key !== 'ArrowRight') { return; }
    event.preventDefault(); setPlaying(false);
    requestFrame(desiredFrame + (event.key === 'ArrowRight' ? 1 : -1));
  });
  viewer.querySelector('#render-previous').addEventListener('click', function() {
    setPlaying(false); requestFrame(desiredFrame - 1);
  });
  viewer.querySelector('#render-next').addEventListener('click', function() {
    setPlaying(false); requestFrame(desiredFrame + 1);
  });
  play.addEventListener('click', function() { setPlaying(!playing); });
  document.addEventListener('visibilitychange', function() {
    if (document.hidden) { setPlaying(false); }
  });
  buttons.forEach(function(button) {
    button.addEventListener('click', function() { selectComponent(button.dataset.component); });
  });
  // Fetch only the selected component, when the viewer approaches the viewport.
  if ('IntersectionObserver' in window) {
    var observer = new IntersectionObserver(function(entries) {
      if (entries.some(function(entry) { return entry.isIntersecting; })) {
        observer.disconnect(); selectComponent(component);
      }
    }, { rootMargin: '200px' });
    observer.observe(viewer);
  } else { selectComponent(component); }
}
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initializeRenderingComparison);
} else { initializeRenderingComparison(); }
