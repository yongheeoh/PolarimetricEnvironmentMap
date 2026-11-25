'use strict';

window.scenePanorama = (() => {
  let active;
  let camera = { pitch: 0, yaw: 0, hfov: 90 };

  function lazyAsset(tag, url) {
    let pending;
    return () => pending ??= new Promise((resolve, reject) => {
      const asset = document.createElement(tag);
      asset.onload = resolve;
      asset.onerror = () => {
        pending = null;
        asset.remove();
        reject(new Error('Could not load ' + url));
      };
      if (tag === 'link') {
        asset.rel = 'stylesheet';
        asset.href = url;
      } else {
        asset.src = url;
      }
      document.head.append(asset);
    });
  }

  const loadStyle = lazyAsset('link', 'vendor/pannellum/pannellum.css');
  const loadScript = lazyAsset('script', 'vendor/pannellum/pannellum.js');

  function release(session) {
    session.viewer?.destroy();
    session.mount.remove();
  }

  function hide() {
    if (!active) return;
    const session = active;
    active = null;
    if (session.viewer) {
      camera = {
        pitch: session.viewer.getPitch(),
        yaw: session.viewer.getYaw(),
        hfov: session.viewer.getHfov()
      };
    }
    session.container.removeAttribute('aria-busy');
    session.resizeObserver?.disconnect();
    session.mount.remove();
    // Pannellum 2.5 still completes in-flight image loads after destroy().
    // Let that load settle on its detached mount before releasing its renderer.
    if (!session.loading) release(session);
  }

  function settled(session, failed) {
    session.loading = false;
    if (session !== active) {
      release(session);
      return;
    }
    if (session.current !== session.requested) {
      loadRequested(session);
    } else if (failed) {
      hide();
      session.container.dispatchEvent(new Event('panoramaerror'));
    } else {
      session.container.removeAttribute('aria-busy');
    }
  }

  function loadRequested(session) {
    session.current = session.requested;
    session.loading = true;
    session.container.setAttribute('aria-busy', 'true');
    const scene = { type: 'equirectangular', panorama: session.current };
    try {
      if (session.viewer) {
        session.viewer.addScene(session.current, scene);
        session.viewer.loadScene(session.current, 'same', 'same', 'same');
      } else {
        session.viewer = window.pannellum.viewer(session.mount, {
          default: { firstScene: session.current, ...camera, autoLoad: true, compass: false, showFullscreenCtrl: false },
          scenes: { [session.current]: scene }
        });
        session.resizeObserver = new ResizeObserver(() => session.viewer.resize());
        session.resizeObserver.observe(session.mount);
        // Let Pannellum finish releasing its previous texture before starting a queued scene.
        session.viewer.on('load', () => queueMicrotask(() => settled(session, false)));
        session.viewer.on('error', () => queueMicrotask(() => settled(session, true)));
      }
    } catch {
      queueMicrotask(() => settled(session, true));
    }
  }

  function show(container, src, label) {
    if (!active || active.container !== container) {
      hide();
      const mount = document.createElement('div');
      mount.style.cssText = 'width:100%;height:100%';
      container.append(mount);
      const session = active = { container, mount, loading: true };
      container.setAttribute('aria-busy', 'true');
      Promise.all([loadStyle(), loadScript()]).then(() => {
        if (active === session) loadRequested(session);
      }, () => {
        if (active === session) {
          hide();
          container.dispatchEvent(new Event('panoramaerror'));
        }
      });
    }
    active.mount.setAttribute('aria-label', label);
    active.requested = src;
    if (!active.loading && active.current !== src) loadRequested(active);
  }

  return { show, hide };
})();
