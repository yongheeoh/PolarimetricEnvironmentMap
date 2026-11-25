// Load and play each explanation video when it enters the viewport.
(() => {
  const videos = document.querySelectorAll('.stitching-video');
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  function load(video) {
    if (video.dataset.src) {
      video.src = video.dataset.src;
      delete video.dataset.src;
    }
  }

  if (!('IntersectionObserver' in window)) {
    videos.forEach(load);
    return;
  }

  const observer = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      const video = entry.target;
      if (entry.isIntersecting) {
        load(video);
        if (!reducedMotion) video.play().catch(() => {});
      } else {
        video.pause();
      }
    });
  }, { threshold: 0.15 });

  videos.forEach(video => {
    if (reducedMotion) video.autoplay = false;
    observer.observe(video);
  });
})();
