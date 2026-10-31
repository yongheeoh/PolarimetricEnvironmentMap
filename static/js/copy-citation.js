function initializeCitationCopy() {
  var button = document.getElementById('copy-citation');
  var citation = document.getElementById('citation-text');
  if (!button || !citation) { return; }
  var resetTimer;

  function copyFallback(text) {
    var field = document.createElement('textarea');
    field.value = text;
    field.setAttribute('readonly', '');
    field.style.position = 'fixed';
    field.style.opacity = '0';
    document.body.appendChild(field);
    field.select();
    var copied = false;
    try { copied = document.execCommand('copy'); } catch (error) { /* Show failure below. */ }
    field.remove();
    button.focus({ preventScroll: true });
    return copied;
  }

  button.addEventListener('click', async function() {
    clearTimeout(resetTimer);
    button.disabled = true;
    var text = citation.textContent;
    var copied = false;
    try {
      if (navigator.clipboard && navigator.clipboard.writeText) {
        await navigator.clipboard.writeText(text);
        copied = true;
      } else {
        copied = copyFallback(text);
      }
    } catch (error) {
      copied = copyFallback(text);
    }
    button.disabled = false;
    button.textContent = copied ? 'Copied!' : 'Copy failed';
    resetTimer = setTimeout(function() { button.textContent = 'Copy'; }, 2000);
  });
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initializeCitationCopy);
} else {
  initializeCitationCopy();
}
