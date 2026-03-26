// AI Campus — main.js
document.addEventListener('DOMContentLoaded', () => {
  // Auto-dismiss flash messages after 4s
  document.querySelectorAll('.flash').forEach(el => {
    setTimeout(() => {
      el.style.transition = 'opacity .5s';
      el.style.opacity = '0';
      setTimeout(() => el.remove(), 500);
    }, 4000);
  });
  // Stagger animate-in elements
  document.querySelectorAll('.animate-in').forEach((el, i) => {
    el.style.animationDelay = (i * 0.07) + 's';
  });
});
