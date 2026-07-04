document.addEventListener('DOMContentLoaded', function () {
  const formValidation = Array.from(document.querySelectorAll('.needs-validation'));
  formValidation.forEach(function (form) {
    form.addEventListener('submit', function (event) {
      if (!form.checkValidity()) {
        event.preventDefault();
        event.stopPropagation();
      }
      form.classList.add('was-validated');
    }, false);
  });

  const historySearch = document.getElementById('historySearch');
  const historyGrid = document.getElementById('historyGrid');
  const filterApproved = document.getElementById('filterApproved');
  const filterRejected = document.getElementById('filterRejected');
  const filterClear = document.getElementById('filterClear');

  if (historySearch && historyGrid) {
    const cards = Array.from(historyGrid.querySelectorAll('.history-card'));

    const applyFilter = () => {
      const query = historySearch.value.trim().toLowerCase();
      cards.forEach(card => {
        const status = card.dataset.status;
        const applicant = card.dataset.applicant;
        const area = card.dataset.area;
        const matchesQuery = !query || applicant.includes(query) || area.includes(query) || status.includes(query);
        card.style.display = matchesQuery ? 'block' : 'none';
      });
    };

    historySearch.addEventListener('input', applyFilter);

    if (filterApproved) {
      filterApproved.addEventListener('click', () => {
        cards.forEach(card => {
          card.style.display = card.dataset.status === 'approved' ? 'block' : 'none';
        });
      });
    }

    if (filterRejected) {
      filterRejected.addEventListener('click', () => {
        cards.forEach(card => {
          card.style.display = card.dataset.status === 'rejected' ? 'block' : 'none';
        });
      });
    }

    if (filterClear) {
      filterClear.addEventListener('click', () => {
        historySearch.value = '';
        cards.forEach(card => {
          card.style.display = 'block';
        });
      });
    }
  }

  const smoothLinks = document.querySelectorAll('a[href^="#"]');
  smoothLinks.forEach(link => {
    link.addEventListener('click', function (event) {
      const target = document.querySelector(this.getAttribute('href'));
      if (target) {
        event.preventDefault();
        target.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    });
  });
});

// Fancy UI interactions: navbar glass on scroll, counters, floating cards
document.addEventListener('scroll', function () {
  const header = document.getElementById('siteHeader');
  if (!header) return;
  if (window.scrollY > 24) header.classList.add('glass'); else header.classList.remove('glass');
});

// Counter animation for elements with data-counter
function animateCounters() {
  const els = document.querySelectorAll('[data-counter]');
  els.forEach(el => {
    const target = parseFloat(el.dataset.counter);
    const duration = 1200;
    let start = null;
    function step(ts) {
      if (!start) start = ts;
      const progress = Math.min((ts - start) / duration, 1);
      el.innerText = Math.round(progress * target);
      if (progress < 1) window.requestAnimationFrame(step);
    }
    window.requestAnimationFrame(step);
  });
}

window.addEventListener('load', function () { setTimeout(animateCounters, 400); });
