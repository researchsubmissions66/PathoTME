/* Dependency-free navigation and accessible, keyboard-operable explorers. */
(() => {
  'use strict';
  document.documentElement.classList.add('js');
  const content = window.PATHOTME_CONTENT;
  const setText = (id, value) => {
    const element = document.getElementById(id);
    if (element) element.textContent = value;
  };

  function bindTabs(selector, update) {
    const tabs = Array.from(document.querySelectorAll(selector));
    if (!tabs.length) return;
    function select(tab, focus = false) {
      tabs.forEach(item => {
        item.setAttribute('aria-selected', String(item === tab));
        item.tabIndex = item === tab ? 0 : -1;
      });
      const panel = document.getElementById(tab.getAttribute('aria-controls'));
      panel.setAttribute('aria-labelledby', tab.id);
      update(tab);
      if (focus) tab.focus();
    }
    tabs.forEach((tab, index) => {
      tab.addEventListener('click', () => select(tab));
      tab.addEventListener('keydown', event => {
        let next;
        if (event.key === 'ArrowRight') next = (index + 1) % tabs.length;
        else if (event.key === 'ArrowLeft') next = (index + tabs.length - 1) % tabs.length;
        else if (event.key === 'Home') next = 0;
        else if (event.key === 'End') next = tabs.length - 1;
        else return;
        event.preventDefault();
        select(tabs[next], true);
      });
    });
  }

  if (content) {
    bindTabs('[data-model]', tab => {
      const model = content.architectures[tab.dataset.model];
      setText('model-scale', model.scale);
      setText('model-title', model.title);
      setText('query-label', model.query);
      setText('aggregation-label', model.aggregation);
      setText('model-note', model.note);
    });
    bindTabs('[data-cohort-panel]', tab => {
      const panel = content.panels[tab.dataset.cohortPanel];
      setText('panel-id', panel.id);
      setText('panel-name', panel.name);
      setText('feature-number', panel.count);
      setText('panel-token-note', panel.tokens);
      setText('panel-detail', panel.detail);
      const families = document.getElementById('feature-families');
      families.replaceChildren(...panel.families.map(family => {
        const row = document.createElement('div');
        row.className = 'feature-family';
        const line = document.createElement('span');
        line.className = `family-line ${family.color}`;
        const copy = document.createElement('div');
        const name = document.createElement('strong');
        name.textContent = family.name;
        const detail = document.createElement('span');
        detail.textContent = family.detail;
        copy.append(name, detail);
        const count = document.createElement('b');
        count.textContent = family.count;
        row.append(line, copy, count);
        return row;
      }));
    });
  }

  const toggle = document.querySelector('.menu-toggle');
  const links = document.getElementById('nav-links');
  const closeMenu = () => {
    if (!toggle || !links) return;
    links.classList.remove('open');
    toggle.setAttribute('aria-expanded', 'false');
    toggle.setAttribute('aria-label', 'Open navigation');
  };
  if (toggle && links) {
    toggle.addEventListener('click', () => {
      const opened = links.classList.toggle('open');
      toggle.setAttribute('aria-expanded', String(opened));
      toggle.setAttribute('aria-label', opened ? 'Close navigation' : 'Open navigation');
    });
    links.querySelectorAll('a').forEach(link => link.addEventListener('click', closeMenu));
    document.addEventListener('keydown', event => {
      if (event.key === 'Escape' && links.classList.contains('open')) {
        closeMenu();
        toggle.focus();
      }
    });
    document.addEventListener('click', event => {
      if (!event.target.closest('.navigation')) closeMenu();
    });
    window.addEventListener('resize', () => {
      if (window.innerWidth > 600) closeMenu();
    }, {passive: true});
  }

  const progress = document.getElementById('scroll-progress');
  const navItems = Array.from(document.querySelectorAll('.nav-links a[href^="#"]'));
  let scheduled = false;
  function updateScroll() {
    const extent = document.documentElement.scrollHeight - window.innerHeight;
    const fraction = extent > 0 ? Math.min(1, Math.max(0, window.scrollY / extent)) : 0;
    if (progress) progress.style.transform = `scaleX(${fraction})`;
    let current = null;
    navItems.forEach(link => {
      const section = document.getElementById(link.hash.slice(1));
      if (section && section.getBoundingClientRect().top <= 150) current = link;
    });
    navItems.forEach(link => {
      link.classList.toggle('active', link === current);
      if (link === current) link.setAttribute('aria-current', 'location');
      else link.removeAttribute('aria-current');
    });
    scheduled = false;
  }
  window.addEventListener('scroll', () => {
    if (!scheduled) {
      scheduled = true;
      window.requestAnimationFrame(updateScroll);
    }
  }, {passive: true});
  window.addEventListener('resize', updateScroll, {passive: true});
  updateScroll();
})();
