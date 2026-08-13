(() => {
  "use strict";

  const currentPage = document.body.dataset.page || "index.html";
  const progress = document.querySelector("#reading-progress");
  const nav = document.querySelector("#course-nav");
  const navToggle = document.querySelector("[data-nav-toggle]");
  const progressText = document.querySelector("[data-course-progress]");
  const progressBar = document.querySelector("[data-course-progress-bar]");
  const storageKey = "lora-studio-mastery-visited";

  const updateReadingProgress = () => {
    const available = document.documentElement.scrollHeight - window.innerHeight;
    const percent = available > 0 ? Math.min(100, (window.scrollY / available) * 100) : 100;
    if (progress) progress.style.width = `${percent}%`;
  };
  addEventListener("scroll", updateReadingProgress, { passive: true });
  addEventListener("resize", updateReadingProgress, { passive: true });
  updateReadingProgress();

  const chapterPages = [...document.querySelectorAll(".course-nav nav a")]
    .map(link => link.getAttribute("href"))
    .filter(Boolean);
  let visited = [];
  try {
    visited = JSON.parse(localStorage.getItem(storageKey) || "[]");
    if (!Array.isArray(visited)) visited = [];
  } catch {
    visited = [];
  }
  if (chapterPages.includes(currentPage) && !visited.includes(currentPage)) {
    visited.push(currentPage);
    try { localStorage.setItem(storageKey, JSON.stringify(visited)); } catch { /* private mode */ }
  }
  const completed = chapterPages.filter(page => visited.includes(page)).length;
  const coursePercent = chapterPages.length ? Math.round((completed / chapterPages.length) * 100) : 0;
  if (progressText) progressText.textContent = `${coursePercent}%`;
  if (progressBar) progressBar.style.width = `${coursePercent}%`;

  navToggle?.addEventListener("click", () => {
    const open = nav?.classList.toggle("open") || false;
    navToggle.setAttribute("aria-expanded", String(open));
  });
  nav?.querySelectorAll("a").forEach(link => link.addEventListener("click", () => {
    nav.classList.remove("open");
    navToggle?.setAttribute("aria-expanded", "false");
  }));

  const dialog = document.querySelector("[data-search-dialog]");
  const input = document.querySelector("[data-search-input]");
  const results = document.querySelector("[data-search-results]");
  const searchRecords = Array.isArray(window.HANDBOOK_SEARCH) ? window.HANDBOOK_SEARCH : [];

  const escapeHtml = value => value.replace(/[&<>'"]/g, character => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;"
  })[character]);

  const renderResults = query => {
    if (!results) return;
    const terms = query.toLowerCase().trim().split(/\s+/).filter(Boolean);
    if (!terms.length) {
      results.innerHTML = "";
      return;
    }
    const matches = searchRecords
      .map(record => {
        const haystack = `${record.title} ${record.headings} ${record.text}`.toLowerCase();
        const score = terms.reduce((total, term) => total + (haystack.includes(term) ? 1 : 0), 0);
        return { record, score };
      })
      .filter(item => item.score === terms.length)
      .slice(0, 12);
    results.innerHTML = matches.length
      ? matches.map(({ record }) => `<li><a href="${encodeURI(record.url)}"><strong>${escapeHtml(record.title)}</strong><span>${escapeHtml(record.summary)}</span></a></li>`).join("")
      : "<li><span>No matching chapter. Try a broader term.</span></li>";
  };

  document.querySelectorAll("[data-search-open]").forEach(button => button.addEventListener("click", () => {
    dialog?.showModal();
    setTimeout(() => input?.focus(), 0);
  }));
  input?.addEventListener("input", event => renderResults(event.target.value));
  addEventListener("keydown", event => {
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "k") {
      event.preventDefault();
      dialog?.showModal();
      setTimeout(() => input?.focus(), 0);
    }
  });
})();
