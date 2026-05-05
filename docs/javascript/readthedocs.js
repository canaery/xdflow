document.addEventListener("DOMContentLoaded", function () {
  const searchInput = document.querySelector(".md-search__input");
  if (searchInput) {
    searchInput.addEventListener("focus", function () {
      document.dispatchEvent(new CustomEvent("readthedocs-search-show"));
    });
  }
});

document.addEventListener("readthedocs-addons-data-ready", function (event) {
  const config = event.detail.data();
  const activeVersions = (config.versions && config.versions.active) || [];
  const currentSlug = config.versions && config.versions.current ? config.versions.current.slug : "latest";

  const markup = `
    <div class="md-version">
      <button class="md-version__current" aria-label="Select version">
        ${currentSlug}
      </button>
      <ul class="md-version__list">
        ${activeVersions
          .map(
            (version) => `
              <li class="md-version__item">
                <a href="${version.urls.documentation}" class="md-version__link">${version.slug}</a>
              </li>`
          )
          .join("\n")}
      </ul>
    </div>`;

  const existing = document.querySelector(".md-version");
  if (existing) {
    existing.remove();
  }

  const target = document.querySelector(".md-header__topic");
  if (target) {
    target.insertAdjacentHTML("beforeend", markup);
  }
});
