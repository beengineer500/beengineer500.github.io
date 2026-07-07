(function () {
  "use strict";

  var root = document.documentElement;
  var mq = window.matchMedia("(prefers-color-scheme: dark)");

  function currentTheme() {
    return root.getAttribute("data-theme") || (mq.matches ? "dark" : "light");
  }

  function syncToggle(btn, theme) {
    var isDark = theme === "dark";
    btn.setAttribute("aria-pressed", String(isDark));
    btn.setAttribute("aria-label", isDark ? "라이트 모드로 전환" : "다크 모드로 전환");
    btn.querySelector(".theme-toggle-icon").textContent = isDark ? "☾" : "☀";
    btn.querySelector(".theme-toggle-label").textContent = isDark ? "다크" : "라이트";
  }

  var toggle = document.getElementById("theme-toggle");
  if (toggle) {
    syncToggle(toggle, currentTheme());
    toggle.addEventListener("click", function () {
      var next = currentTheme() === "dark" ? "light" : "dark";
      root.setAttribute("data-theme", next);
      try {
        localStorage.setItem("theme", next);
      } catch (e) {}
      syncToggle(toggle, next);
    });
  }

  document.querySelectorAll(".article pre").forEach(function (pre) {
    var btn = document.createElement("button");
    btn.type = "button";
    btn.className = "code-copy";
    btn.textContent = "복사";
    btn.setAttribute("aria-label", "코드 복사");
    pre.appendChild(btn);
    btn.addEventListener("click", function () {
      var code = pre.querySelector("code");
      var text = code ? code.textContent : pre.textContent;
      navigator.clipboard.writeText(text).then(function () {
        btn.textContent = "복사됨";
        btn.disabled = true;
        setTimeout(function () {
          btn.textContent = "복사";
          btn.disabled = false;
        }, 1500);
      });
    });
  });

  var headings = document.querySelectorAll(".article section > h2");
  var tocMount = document.getElementById("toc");
  if (tocMount && headings.length) {
    var heading = document.createElement("p");
    heading.className = "eyebrow";
    heading.textContent = "contents";
    tocMount.appendChild(heading);

    var list = document.createElement("ol");
    list.className = "toc-list";
    headings.forEach(function (h, i) {
      if (!h.id) h.id = "section-" + (i + 1);
      var li = document.createElement("li");
      var a = document.createElement("a");
      a.href = "#" + h.id;
      a.textContent = h.textContent;
      li.appendChild(a);
      list.appendChild(li);
    });
    tocMount.appendChild(list);
    tocMount.hidden = false;

    var links = list.querySelectorAll("a");
    var observer = new IntersectionObserver(
      function (entries) {
        entries.forEach(function (entry) {
          if (!entry.isIntersecting) return;
          links.forEach(function (a) {
            a.removeAttribute("aria-current");
          });
          var active = list.querySelector('a[href="#' + entry.target.id + '"]');
          if (active) active.setAttribute("aria-current", "true");
        });
      },
      { rootMargin: "-20% 0px -70% 0px" }
    );
    headings.forEach(function (h) {
      observer.observe(h);
    });
  }
})();
