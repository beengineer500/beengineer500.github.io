import { marked } from "marked";
import fs from "node:fs";
import path from "node:path";

const ROOT = path.resolve(import.meta.dirname, "..");
const CONTENT_DIR = path.join(ROOT, "content", "posts");
const OUTPUT_DIR = path.join(ROOT, "posts");
const TAXONOMY_PATH = path.join(ROOT, "content", "taxonomy.json");
const SITE_URL = "https://beengineer500.github.io";
const HERO_IMAGE_URL = `${SITE_URL}/assets/hero-engineering-desk-1280.jpg`;
const SITE_DESCRIPTION =
  "beengineer500의 개인 기술 메모장. 개발, 운영, 자동화 과정에서 배운 것을 짧고 다시 실행 가능하게 기록합니다.";

function slugify(text) {
  return text
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9가-힣]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

function parseFrontmatter(raw) {
  const match = raw.match(/^---\n([\s\S]*?)\n---\n?([\s\S]*)$/);
  if (!match) throw new Error("Missing frontmatter block (---...---)");
  const [, frontmatter, body] = match;
  const data = {};
  for (const line of frontmatter.split("\n")) {
    if (!line.trim()) continue;
    const sep = line.indexOf(":");
    if (sep === -1) continue;
    data[line.slice(0, sep).trim()] = line.slice(sep + 1).trim();
  }
  return { data, body: body.trim() };
}

function readingTimeMinutes(html) {
  const chars = html.replace(/<[^>]+>/g, "").replace(/\s+/g, "").length;
  return Math.max(1, Math.round(chars / 500));
}

function renderSections(markdownBody) {
  const tokens = marked.lexer(markdownBody);
  const sections = [];
  let current = null;
  for (const token of tokens) {
    if (token.type === "heading" && token.depth === 2) {
      current = [token];
      sections.push(current);
    } else {
      if (!current) {
        current = [];
        sections.push(current);
      }
      current.push(token);
    }
  }
  return sections.map((toks) => `<section>\n${marked.parser(toks)}</section>`).join("\n\n");
}

function formatDate(iso) {
  return iso.replace(/-/g, ".");
}

function escapeXml(text) {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&apos;");
}

function toRfc822(dateIso) {
  return new Date(`${dateIso}T00:00:00Z`).toUTCString();
}

function headExtras() {
  return `<link rel="alternate" type="application/rss+xml" title="beengineer500 RSS Feed" href="${SITE_URL}/feed.xml" />
    <meta property="og:image" content="${HERO_IMAGE_URL}" />
    <meta property="og:image:width" content="1280" />
    <meta property="og:image:height" content="720" />`;
}

// ---------------------------------------------------------------------------
// Post page rendering (content/posts/<year>/<slug>/index.md -> posts/<year>/<slug>/index.html)
// ---------------------------------------------------------------------------

function renderPage({ title, description, canonical, dateIso, category, tags, readingMinutes, articleHtml }) {
  const categorySlug = slugify(category);
  const tagLinks = tags
    .map((tag) => `<a href="/tags/${slugify(tag)}.html">${tag}</a>`)
    .join("\n            ");

  return `<!doctype html>
<html lang="ko">
  <head>
    <meta charset="utf-8" />
    <script>
      (function () { try { var t = localStorage.getItem("theme"); if (t) document.documentElement.setAttribute("data-theme", t); } catch (e) {} })();
    </script>
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>${title} · beengineer500</title>
    <meta name="post-category" content="${categorySlug}" />
    <meta name="post-tags" content="${tags.join(", ")}" />
    <meta name="description" content="${description}" />
    <link rel="canonical" href="${canonical}" />
    <link rel="icon" href="/favicon.svg" type="image/svg+xml" />
    ${headExtras()}
    <link rel="preload" href="/styles.css" as="style" />
    <link rel="stylesheet" href="/styles.css" />
    <script src="/main.js" defer></script>
  </head>
  <body>
    <a class="skip-link" href="#main">본문으로 이동</a>

    <header class="site-header" aria-label="사이트 헤더">
      <div class="shell masthead">
        <a class="brand" href="/" aria-label="beengineer500 홈">
          <span>beengineer500</span>
        </a>
        <div class="nav-group">
          <nav class="site-nav" aria-label="주요 메뉴">
            <a href="/">홈</a>
            <a href="/posts/">글</a>
            <a href="/categories/">분류</a>
            <a href="https://github.com/beengineer500">GitHub</a>
          </nav>
          <button type="button" class="theme-toggle" id="theme-toggle" aria-pressed="false">
            <span class="theme-toggle-icon" aria-hidden="true">☀</span>
            <span class="theme-toggle-label">라이트</span>
          </button>
        </div>
      </div>
    </header>

    <main id="main" class="shell article-shell" tabindex="-1">
      <article class="article">
        <header class="article-header">
          <p class="eyebrow">${categorySlug}</p>
          <h1>${title}</h1>
          <p class="lead">
            ${description}
          </p>
          <p class="status-line">${formatDate(dateIso)} · ${readingMinutes} min read</p>
          <div class="note-tags article-tags" aria-label="분류와 태그">
            <a href="/categories/${categorySlug}.html">${category}</a>
            ${tagLinks}
          </div>
        </header>

        <nav class="toc" id="toc" aria-label="목차" hidden></nav>

        ${articleHtml}

        <p class="article-back">
          <a href="/posts/">글 인덱스로 돌아가기</a>
        </p>
      </article>
    </main>

    <footer class="site-footer">
      <div class="shell footer-grid">
        <p>beengineer500 · personal engineering notes</p>
        <nav aria-label="푸터 링크">
          <a href="/">홈</a>
          <a href="/posts/">글 인덱스</a>
          <a href="/categories/">분류</a>
          <a href="https://github.com/beengineer500">GitHub</a>
        </nav>
      </div>
    </footer>
  </body>
</html>
`;
}

function buildPost(year, slug) {
  const srcDir = path.join(CONTENT_DIR, year, slug);
  const mdPath = path.join(srcDir, "index.md");
  const raw = fs.readFileSync(mdPath, "utf8");
  const { data, body } = parseFrontmatter(raw);

  const required = ["title", "description", "date", "category", "tags"];
  for (const key of required) {
    if (!data[key]) throw new Error(`${mdPath}: missing "${key}" in frontmatter`);
  }

  const tags = data.tags.split(",").map((t) => t.trim()).filter(Boolean);
  const articleHtml = renderSections(body);
  const canonical = `${SITE_URL}/posts/${year}/${slug}/`;

  const html = renderPage({
    title: data.title,
    description: data.description,
    canonical,
    dateIso: data.date,
    category: data.category,
    tags,
    readingMinutes: readingTimeMinutes(articleHtml),
    articleHtml,
  });

  const outDir = path.join(OUTPUT_DIR, year, slug);
  fs.mkdirSync(outDir, { recursive: true });
  fs.writeFileSync(path.join(outDir, "index.html"), html);

  for (const entry of fs.readdirSync(srcDir)) {
    if (entry === "index.md") continue;
    fs.copyFileSync(path.join(srcDir, entry), path.join(outDir, entry));
  }

  console.log(`built posts/${year}/${slug}/index.html`);
}

// ---------------------------------------------------------------------------
// Post records (frontmatter across all posts, used by every listing/taxonomy page)
// ---------------------------------------------------------------------------

function collectPosts() {
  const posts = [];
  if (!fs.existsSync(CONTENT_DIR)) return posts;

  for (const year of fs.readdirSync(CONTENT_DIR)) {
    const yearDir = path.join(CONTENT_DIR, year);
    if (!fs.statSync(yearDir).isDirectory()) continue;

    for (const slug of fs.readdirSync(yearDir)) {
      const slugDir = path.join(yearDir, slug);
      if (!fs.statSync(slugDir).isDirectory()) continue;
      const mdPath = path.join(slugDir, "index.md");
      if (!fs.existsSync(mdPath)) continue;

      const raw = fs.readFileSync(mdPath, "utf8");
      const { data } = parseFrontmatter(raw);

      const required = ["title", "description", "date", "category", "tags"];
      for (const key of required) {
        if (!data[key]) throw new Error(`${mdPath}: missing "${key}" in frontmatter`);
      }

      const tags = data.tags
        .split(",")
        .map((t) => t.trim())
        .filter(Boolean)
        .map((name) => ({ name, slug: slugify(name) }));

      posts.push({
        year,
        slug,
        title: data.title,
        description: data.description,
        dateIso: data.date,
        category: data.category,
        categorySlug: slugify(data.category),
        tags,
        canonical: `${SITE_URL}/posts/${year}/${slug}/`,
      });
    }
  }

  posts.sort((a, b) => (a.dateIso < b.dateIso ? 1 : a.dateIso > b.dateIso ? -1 : 0));
  return posts;
}

function groupPosts(posts) {
  const categories = new Map();
  const tags = new Map();

  for (const post of posts) {
    if (!categories.has(post.categorySlug)) {
      categories.set(post.categorySlug, { name: post.category, posts: [] });
    }
    categories.get(post.categorySlug).posts.push(post);

    for (const tag of post.tags) {
      if (!tags.has(tag.slug)) {
        tags.set(tag.slug, { name: tag.name, posts: [] });
      }
      tags.get(tag.slug).posts.push(post);
    }
  }

  return { categories, tags };
}

function loadTaxonomy() {
  if (!fs.existsSync(TAXONOMY_PATH)) return { categories: {}, tags: {} };
  return JSON.parse(fs.readFileSync(TAXONOMY_PATH, "utf8"));
}

function describeTerm(taxonomy, kind, slug, name) {
  const entry = taxonomy?.[kind]?.[slug];
  if (entry) return entry;
  return `${name} 관련 기록을 모아둔 페이지입니다.`;
}

// ---------------------------------------------------------------------------
// Shared row renderers
// ---------------------------------------------------------------------------

function renderNoteRow(post, { categoryLinked, selfTagSlug = null, categoryInTags = false }) {
  const categoryHtml = categoryLinked
    ? `<a href="/categories/${post.categorySlug}.html">${post.category}</a>`
    : `<span>${post.category}</span>`;

  const tagItems = [];
  if (categoryInTags) {
    tagItems.push(`<a href="/categories/${post.categorySlug}.html">${post.category}</a>`);
  }
  for (const tag of post.tags) {
    tagItems.push(
      tag.slug === selfTagSlug
        ? `<span>${tag.name}</span>`
        : `<a href="/tags/${tag.slug}.html">${tag.name}</a>`
    );
  }

  return `<article class="note-row">
            <div class="note-meta">
              <time datetime="${post.dateIso}">${formatDate(post.dateIso)}</time>
              ${categoryHtml}
            </div>
            <div class="note-body">
              <h3><a href="/posts/${post.year}/${post.slug}/">${post.title}</a></h3>
              <p>
                ${post.description}
              </p>
              <div class="note-tags" aria-label="태그">
                ${tagItems.join("\n                ")}
              </div>
            </div>
          </article>`;
}

function renderNoteList(posts, rowOptions) {
  const rows = posts.map((post) => renderNoteRow(post, rowOptions)).join("\n\n          ");
  return `<div class="note-list">
          ${rows}
        </div>`;
}

function renderTermRow(kind, slug, name, description, count) {
  const href = kind === "category" ? `/categories/${slug}.html` : `/tags/${slug}.html`;
  return `<a class="term-row" href="${href}">
            <span class="term-main">
              <span class="term-name">${name}</span>
              <span class="term-description">${description}</span>
            </span>
            <span class="term-count">${count} note${count === 1 ? "" : "s"}</span>
          </a>`;
}

function renderTermList(rows) {
  const body = rows.join("\n\n          ");
  return `<div class="term-list">
          ${body}
        </div>`;
}

// ---------------------------------------------------------------------------
// Shared page shell (header/footer chrome around each listing/taxonomy page)
// ---------------------------------------------------------------------------

function renderShell({ title, description, canonical, currentNav, footerLinks, main }) {
  const navItems = [
    { href: "/", label: "홈" },
    { href: "/posts/", label: "글" },
    { href: "/categories/", label: "분류" },
  ];
  const navLinks = navItems
    .map((item) =>
      item.label === currentNav
        ? `<a href="${item.href}" aria-current="page">${item.label}</a>`
        : `<a href="${item.href}">${item.label}</a>`
    )
    .concat('<a href="https://github.com/beengineer500">GitHub</a>')
    .join("\n            ");

  const footerNav = footerLinks
    .map((link) => `<a href="${link.href}">${link.label}</a>`)
    .join("\n          ");

  return `<!doctype html>
<html lang="ko">
  <head>
    <meta charset="utf-8" />
    <script>
      (function () { try { var t = localStorage.getItem("theme"); if (t) document.documentElement.setAttribute("data-theme", t); } catch (e) {} })();
    </script>
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>${title} · beengineer500</title>
    <meta name="description" content="${description}" />
    <link rel="canonical" href="${canonical}" />
    <link rel="icon" href="/favicon.svg" type="image/svg+xml" />
    ${headExtras()}
    <link rel="preload" href="/styles.css" as="style" />
    <link rel="stylesheet" href="/styles.css" />
    <script src="/main.js" defer></script>
  </head>
  <body>
    <a class="skip-link" href="#main">본문으로 이동</a>

    <header class="site-header" aria-label="사이트 헤더">
      <div class="shell masthead">
        <a class="brand" href="/" aria-label="beengineer500 홈">
          <span>beengineer500</span>
        </a>
        <div class="nav-group">
          <nav class="site-nav" aria-label="주요 메뉴">
            ${navLinks}
          </nav>
          <button type="button" class="theme-toggle" id="theme-toggle" aria-pressed="false">
            <span class="theme-toggle-icon" aria-hidden="true">☀</span>
            <span class="theme-toggle-label">라이트</span>
          </button>
        </div>
      </div>
    </header>

    <main id="main" class="shell archive-shell" tabindex="-1">
${main}
    </main>

    <footer class="site-footer">
      <div class="shell footer-grid">
        <p>beengineer500 · personal engineering notes</p>
        <nav aria-label="푸터 링크">
          ${footerNav}
        </nav>
      </div>
    </footer>
  </body>
</html>
`;
}

// ---------------------------------------------------------------------------
// posts/index.html
// ---------------------------------------------------------------------------

function yearRangeLabel(posts) {
  const years = [...new Set(posts.map((p) => p.year))].sort();
  if (years.length === 0) return "";
  if (years.length === 1) return years[0];
  return `${years[0]}–${years[years.length - 1]}`;
}

function buildPostsIndexPage(posts) {
  const years = [...new Set(posts.map((p) => p.year))].sort((a, b) => b.localeCompare(a));

  const yearSections = years
    .map((year) => {
      const yearPosts = posts.filter((p) => p.year === year);
      return `<section class="section-block" aria-labelledby="year-${year}">
        <div class="section-heading">
          <p class="eyebrow">${year}</p>
          <h2 id="year-${year}">${year}년</h2>
        </div>

        ${renderNoteList(yearPosts, { categoryLinked: true })}
      </section>`;
    })
    .join("\n\n      ");

  const count = posts.length;
  const statusLine = `${count} published note${count === 1 ? "" : "s"} · ${yearRangeLabel(posts)}`;

  const main = `      <section class="archive-header" aria-labelledby="archive-title">
        <p class="eyebrow">post index</p>
        <h1 id="archive-title">글 인덱스</h1>
        <p class="lead">
          모든 메모를 날짜 역순으로 둡니다. 제목, 요약, 분류를 한 번에 보고
          필요한 기록을 바로 찾을 수 있게 유지합니다.
        </p>
        <p class="status-line">${statusLine}</p>
      </section>

      ${yearSections}`;

  const html = renderShell({
    title: "글 인덱스",
    description: "beengineer500 블로그의 모든 기술 메모를 날짜 역순으로 모아둔 글 인덱스입니다.",
    canonical: `${SITE_URL}/posts/`,
    currentNav: "글",
    footerLinks: [
      { href: "/", label: "홈" },
      { href: "/categories/", label: "분류" },
      { href: "https://github.com/beengineer500", label: "GitHub" },
    ],
    main,
  });

  fs.writeFileSync(path.join(ROOT, "posts", "index.html"), html);
  console.log("built posts/index.html");
}

// ---------------------------------------------------------------------------
// categories/index.html
// ---------------------------------------------------------------------------

function buildCategoriesIndexPage(posts, taxonomy) {
  const { categories, tags } = groupPosts(posts);

  const categoryRows = [...categories.entries()].map(([slug, entry]) =>
    renderTermRow("category", slug, entry.name, describeTerm(taxonomy, "categories", slug, entry.name), entry.posts.length)
  );
  const tagRows = [...tags.entries()].map(([slug, entry]) =>
    renderTermRow("tag", slug, entry.name, describeTerm(taxonomy, "tags", slug, entry.name), entry.posts.length)
  );

  const statusLine = `${categories.size} category · ${tags.size} tags · ${posts.length} note${posts.length === 1 ? "" : "s"}`;

  const main = `      <section class="archive-header" aria-labelledby="taxonomy-title">
        <p class="eyebrow">taxonomy</p>
        <h1 id="taxonomy-title">분류</h1>
        <p class="lead">
          카테고리는 글의 큰 맥락을 하나만 고르고, 태그는 다시 찾을 때 쓸 키워드로 붙입니다.
          글이 늘어나도 이 페이지에서 주제별 입구를 유지합니다.
        </p>
        <p class="status-line">${statusLine}</p>
      </section>

      <section class="section-block" aria-labelledby="categories-title">
        <div class="section-heading">
          <p class="eyebrow">categories</p>
          <h2 id="categories-title">카테고리</h2>
        </div>

        ${renderTermList(categoryRows)}
      </section>

      <section class="section-block" aria-labelledby="tags-title">
        <div class="section-heading">
          <p class="eyebrow">tags</p>
          <h2 id="tags-title">태그</h2>
        </div>

        ${renderTermList(tagRows)}
      </section>`;

  const html = renderShell({
    title: "분류",
    description: "beengineer500 블로그의 글을 카테고리와 태그 기준으로 찾아볼 수 있는 분류 인덱스입니다.",
    canonical: `${SITE_URL}/categories/`,
    currentNav: "분류",
    footerLinks: [
      { href: "/", label: "홈" },
      { href: "/posts/", label: "글 인덱스" },
      { href: "https://github.com/beengineer500", label: "GitHub" },
    ],
    main,
  });

  fs.writeFileSync(path.join(ROOT, "categories", "index.html"), html);
  console.log("built categories/index.html");
}

// ---------------------------------------------------------------------------
// categories/<slug>.html
// ---------------------------------------------------------------------------

function buildCategoryPages(posts, taxonomy) {
  const { categories } = groupPosts(posts);

  for (const [slug, entry] of categories) {
    const description = describeTerm(taxonomy, "categories", slug, entry.name);
    const count = entry.posts.length;

    const main = `      <section class="archive-header" aria-labelledby="category-title">
        <p class="eyebrow">category</p>
        <h1 id="category-title">${entry.name}</h1>
        <p class="lead">
          ${description}
        </p>
        <p class="status-line">${count} note${count === 1 ? "" : "s"}</p>
      </section>

      <section class="section-block" aria-labelledby="category-posts-title">
        <div class="section-heading">
          <p class="eyebrow">notes</p>
          <h2 id="category-posts-title">${entry.name} 메모</h2>
        </div>

        ${renderNoteList(entry.posts, { categoryLinked: false })}
      </section>`;

    const html = renderShell({
      title: entry.name,
      description: `${entry.name} 카테고리에 속한 beengineer500 블로그 메모를 모아둔 페이지입니다.`,
      canonical: `${SITE_URL}/categories/${slug}.html`,
      currentNav: "분류",
      footerLinks: [
        { href: "/", label: "홈" },
        { href: "/posts/", label: "글 인덱스" },
        { href: "/categories/", label: "분류" },
      ],
      main,
    });

    fs.mkdirSync(path.join(ROOT, "categories"), { recursive: true });
    fs.writeFileSync(path.join(ROOT, "categories", `${slug}.html`), html);
    console.log(`built categories/${slug}.html`);
  }
}

// ---------------------------------------------------------------------------
// tags/<slug>.html
// ---------------------------------------------------------------------------

function buildTagPages(posts, taxonomy) {
  const { tags } = groupPosts(posts);

  for (const [slug, entry] of tags) {
    const description = describeTerm(taxonomy, "tags", slug, entry.name);
    const count = entry.posts.length;

    const main = `      <section class="archive-header" aria-labelledby="tag-title">
        <p class="eyebrow">tag</p>
        <h1 id="tag-title">${entry.name}</h1>
        <p class="lead">
          ${description}
        </p>
        <p class="status-line">${count} note${count === 1 ? "" : "s"}</p>
      </section>

      <section class="section-block" aria-labelledby="tag-posts-title">
        <div class="section-heading">
          <p class="eyebrow">notes</p>
          <h2 id="tag-posts-title">${entry.name} 메모</h2>
        </div>

        ${renderNoteList(entry.posts, { categoryLinked: true, selfTagSlug: slug })}
      </section>`;

    const html = renderShell({
      title: entry.name,
      description: `${entry.name} 태그가 붙은 beengineer500 블로그 메모를 모아둔 페이지입니다.`,
      canonical: `${SITE_URL}/tags/${slug}.html`,
      currentNav: "분류",
      footerLinks: [
        { href: "/", label: "홈" },
        { href: "/posts/", label: "글 인덱스" },
        { href: "/categories/", label: "분류" },
      ],
      main,
    });

    fs.mkdirSync(path.join(ROOT, "tags"), { recursive: true });
    fs.writeFileSync(path.join(ROOT, "tags", `${slug}.html`), html);
    console.log(`built tags/${slug}.html`);
  }
}

// ---------------------------------------------------------------------------
// index.html "최근 메모" block (only the marked region is touched)
// ---------------------------------------------------------------------------

function updateHomeLatestNotes(posts) {
  const indexPath = path.join(ROOT, "index.html");
  const html = fs.readFileSync(indexPath, "utf8");
  const startMarker = "<!-- BUILD:LATEST_NOTES:START -->";
  const endMarker = "<!-- BUILD:LATEST_NOTES:END -->";
  const startIdx = html.indexOf(startMarker);
  const endIdx = html.indexOf(endMarker);
  if (startIdx === -1 || endIdx === -1) {
    throw new Error("index.html is missing BUILD:LATEST_NOTES markers");
  }

  const latest = posts.slice(0, 3);
  const rows = latest
    .map((post) => renderNoteRow(post, { categoryLinked: false, categoryInTags: true }))
    .join("\n\n          ");

  const updated =
    html.slice(0, startIdx + startMarker.length) +
    `\n          ${rows}\n          ` +
    html.slice(endIdx);

  fs.writeFileSync(indexPath, updated);
  console.log("updated index.html latest notes");
}

// ---------------------------------------------------------------------------
// feed.xml
// ---------------------------------------------------------------------------

function buildFeed(posts) {
  const items = posts
    .map(
      (post) => `    <item>
      <title>${escapeXml(post.title)}</title>
      <link>${post.canonical}</link>
      <description>${escapeXml(post.description)}</description>
      <pubDate>${toRfc822(post.dateIso)}</pubDate>
      <guid>${post.canonical}</guid>
    </item>`
    )
    .join("\n");

  const xml = `<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>beengineer500</title>
    <link>${SITE_URL}/</link>
    <description>${escapeXml(SITE_DESCRIPTION)}</description>
${items}
  </channel>
</rss>
`;

  fs.writeFileSync(path.join(ROOT, "feed.xml"), xml);
  console.log("built feed.xml");
}

// ---------------------------------------------------------------------------

function main() {
  const taxonomy = loadTaxonomy();

  if (fs.existsSync(CONTENT_DIR)) {
    for (const year of fs.readdirSync(CONTENT_DIR)) {
      const yearDir = path.join(CONTENT_DIR, year);
      if (!fs.statSync(yearDir).isDirectory()) continue;
      for (const slug of fs.readdirSync(yearDir)) {
        const slugDir = path.join(yearDir, slug);
        if (!fs.statSync(slugDir).isDirectory()) continue;
        if (!fs.existsSync(path.join(slugDir, "index.md"))) continue;
        buildPost(year, slug);
      }
    }
  }

  const posts = collectPosts();
  buildPostsIndexPage(posts);
  buildCategoriesIndexPage(posts, taxonomy);
  buildCategoryPages(posts, taxonomy);
  buildTagPages(posts, taxonomy);
  updateHomeLatestNotes(posts);
  buildFeed(posts);
}

main();
