import { marked } from "marked";
import fs from "node:fs";
import path from "node:path";

const ROOT = path.resolve(import.meta.dirname, "..");
const CONTENT_DIR = path.join(ROOT, "content", "posts");
const OUTPUT_DIR = path.join(ROOT, "posts");
const SITE_URL = "https://beengineer500.github.io";

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

function main() {
  if (!fs.existsSync(CONTENT_DIR)) return;
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

main();
