# beengineer500.github.io

Personal static blog served by GitHub Pages.

## Local Preview

```bash
python3 -m http.server 8000
```

Open `http://localhost:8000`.

## Content Model

- `posts/`, `categories/`, `tags/`, and `index.html`'s latest-notes block are **built HTML** — this is what GitHub Pages serves. Don't hand-edit post pages under `posts/`, `posts/index.html`, `categories/`, or `tags/`; edit the markdown source and rebuild.
- `content/posts/<year>/<slug>/index.md` is the markdown source for a **published** post. Any other files in that same folder (images, etc.) are attachments and get copied alongside the built page.
- `content/drafts/` is for **unpublished** drafts. It's git-ignored — anything there stays on this machine only and is never pushed. Move a draft into `content/posts/<year>/<slug>/` when it's ready to publish.
- `content/taxonomy.json` is the one hand-authored file for the listing/taxonomy pages — it holds the one-line Korean blurb shown under each category/tag name. Add an entry when you introduce a brand-new category or tag (a generic fallback line is used until then).
- Each post's frontmatter needs `title`, `description`, `date` (YYYY-MM-DD), `category`, and `tags` (comma-separated).
- The home page's "예정" placeholder row (after the real post rows) stays hand-authored; the real post rows above it are generated.
- Give each post one category and any number of tags.

## Building Posts

```bash
npm install   # once
npm run build
```

This renders every `content/posts/<year>/<slug>/index.md` into `posts/<year>/<slug>/index.html` using the site's existing template, copies any attachments in that folder, and regenerates `posts/index.html`, `categories/index.html`, `categories/<slug>.html`, `tags/<slug>.html`, the home page's latest-notes block, and `feed.xml` from the posts' frontmatter plus `content/taxonomy.json`. Run `npm run build` and commit the result whenever you add, edit, or remove a post. It does not touch `content/drafts/`.

## Local Preview

```bash
python3 -m http.server 8000
```

Open `http://localhost:8000`.

## Publishing

Run `npm run build`, commit the generated `posts/<year>/<slug>/` output along with the markdown source, then push to `main`. GitHub Pages serves the site from the repository root.
