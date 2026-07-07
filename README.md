# beengineer500.github.io

Personal static blog served by GitHub Pages.

## Local Preview

```bash
python3 -m http.server 8000
```

Open `http://localhost:8000`.

## Content Model

- `posts/`, `categories/`, `tags/`, and `index.html` are **built HTML** — this is what GitHub Pages serves. Don't hand-edit post pages under `posts/`; edit the markdown source and rebuild.
- `content/posts/<year>/<slug>/index.md` is the markdown source for a **published** post. Any other files in that same folder (images, etc.) are attachments and get copied alongside the built page.
- `content/drafts/` is for **unpublished** drafts. It's git-ignored — anything there stays on this machine only and is never pushed. Move a draft into `content/posts/<year>/<slug>/` when it's ready to publish.
- Each post's frontmatter needs `title`, `description`, `date` (YYYY-MM-DD), `category`, and `tags` (comma-separated).
- Keep `/posts/` as the date-descending post index (hand-maintained, like `categories/` and `tags/`).
- Give each post one category and any number of tags.

## Building Posts

```bash
npm install   # once
npm run build
```

This renders every `content/posts/<year>/<slug>/index.md` into `posts/<year>/<slug>/index.html` using the site's existing template, and copies any attachments in that folder. It does not touch `content/drafts/` and does not regenerate `/posts/`, `/categories/`, or `/tags/` listing pages — update those links by hand when you publish a new post.

## Local Preview

```bash
python3 -m http.server 8000
```

Open `http://localhost:8000`.

## Publishing

Run `npm run build`, commit the generated `posts/<year>/<slug>/` output along with the markdown source, then push to `main`. GitHub Pages serves the site from the repository root.
