# beengineer500.github.io

Personal static blog served by GitHub Pages.

## Local Preview

```bash
python3 -m http.server 8000
```

Open `http://localhost:8000`.

## Content Model

- Write posts in `posts/`.
- Keep `/posts/` as the date-descending post index.
- Give each post one category and any number of tags.
- Keep category archives in `categories/`.
- Keep tag archives in `tags/`.
- Add post metadata to the article head with `post-category` and `post-tags`.

## Publishing

Push changes to `main`. GitHub Pages serves the site from the repository root.
