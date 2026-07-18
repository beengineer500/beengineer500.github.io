# beengineer500.github.io Design System

## 1. Atmosphere & Identity

An engineer's notebook: plain, organized, and built for re-reading. The signature is a narrow memo surface with clear dated entries, quiet dividers, and a small notebook mark instead of a photographic hero.

## 2. Color

### Palette

| Role | Token | Light | Dark | Usage |
|------|-------|-------|------|-------|
| Surface/page | --surface-page | #cdccc5 | #161615 | Page canvas |
| Surface/panel | --surface-panel | #d7d6cf | #201f1e | Main reading surface |
| Surface/muted | --surface-muted | #c1c0b8 | #292827 | Tags, code, quiet blocks |
| Text/primary | --text-primary | #1a1a18 | #ecebe6 | Headings and body |
| Text/secondary | --text-secondary | #47463f | #b4b3ac | Metadata and descriptions |
| Text/tertiary | --text-tertiary | #55544c | #8f8e86 | Low-priority labels |
| Border/subtle | --border-subtle | #b2b1a8 | #33322f | Dividers |
| Border/strong | --border-strong | #98978e | #55544e | Focused structural lines |
| Accent/primary | --accent-primary | #1a1a18 | #e6e5df | Links and focus |
| Accent/hover | --accent-hover | #000000 | #ffffff | Link hover |
| Accent/warm | --accent-warm | #5d5c54 | #9a998f | Second accent: hover feedback, tag hover, active TOC entry, sketch accent |
| Accent/warm hover | --accent-warm-hover | #3e3d36 | #c2c1b6 | Warm accent hover/active state |

### Rules

- The palette is grayscale "ink on paper" (E-Ink aesthetic): neutral warm-gray paper surfaces, charcoal/off-white ink for text, no saturated hues anywhere.
- `--accent-primary` and `--accent-warm` are both neutral ink tones, not colors — they exist to carry emphasis (links, focus, hover feedback, active TOC entry, sketch accent), not to add visual color.
- The page must stay calm: no black ribbons, no gradients, no decorative color blocks.
- Dark mode is provided both via `:root[data-theme="dark"]` (manual override, saved to `localStorage`) and `@media (prefers-color-scheme: dark)` (OS default, guarded by `:not([data-theme="light"])` so a manual choice always wins).
- New colors must be added here before use, and must stay within the grayscale/ink palette.

## 3. Typography

### Scale

| Level | Size | Weight | Line Height | Tracking | Usage |
|-------|------|--------|-------------|----------|-------|
| Display | clamp(2.3rem, 6vw, 3.5rem) | 700 | 1.08 | 0 | Home title |
| H1 | clamp(2rem, 5vw, 3rem) | 700 | 1.14 | 0 | Article title |
| H2 | clamp(1.35rem, 3vw, 1.75rem) | 700 | 1.25 | 0 | Section heading |
| H3 | 1.25rem | 650 | 1.35 | 0 | Note titles |
| Body/lg | 1.125rem | 400 | 1.75 | 0 | Intro copy |
| Body | 1rem | 400 | 1.75 | 0 | Default text |
| Body/sm | 0.875rem | 400 | 1.6 | 0 | Secondary copy |
| Caption | 0.75rem | 600 | 1.45 | 0.02em | Date, kind, labels |

### Font Stack

- Primary: -apple-system, BlinkMacSystemFont, "Apple SD Gothic Neo", "Noto Sans KR", "Segoe UI", system-ui, sans-serif
- Mono: "JetBrains Mono", "SF Mono", Monaco, Consolas, "Liberation Mono", monospace (`--font-mono`)

### Rules

- No display serif for Korean headings.
- Body width stays near 65 characters.
- Korean text uses `word-break: keep-all` and `text-wrap: pretty` where supported.
- JetBrains Mono is self-hosted (`/assets/jetbrains-mono-400.woff2`, `/assets/jetbrains-mono-600.woff2`, `font-display: swap`) and applies only to elements already on the mono stack (eyebrows, dates/status lines, tags, code). All mono-styled copy on this site is Latin/ASCII, so no Korean fallback gap.

## 4. Spacing & Layout

### Base Unit

All spacing derives from 4px.

| Token | Value | Usage |
|-------|-------|-------|
| --space-1 | 4px | Tight inline spacing |
| --space-2 | 8px | Metadata gaps |
| --space-3 | 12px | Compact padding |
| --space-4 | 16px | Mobile gutters |
| --space-5 | 20px | Note internals |
| --space-6 | 24px | Default section gap |
| --space-8 | 32px | Section rhythm |
| --space-10 | 40px | Major group gap |
| --space-12 | 48px | Page intro gap |
| --space-16 | 64px | Large top/bottom rhythm |

### Grid

- Max content width: 880px
- Reading width: 700px
- Layout: one primary content column with dated memo rows
- Breakpoint: 720px

### Rules

- Content comes before decoration.
- Repeated items use rows and dividers, not floating cards.

## 5. Components

### Header
- **Structure**: skip link, brand, navigation, theme toggle.
- **Variants**: inline desktop, wrapped mobile.
- **Spacing**: `--space-3` to `--space-6`.
- **States**: link default, hover, focus.
- **Accessibility**: landmarks, visible focus, sufficient touch targets.
- **Motion**: color transition only.

### Theme Toggle
- **Structure**: bordered mono button with a small icon (☀/☾) and a text label; sits at the end of the header nav group.
- **Variants**: light active, dark active.
- **Spacing**: `--space-1` by `--space-3`.
- **States**: default, hover (border and warm-accent text), focus.
- **Accessibility**: `aria-pressed` reflects dark state, `aria-label` names the action ("다크/라이트 모드로 전환"); never icon-only.
- **Behavior**: follows OS `prefers-color-scheme` until the user clicks; the choice is saved to `localStorage` and applied via `data-theme` on `<html>`. A tiny inline head script sets `data-theme` before first paint to avoid a flash.
- **Motion**: color and border transition only.

### Memo Intro
- **Structure**: label, title, lead, small status line.
- **Variants**: home only.
- **Spacing**: `--space-8` to `--space-12`.
- **States**: static.
- **Accessibility**: one H1, clear reading order.
- **Motion**: none.

### Note Row
- **Structure**: date/kind column, title link, summary, tags.
- **Variants**: linked note and planned note.
- **Spacing**: `--space-4` to `--space-6`.
- **States**: title hover, focus.
- **Accessibility**: title remains the link target; metadata stays text.
- **Motion**: color transition only.

### Archive Header
- **Structure**: label, H1, lead, compact status line.
- **Variants**: all posts, category archive, tag archive.
- **Spacing**: `--space-8` below the title block.
- **States**: static.
- **Accessibility**: one H1 per archive page and clear landmark flow.
- **Motion**: none.

### Taxonomy Row
- **Structure**: term name, description, count.
- **Variants**: category row and tag row.
- **Spacing**: `--space-5` vertical rhythm with dividers.
- **States**: row link hover, focus.
- **Accessibility**: the full row is the link target; count is supporting text.
- **Motion**: color transition only.

### Tag Link
- **Structure**: compact text label inside a bordered muted surface.
- **Variants**: linked tag/category and inactive current tag.
- **Spacing**: `--space-1` by `--space-2`.
- **States**: link default, hover, focus.
- **Accessibility**: visible text label; no icon-only taxonomy controls.
- **Motion**: color and border transition only.

### Article Body
- **Structure**: title block, table of contents, prose sections, inline code, code blocks.
- **Variants**: regular post and 404 note.
- **Spacing**: reading width with `--space-6` paragraph rhythm.
- **States**: link hover and focus.
- **Accessibility**: semantic headings and readable line length.
- **Motion**: none.

### Table of Contents
- **Structure**: bordered muted box under the article header, listing article `h2` headings as an ordered list.
- **Variants**: only rendered when an article has `h2` sections; hidden (via the `hidden` attribute) until JS builds it, so pages without JS show nothing rather than an empty box.
- **Spacing**: `--space-4`/`--space-5` padding, `--space-2` list rhythm.
- **States**: link hover, focus, and the currently-scrolled section marked with `aria-current="true"` in warm accent.
- **Accessibility**: `<nav aria-label="목차">`; current-section highlighting is a visual aid only, not required for navigation.
- **Motion**: color transition only; section highlighting uses `IntersectionObserver`, not scroll-linked animation.

### Code Block Copy Button
- **Structure**: small bordered mono button pinned to the top-right corner of `.article pre`.
- **Variants**: idle ("복사"), copied feedback ("복사됨", briefly disabled).
- **Spacing**: `--space-1` by `--space-2`; block padding increases at the top to keep the first code line clear of the button.
- **States**: default, hover (warm accent text), focus, disabled (post-copy feedback).
- **Accessibility**: `aria-label="코드 복사"`; button text also changes so the state is not color-only.
- **Motion**: color and border transition only.

### Footer
- **Structure**: quiet divider, identity line, links.
- **Variants**: stacked mobile, inline desktop.
- **Spacing**: `--space-6` to `--space-8`.
- **States**: link hover and focus.
- **Accessibility**: landmark footer.
- **Motion**: color transition only.

## 6. Motion & Interaction

### Timing

| Type | Duration | Easing | Usage |
|------|----------|--------|-------|
| Micro | 140ms | ease-out | Link and focus transitions |

### Rules

- No decorative motion.
- Every focusable element has a visible focus state.
- `prefers-reduced-motion` disables transitions.

## 7. Depth & Surface

### Strategy

Tonal-shift plus subtle borders.

| Type | Value | Usage |
|------|-------|-------|
| Page tint | var(--surface-page) | Overall notebook canvas |
| Panel | var(--surface-panel) | Main reading area |
| Hairline | 1px solid var(--border-subtle) | Memo row separation |
| Strong line | 1px solid var(--border-strong) | Header/footer emphasis |

No shadows, glass, black slabs, or large photographic hero surfaces.

### Texture

A very quiet notebook texture, applied consistently to both themes. The page canvas is a flat, untextured surface — no dot grid or grain.

| Element | Treatment | Usage |
|---------|-----------|-------|
| Row/section divider | 1px top hairline, `--border-subtle` | Memo rows, taxonomy rows, article sections — a clean top-divider only, no framing |

Texture stays low-contrast and structural — it must read as quiet dividers, not as decoration competing with content.
