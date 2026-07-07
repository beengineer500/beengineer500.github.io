# beengineer500.github.io Design System

## 1. Atmosphere & Identity

An engineer's notebook: plain, organized, and built for re-reading. The signature is a narrow memo surface with clear dated entries, quiet dividers, and a small notebook mark instead of a photographic hero.

## 2. Color

### Palette

| Role | Token | Light | Dark | Usage |
|------|-------|-------|------|-------|
| Surface/page | --surface-page | #fbfbfa | #181a19 | Page canvas |
| Surface/panel | --surface-panel | #ffffff | #202322 | Main reading surface |
| Surface/muted | --surface-muted | #f5f6f3 | #282c2a | Tags, code, quiet blocks |
| Text/primary | --text-primary | #202322 | #f5f6f3 | Headings and body |
| Text/secondary | --text-secondary | #4f5a55 | #b8c0bb | Metadata and descriptions |
| Text/tertiary | --text-tertiary | #626b66 | #9ca6a0 | Low-priority labels |
| Border/subtle | --border-subtle | #e2e6e1 | #363d39 | Dividers |
| Border/strong | --border-strong | #c8d0ca | #606a65 | Focused structural lines |
| Accent/primary | --accent-primary | #2c6759 | #7fc8b5 | Links and focus |
| Accent/hover | --accent-hover | #1f4f44 | #a4dfd0 | Link hover |

### Rules

- Accent is only for links, focus states, and selected metadata.
- The page must stay calm: no black ribbons, no gradients, no decorative color blocks.
- New colors must be added here before use.

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
- Mono: "SF Mono", Monaco, Consolas, "Liberation Mono", monospace

### Rules

- No display serif for Korean headings.
- Body width stays near 65 characters.
- Korean text uses `word-break: keep-all` and `text-wrap: pretty` where supported.

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
- **Structure**: skip link, small notebook mark, brand, navigation.
- **Variants**: inline desktop, wrapped mobile.
- **Spacing**: `--space-3` to `--space-6`.
- **States**: link default, hover, focus.
- **Accessibility**: landmarks, visible focus, sufficient touch targets.
- **Motion**: color transition only.

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

### Article Body
- **Structure**: title block, prose sections, inline code, code blocks.
- **Variants**: regular post and 404 note.
- **Spacing**: reading width with `--space-6` paragraph rhythm.
- **States**: link hover and focus.
- **Accessibility**: semantic headings and readable line length.
- **Motion**: none.

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
