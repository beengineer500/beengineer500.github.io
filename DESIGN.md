# beengineer500.github.io Design System

## 1. Atmosphere & Identity

A compact technical editorial desk: quiet, precise, and made for written engineering notes rather than product marketing. The signature is a paper-white broadsheet grid with hard ink rules, a single blue link accent, and a real desk photograph that gives the site a personal working context.

## 2. Color

### Palette

| Role | Token | Light | Dark | Usage |
|------|-------|-------|------|-------|
| Surface/primary | --surface-primary | #ffffff | #111111 | Page canvas |
| Surface/secondary | --surface-secondary | #f8fafc | #1a1a1a | Quiet grouped areas |
| Surface/inverted | --surface-inverted | #111111 | #ffffff | Footer and dark ribbons |
| Text/primary | --text-primary | #1a1a1a | #ffffff | Headlines, body |
| Text/secondary | --text-secondary | #5f6368 | #cbd5e1 | Metadata and descriptions |
| Text/inverted | --text-inverted | #ffffff | #1a1a1a | Text on inverted surfaces |
| Border/default | --border-default | #111111 | #ffffff | Strong editorial rules |
| Border/subtle | --border-subtle | #dbe3ea | #334155 | Dividers and image outlines |
| Accent/primary | --accent-primary | #057dbc | #58b8e8 | Links and focus |
| Accent/hover | --accent-hover | #035f90 | #8bd4f5 | Link hover |

### Rules

- Accent is reserved for interactive links and focus states.
- Large surfaces stay white or ink black; no decorative gradients.
- New colors must be added here before use.

## 3. Typography

### Scale

| Level | Size | Weight | Line Height | Tracking | Usage |
|-------|------|--------|-------------|----------|-------|
| Display | clamp(3rem, 5vw, 4rem) | 700 | 0.96 | 0 | Site title |
| H1 | clamp(2rem, 5vw, 4.25rem) | 700 | 1.02 | 0 | Article title |
| H2 | clamp(1.5rem, 3vw, 2.25rem) | 700 | 1.12 | 0 | Section heading |
| H3 | 1.25rem | 700 | 1.25 | 0 | Story titles |
| Body/lg | 1.25rem | 400 | 1.6 | 0 | Lead copy |
| Body | 1rem | 400 | 1.7 | 0 | Article body |
| Body/sm | 0.875rem | 400 | 1.5 | 0 | Secondary copy |
| Caption | 0.75rem | 700 | 1.4 | 0.08em | Kicker, metadata |

### Font Stack

- Display: Georgia, "Times New Roman", serif
- Primary: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif
- Mono: "SF Mono", Monaco, Consolas, "Liberation Mono", monospace

### Rules

- Korean display text uses `word-break: keep-all` with manual structure when needed.
- Body copy stays at 16px or larger.
- Mono labels are uppercase where they behave as metadata.

## 4. Spacing & Layout

### Base Unit

All spacing derives from 4px.

| Token | Value | Usage |
|-------|-------|-------|
| --space-1 | 4px | Tight inline spacing |
| --space-2 | 8px | Metadata gaps |
| --space-3 | 12px | Compact padding |
| --space-4 | 16px | Default mobile gutters |
| --space-5 | 20px | Section internals |
| --space-6 | 24px | Card and column gap |
| --space-8 | 32px | Medium section gap |
| --space-10 | 40px | Desktop component gap |
| --space-12 | 48px | Major section gap |
| --space-16 | 64px | Hero and article spacing |
| --space-20 | 80px | Wide hero rhythm |

### Grid

- Max content width: 1180px
- Reading width: 720px
- Column system: editorial CSS grid, single column on mobile, two or three columns above tablet widths
- Breakpoints: 680px and 960px

### Rules

- Sections use hairline rules and whitespace, not nested cards.
- Image aspect ratios are explicit to prevent layout shift.

## 5. Components

### Header
- **Structure**: skip link, masthead, primary navigation.
- **Variants**: desktop inline, mobile wrapping.
- **Spacing**: `--space-4` to `--space-8`.
- **States**: link default, hover, focus.
- **Accessibility**: landmarks, visible focus, sufficient touch targets.
- **Motion**: color transition only.

### Hero
- **Structure**: label, display heading, lead, visual asset.
- **Variants**: desktop overlay grid, mobile stacked.
- **Spacing**: `--space-8` to `--space-16`.
- **States**: static content; links inside lead inherit link states.
- **Accessibility**: meaningful image alt text.
- **Motion**: none.

### Story Link
- **Structure**: article, kicker, title link, deck, metadata.
- **Variants**: featured, standard, numbered.
- **Spacing**: `--space-2` to `--space-6`.
- **States**: headline hover changes to accent blue; focus outline.
- **Accessibility**: whole title remains the link target.
- **Motion**: color transition only.

### Article Body
- **Structure**: title block, prose sections, inline links, code samples.
- **Variants**: regular post and 404 note.
- **Spacing**: reading width with `--space-6` paragraph rhythm.
- **States**: link hover and focus.
- **Accessibility**: semantic headings and readable line length.
- **Motion**: none.

### Footer
- **Structure**: inverted band with short identity line and links.
- **Variants**: stacked mobile, inline desktop.
- **Spacing**: `--space-8` to `--space-12`.
- **States**: link hover and focus.
- **Accessibility**: landmark footer and high contrast.
- **Motion**: color transition only.

## 6. Motion & Interaction

### Timing

| Type | Duration | Easing | Usage |
|------|----------|--------|-------|
| Micro | 140ms | ease-out | Link color and focus transitions |

### Rules

- No decorative motion.
- Focus state is always visible.
- `prefers-reduced-motion` disables transitions.

## 7. Depth & Surface

### Strategy

Borders-only.

| Type | Value | Usage |
|------|-------|-------|
| Hairline | 1px solid var(--border-subtle) | Section and story separation |
| Ink rule | 2px solid var(--border-default) | Masthead, primary emphasis |
| Inverted band | var(--surface-inverted) fill | Footer |

No shadows, glass, blur, or rounded containers. The photograph supplies depth; the interface uses type, rule weight, and spacing.
