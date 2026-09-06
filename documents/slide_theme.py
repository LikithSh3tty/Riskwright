"""Terminal Green theme: the CSS and the deck shell.

Kept apart from `build_slides.py` so the presentation's look sits in one file
and its content in another. Changing the palette should never mean touching a
figure, and adding a figure should never mean touching the CSS.

Style follows the Frontend Slides skill's Terminal Green preset -- JetBrains
Mono throughout, GitHub dark ground, terminal green accent, scan lines and a
blinking cursor. Chosen because this deck is twenty slides of measured
quantities: confusion matrices, four-fifths ratios, association rankings, a
cost sweep and a literal SQL comparison. Monospace aligns tabular columns for
free, and the register reads as engineering rather than sales, which is what a
technical submission should sound like.

Fixed-stage invariants come from the skill and are non-negotiable:

  - every slide authored at 1920x1080 inside `.deck-stage`
  - the stage scaled as a whole by one transform, letterboxed, never reflowed
  - slide switching via visibility/opacity/pointer-events, never `display`,
    because a later `.slide-content { display: flex }` would override it and
    show every slide at once
  - `prefers-reduced-motion` respected
  - no negated CSS functions; `calc(-1 * ...)` where a negative is needed

The skill instructs that `viewport-base.css` be included verbatim. That file is
referenced by the skill but is not shipped in the repository, so the stage CSS
below is written to the documented invariants instead. The same applies to
`scripts/export-pdf.sh`; see `export_slides_pdf.py`.
"""

from __future__ import annotations

STAGE_WIDTH = 1920
STAGE_HEIGHT = 1080

FONT_LINK = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">\n'
    '    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>\n'
    '    <link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
    'family=JetBrains+Mono:ital,wght@0,300;0,400;0,500;0,700;0,800;1,400&display=swap">'
)

CSS = """
/* ===========================================================================
   THEME TOKENS
   Terminal Green. Change these and the whole deck follows.
   Authored at the 1920x1080 stage size, so every length here is a stage pixel
   rather than a viewport unit.
   =========================================================================== */
:root {
    --bg:            #0d1117;   /* GitHub dark ground */
    --bg-panel:      #161b22;
    --bg-raised:     #1c2128;
    --border:        #30363d;
    --border-bright: #484f58;

    --green:         #39d353;   /* terminal green, the one accent */
    --green-dim:     #26a641;
    --green-glow:    rgba(57, 211, 83, 0.16);
    --red:           #f85149;
    --amber:         #d29922;
    --blue:          #58a6ff;

    --text:          #e6edf3;
    --text-dim:      #8b949e;
    --text-faint:    #6e7681;

    --font: 'JetBrains Mono', ui-monospace, 'Cascadia Code', monospace;

    --h1:      104px;
    --h2:      58px;
    --sub:     28px;
    --body:    25px;
    --small:   21px;
    --tiny:    18px;

    --pad:     84px;
    --gap:     30px;

    --ease:    cubic-bezier(0.16, 1, 0.3, 1);
    --dur:     0.55s;
}

* { margin: 0; padding: 0; box-sizing: border-box; }

/* ===========================================================================
   FIXED 16:9 STAGE
   The viewport fills the window; the stage is always 1920x1080 and is scaled
   as a single unit. Content never reflows -- on a phone the deck letterboxes
   rather than rearranging, which is what keeps a slide a slide.
   =========================================================================== */
html, body {
    width: 100%;
    height: 100%;
    overflow: hidden;
    background: #000;
}

.deck-viewport {
    position: fixed;
    inset: 0;
    display: block;
    background: #000;
}

.deck-stage {
    position: absolute;
    top: 0;
    left: 0;
    width: 1920px;
    height: 1080px;
    transform-origin: 0 0;
    background: var(--bg);
    overflow: hidden;
}

/* Slide visibility. Deliberately not `display`: a later
   `.slide-content { display: flex }` would win the cascade and every slide
   would paint at once. */
.slide {
    position: absolute;
    inset: 0;
    padding: var(--pad);
    visibility: hidden;
    opacity: 0;
    pointer-events: none;
    transition: opacity 0.4s var(--ease);
    font-family: var(--font);
    color: var(--text);
    font-size: var(--body);
    line-height: 1.5;
}

.slide.active {
    visibility: visible;
    opacity: 1;
    pointer-events: auto;
}

/* ===========================================================================
   ATMOSPHERE
   Scan lines and a faint corner glow, the Terminal Green signature. Both are
   pointer-events:none overlays so they never intercept a click.
   =========================================================================== */
.deck-stage::before {
    content: '';
    position: absolute;
    inset: 0;
    z-index: 900;
    pointer-events: none;
    background: repeating-linear-gradient(
        to bottom,
        rgba(255, 255, 255, 0.014) 0px,
        rgba(255, 255, 255, 0.014) 1px,
        transparent 1px,
        transparent 3px
    );
}

.deck-stage::after {
    content: '';
    position: absolute;
    inset: 0;
    z-index: 899;
    pointer-events: none;
    background:
        radial-gradient(circle at 88% 8%, var(--green-glow), transparent 42%),
        radial-gradient(circle at 4% 96%, rgba(88, 166, 255, 0.07), transparent 38%);
}

/* ===========================================================================
   SLIDE FURNITURE
   A terminal prompt instead of a title bar, and a slide counter bottom-right.
   =========================================================================== */
.slide-head { margin-bottom: 40px; }

.prompt {
    font-size: var(--small);
    color: var(--green-dim);
    letter-spacing: 0.06em;
    margin-bottom: 14px;
    font-weight: 500;
}
.prompt .sigil { color: var(--text-faint); }

h1 {
    font-size: var(--h1);
    font-weight: 800;
    letter-spacing: -0.035em;
    line-height: 1.02;
    color: var(--text);
}

h2 {
    font-size: var(--h2);
    font-weight: 700;
    letter-spacing: -0.028em;
    line-height: 1.08;
    color: var(--text);
}

.subtitle {
    font-size: var(--sub);
    color: var(--text-dim);
    font-weight: 400;
    margin-top: 16px;
    max-width: 1500px;
}

.rule {
    height: 2px;
    background: linear-gradient(to right, var(--green), var(--border) 62%, transparent);
    margin: 26px 0 0;
}

.slide-no {
    position: absolute;
    right: var(--pad);
    bottom: 44px;
    font-size: var(--tiny);
    color: var(--text-faint);
    letter-spacing: 0.14em;
    z-index: 950;
}

/* Blinking cursor, title slide only. Honours reduced motion below. */
.cursor {
    display: inline-block;
    width: 0.58em;
    height: 0.94em;
    background: var(--green);
    margin-left: 0.1em;
    vertical-align: -0.08em;
    animation: blink 1.15s steps(1) infinite;
}
@keyframes blink { 0%, 49% { opacity: 1 } 50%, 100% { opacity: 0 } }

/* ===========================================================================
   LAYOUT
   =========================================================================== */
.cols {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 52px;
    align-items: start;
}
.cols.wide-left  { grid-template-columns: 1.35fr 1fr; }
.cols.wide-right { grid-template-columns: 1fr 1.35fr; }

.stack { display: flex; flex-direction: column; gap: var(--gap); }

/* ===========================================================================
   PANELS, TABLES, CHIPS
   Terminal output rather than slideware: hard corners, hairline borders,
   green header rows.
   =========================================================================== */
.panel {
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-left: 3px solid var(--green-dim);
    padding: 26px 30px;
}
.panel.plain { border-left-color: var(--border); }
.panel.warn  { border-left-color: var(--red); }

.panel-label {
    font-size: var(--tiny);
    text-transform: uppercase;
    letter-spacing: 0.18em;
    color: var(--text-faint);
    margin-bottom: 12px;
}

table {
    width: 100%;
    border-collapse: collapse;
    font-size: var(--small);
    font-variant-numeric: tabular-nums;
}
th {
    text-align: left;
    font-weight: 700;
    color: var(--green);
    border-bottom: 2px solid var(--border-bright);
    padding: 12px 16px 11px;
    font-size: var(--tiny);
    letter-spacing: 0.09em;
    text-transform: uppercase;
    white-space: nowrap;
}
td {
    padding: 11px 16px;
    border-bottom: 1px solid var(--border);
    color: var(--text);
}
tr:last-child td { border-bottom: none; }
tr.hi td {
    background: rgba(57, 211, 83, 0.09);
    font-weight: 700;
    color: var(--text);
}
td.num, th.num { text-align: right; }
td.dim { color: var(--text-dim); }

.chip {
    display: inline-block;
    padding: 3px 12px 4px;
    font-size: var(--tiny);
    font-weight: 700;
    letter-spacing: 0.09em;
    border: 1px solid;
}
.chip.pass { color: var(--green); border-color: var(--green-dim);
             background: rgba(57, 211, 83, 0.10); }
.chip.fail { color: var(--red);   border-color: var(--red);
             background: rgba(248, 81, 73, 0.10); }

/* Big single figures, for the moments a number carries the slide. */
.figure-row { display: flex; gap: 64px; flex-wrap: wrap; }
.figure .value {
    font-size: 76px;
    font-weight: 800;
    letter-spacing: -0.04em;
    line-height: 1;
    color: var(--green);
    font-variant-numeric: tabular-nums;
}
.figure .value.neutral { color: var(--text); }
.figure .value.bad     { color: var(--red); }
.figure .label {
    font-size: var(--tiny);
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: 0.15em;
    margin-top: 10px;
}

ul { list-style: none; }
li {
    position: relative;
    padding-left: 30px;
    margin-bottom: 13px;
    color: var(--text-dim);
    font-size: var(--body);
}
li::before {
    content: '>';
    position: absolute;
    left: 0;
    color: var(--green-dim);
    font-weight: 700;
}
li strong { color: var(--text); font-weight: 700; }
li.bad::before { content: '!'; color: var(--red); }

p { color: var(--text-dim); font-size: var(--body); }
p strong, .lede strong { color: var(--text); font-weight: 700; }
p + p { margin-top: 16px; }
.lede { color: var(--text); font-size: 30px; line-height: 1.42; }
code, .mono-block {
    font-family: var(--font);
    color: var(--green);
    background: rgba(57, 211, 83, 0.08);
    padding: 1px 7px;
    font-size: 0.92em;
}
.mono-block {
    display: block;
    background: #010409;
    border: 1px solid var(--border);
    color: var(--text-dim);
    padding: 22px 26px;
    font-size: var(--small);
    line-height: 1.62;
    white-space: pre;
    overflow: hidden;
}
.mono-block .ok  { color: var(--green); }
.mono-block .no  { color: var(--red); }
.mono-block .key { color: var(--blue); }

.note { font-size: var(--small); color: var(--text-faint); line-height: 1.5; }
.note.alert { color: var(--red); }
.note.good  { color: var(--green); }

/* Horizontal bars for the SHAP ranking. Width is set inline from the data. */
.bar-row {
    display: grid;
    grid-template-columns: 360px 1fr 92px;
    align-items: center;
    gap: 18px;
    margin-bottom: 10px;
    font-size: 19px;
}
.bar-row .name { color: var(--text-dim); text-align: right;
                 white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.bar-track { background: var(--bg-raised); height: 24px; border: 1px solid var(--border); }
.bar-fill  { height: 100%; background: linear-gradient(to right, var(--green-dim), var(--green)); }
.bar-row .val { color: var(--text); font-variant-numeric: tabular-nums; }

/* Screenshots. Direct file paths, per the skill -- the deck is viewed from
   the folder it lives in, so base64 would only inflate the file. */
.shot {
    width: 100%;
    max-height: 700px;
    object-fit: contain;
    border: 1px solid var(--border-bright);
    background: var(--bg-panel);
}

/* ===========================================================================
   ENTRANCE ANIMATION
   One orchestrated reveal per slide, staggered. High-impact once beats
   scattered micro-interactions.
   =========================================================================== */
.reveal {
    opacity: 0;
    transform: translateY(22px);
    transition: opacity var(--dur) var(--ease), transform var(--dur) var(--ease);
}
.slide.active .reveal { opacity: 1; transform: translateY(0); }
.slide.active .reveal:nth-child(1) { transition-delay: 0.06s; }
.slide.active .reveal:nth-child(2) { transition-delay: 0.14s; }
.slide.active .reveal:nth-child(3) { transition-delay: 0.22s; }
.slide.active .reveal:nth-child(4) { transition-delay: 0.30s; }
.slide.active .reveal:nth-child(5) { transition-delay: 0.38s; }
.slide.active .reveal:nth-child(6) { transition-delay: 0.46s; }

/* ===========================================================================
   PROGRESS BAR  (outside the stage, so it is not part of the 16:9 canvas)
   =========================================================================== */
.progress {
    position: fixed;
    left: 0; bottom: 0;
    height: 3px;
    background: var(--green);
    z-index: 10000;
    transition: width 0.35s var(--ease);
}

/* ===========================================================================
   REDUCED MOTION
   =========================================================================== */
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation-duration: 0.001ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.001ms !important;
    }
    .reveal { opacity: 1; transform: none; }
    .cursor { opacity: 1; }
}

/* Print/export: the PDF exporter screenshots each slide, so nothing here
   should hide content when a slide is forced visible. */
.slide.export-visible { visibility: visible; opacity: 1; }
.slide.export-visible .reveal { opacity: 1; transform: none; }
"""


CONTROLLER_JS = """
/* ===========================================================================
   SLIDE CONTROLLER
   Scales the fixed stage to the window, and drives navigation by keyboard,
   wheel, and touch. Nothing here reflows a slide: the only geometry change
   is one transform on the stage.
   =========================================================================== */
class SlidePresentation {
    constructor() {
        this.slides   = Array.from(document.querySelectorAll('.slide'));
        this.stage    = document.getElementById('deckStage');
        this.progress = document.getElementById('progress');
        this.current  = 0;

        this.setupStageScale();
        this.setupKeyboardNav();
        this.setupWheelNav();
        this.setupTouchNav();
        this.show(0);
    }

    /* Uniform scale + centring. Letterboxes on tall windows, pillarboxes on
       wide ones. Never re-lays-out the slide. */
    setupStageScale() {
        const scale = () => {
            const f = Math.min(window.innerWidth / 1920, window.innerHeight / 1080);
            const x = (window.innerWidth  - 1920 * f) / 2;
            const y = (window.innerHeight - 1080 * f) / 2;
            this.stage.style.transform = `translate(${x}px, ${y}px) scale(${f})`;
        };
        scale();
        window.addEventListener('resize', scale);
    }

    setupKeyboardNav() {
        document.addEventListener('keydown', (e) => {
            const forward = ['ArrowRight', 'ArrowDown', ' ', 'PageDown'];
            const back    = ['ArrowLeft', 'ArrowUp', 'PageUp'];
            if (forward.includes(e.key)) { e.preventDefault(); this.next(); }
            else if (back.includes(e.key)) { e.preventDefault(); this.prev(); }
            else if (e.key === 'Home') { e.preventDefault(); this.show(0); }
            else if (e.key === 'End')  { e.preventDefault(); this.show(this.slides.length - 1); }
        });
    }

    /* Throttled, or a single trackpad flick skips five slides. */
    setupWheelNav() {
        let locked = false;
        window.addEventListener('wheel', (e) => {
            if (locked || Math.abs(e.deltaY) < 18) return;
            locked = true;
            e.deltaY > 0 ? this.next() : this.prev();
            setTimeout(() => { locked = false; }, 480);
        }, { passive: true });
    }

    setupTouchNav() {
        let startX = 0, startY = 0;
        window.addEventListener('touchstart', (e) => {
            startX = e.touches[0].clientX;
            startY = e.touches[0].clientY;
        }, { passive: true });
        window.addEventListener('touchend', (e) => {
            const dx = e.changedTouches[0].clientX - startX;
            const dy = e.changedTouches[0].clientY - startY;
            if (Math.abs(dx) > 55 && Math.abs(dx) > Math.abs(dy)) {
                dx < 0 ? this.next() : this.prev();
            }
        }, { passive: true });
    }

    next() { this.show(this.current + 1); }
    prev() { this.show(this.current - 1); }

    show(i) {
        this.current = Math.max(0, Math.min(i, this.slides.length - 1));
        this.slides.forEach((s, n) => s.classList.toggle('active', n === this.current));
        const pct = ((this.current + 1) / this.slides.length) * 100;
        this.progress.style.width = pct + '%';
        if (location.hash !== '#' + (this.current + 1)) {
            history.replaceState(null, '', '#' + (this.current + 1));
        }
    }
}

const deck = new SlidePresentation();

/* Deep link: /slides.html#7 opens slide 7. */
const fromHash = parseInt((location.hash || '').slice(1), 10);
if (!Number.isNaN(fromHash)) deck.show(fromHash - 1);

/* Hook the PDF exporter uses to force one slide visible at a time. */
window.__deckShow = (i) => deck.show(i);
window.__deckCount = deck.slides.length;
"""


def document(title: str, slides_html: str) -> str:
    """Assemble the single self-contained file."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    {FONT_LINK}
    <style>{CSS}</style>
</head>
<body>
    <div class="deck-viewport">
        <main class="deck-stage" id="deckStage">
{slides_html}
        </main>
    </div>
    <div class="progress" id="progress"></div>
    <script>{CONTROLLER_JS}</script>
</body>
</html>
"""
