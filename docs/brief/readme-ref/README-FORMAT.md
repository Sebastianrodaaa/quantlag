# README format for every project (Sebastian, 2026-11-02)
Model: https://github.com/ghgoodreau/athanor/blob/main/README.md (copy here: athanor-README.md, banner: athanor-banner.png)
PALETTE: black and white only. Pure black background, white/grey type and art. No sepia, no color accents anywhere (banner, screenshots, diagrams, badges).

## Banner (similar to athanor's)
- 1280x400 PNG at docs/screenshots/banner.png, first line of README: ![name](docs/screenshots/banner.png)
- Centered project name in a large pixel blackletter/gothic bitmap font, lowercase.
- Under it, a 1-2 line lowercase monospace tagline.
- Left and right edges: public-domain engravings (e.g. Dore, Durer) relevant to the theme, dithered to two tones (1-bit/ordered or Floyd-Steinberg), fading into the black. Credit sources in NOTICE.md.

## Structure
1. Banner, then 2-3 plain sentences saying what it is.
2. One large hero screenshot of the real product/UI right away.
3. Short themed "## The X" sections (e.g. The Method, The Results, The Screen), each a few short plain paragraphs followed by one real screenshot or plot. Inline `code` for commands.
4. A 2x2 markdown image grid where it helps (| | | table).
5. ## Install: one sh block (clone, dry run, run), then what it touches and how to undo.
6. ## The Name: a short paragraph on the name's origin.
7. ## Inspirations: short, links INSPIRATIONS.md / related work.
8. ## License: MIT, third-party terms in NOTICE.md.
Voice: calm, literal, short sentences, a little wit, no hype, no emoji.
Keep the hashtags/topics and a single small row of black-and-white shields.io badges (style=flat-square, color black/white) under the tagline paragraph; demo GIF/video from a real run stays required.
