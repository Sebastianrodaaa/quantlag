"""Draw docs/screenshots/banner.png (1280x400, black and white).

Engravings: Duerer, Melencolia I (1514) and Knight, Death and the Devil (1513),
public domain, from Wikimedia Commons (sources in NOTICE.md). Fetched on first run.
"""
import os, urllib.request
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 1280, 400
SRC = {
    "melencolia.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/1/18/D%C3%BCrer_Melancholia_I.jpg/1280px-D%C3%BCrer_Melancholia_I.jpg",
    "knight.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/2/28/Knight%2C_Death_and_the_Devil_MET_DP159049.jpg/1280px-Knight%2C_Death_and_the_Devil_MET_DP159049.jpg",
}
CACHE = os.path.expanduser("~/.cache/quantlag")
BAYER = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]]) / 16 + 1 / 32


def plate(name, flip_fade):
    """The whole engraving scaled to the banner height, dithered to two tones, fading toward the centre."""
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, name)
    if not os.path.exists(path):
        req = urllib.request.Request(SRC[name], headers={"User-Agent": "quantlag-banner/0.1"})
        open(path, "wb").write(urllib.request.urlopen(req).read())
    im = Image.open(path).convert("L")
    im = im.resize((int(im.width * H / im.height), H), Image.LANCZOS)
    a = np.asarray(im, dtype=float) / 255
    a = np.clip((a - 0.3) / 0.55, 0, 1) ** 1.3 * 0.95  # engraving lines dark, paper light
    w = a.shape[1]
    fade = np.clip(np.linspace(1.6, -0.2, w), 0, 1) ** 1.5
    a = a * (fade[::-1] if flip_fade else fade)[None, :]
    on = a > np.tile(BAYER, (H // 4 + 1, w // 4 + 1))[:H, :w]
    return np.where(on, 200, 0).astype(np.uint8)


def pixel_text(draw_on, text, font_path, size, scale, fill, cy):
    f = ImageFont.truetype(font_path, size)
    l, t, r, b = f.getbbox(text)
    small = Image.new("L", (r - l + 4, b - t + 4), 0)
    d = ImageDraw.Draw(small)
    d.fontmode = "1"  # no antialiasing: hard pixels
    d.text((2 - l, 2 - t), text, font=f, fill=255)
    big = small.resize((small.width * scale, small.height * scale), Image.NEAREST)
    x, y = (W - big.width) // 2, cy - big.height // 2
    draw_on.paste(Image.new("L", big.size, fill), (x, y), big)


def main():
    canvas = np.zeros((H, W), np.uint8)
    left, right = plate("melencolia.jpg", False), plate("knight.jpg", True)
    canvas[:, :left.shape[1]] = left
    canvas[:, W - right.shape[1]:] = right
    im = Image.fromarray(canvas)
    pixel_text(im, "quantlag", "docs/fonts/UnifrakturMaguntia.ttf", 44, 3, 236, 150)
    pixel_text(im, "unbiased export of rl updates", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 12, 2, 170, 262)
    pixel_text(im, "into int8 and 4-bit cpu rollouts", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 12, 2, 170, 298)
    im.save("docs/screenshots/banner.png")


if __name__ == "__main__":
    main()
