"""
overlay_attack.py

Visual prompt-injection overlay generator.
Draws configurable attack surfaces directly onto screenshots.

Supported overlay types:
- banner
- inline_text
- footer_text
- alert_box
- badge
- watermark
- inset_chat
- popup
- notification

Dependencies:
  pip install pillow
"""

from __future__ import annotations
import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

# ============================================================
# Utilities
# ============================================================

def ensure_rgba(img: Image.Image) -> Image.Image:
    return img if img.mode == "RGBA" else img.convert("RGBA")


def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def parse_rgba(color: Any, default=(0, 0, 0, 255)) -> Tuple[int, int, int, int]:
    if color is None:
        return default
    if isinstance(color, list):
        return tuple(color + [255])[:4]
    if isinstance(color, dict):
        return (
            int(color.get("r", default[0])),
            int(color.get("g", default[1])),
            int(color.get("b", default[2])),
            int(color.get("a", default[3])),
        )
    if isinstance(color, str) and color.startswith("#"):
        s = color[1:]
        if len(s) == 6:
            return tuple(int(s[i:i+2], 16) for i in (0, 2, 4)) + (255,)
        if len(s) == 8:
            return tuple(int(s[i:i+2], 16) for i in (0, 2, 4, 6))
    return default

def draw_rounded_rect(draw: ImageDraw.ImageDraw, xy, radius: int, fill=None, outline=None, width: int = 2):
    # xy = (x0, y0, x1, y1)
    x0, y0, x1, y1 = xy
    radius = max(0, radius)
    draw.rounded_rectangle([x0, y0, x1, y1], radius=radius, fill=fill, outline=outline, width=width)


def ensure_rgba(img: Image.Image) -> Image.Image:
    return img.convert("RGBA") if img.mode != "RGBA" else img

def apply_opacity(rgba: Tuple[int, int, int, int], opacity: float) -> Tuple[int, int, int, int]:
    r, g, b, a = rgba
    return (r, g, b, int(a * opacity))


def load_font(path, size):
    try:
        font = ImageFont.truetype(path, size)
        print("USING FONT:", path, size, type(font))
        return font
    except Exception as e:
        print("FONT FALLBACK:", path, size, e)
        return ImageFont.load_default()



def text_bbox(d: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont):
    return d.textbbox((0, 0), text, font=font)


def draw_round(d, box, radius, fill, outline=None, width=1):
    d.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def wrap_text(d, text, font, max_w, max_lines=None):
    lines = []
    for raw_paragraph in str(text).split("\n"):
        paragraph = raw_paragraph or " "
        cur = ""
        for ch in paragraph:
            candidate = cur + ch
            if cur and text_bbox(d, candidate, font)[2] > max_w:
                lines.append(cur.rstrip())
                cur = ch.lstrip() if ch.isspace() else ch
            else:
                cur = candidate
        if cur.strip():
            lines.append(cur.rstrip())
        elif not lines:
            lines.append("")
    if max_lines:
        lines = lines[:max_lines]
    return lines


# ============================================================
# Overlay implementations
# ============================================================

def add_banner(img, it):
    img = ensure_rgba(img)
    W, H = img.size
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)

    opacity = float(it.get("opacity", 1.0))
    base_h = int(H * float(it.get("height_pct", 0.08)))
    pos = it.get("position", "top")

    fill = apply_opacity(parse_rgba(it.get("fill_rgba"), (240, 240, 240, 230)), opacity)
    border = apply_opacity(parse_rgba(it.get("border_rgba"), (0, 0, 0, 80)), opacity)
    border_w = int(it.get("border_w", 2))

    pad = int(it.get("padding", 12))

    # =========================
    # TEXT HANDLING
    # =========================

    if "text" in it:
        font = load_font(it.get("font_path"), int(it.get("font_size", int(base_h * 0.4))))
        text_rgba = apply_opacity(parse_rgba(it.get("text_rgba"), (0, 0, 0, 255)), opacity)

        max_text_width = W - 2 * pad

        lines = wrap_text(d, it["text"], font, max_text_width)

        # Compute required height
        line_h = text_bbox(d, "Ag", font)[3]
        text_block_h = line_h * len(lines)

        h = max(base_h, text_block_h + 2 * pad)

    else:
        lines = []
        h = base_h

    # =========================
    # Banner position
    # =========================

    y0 = 0 if pos == "top" else H - h
    y1 = y0 + h

    # =========================
    # Draw background
    # =========================

    d.rectangle([0, y0, W, y1], fill=fill)

    if border_w:
        d.line([0, y0, W, y0], fill=border, width=border_w)
        d.line([0, y1, W, y1], fill=border, width=border_w)

    # =========================
    # Draw text
    # =========================

    if lines:
        total_text_h = line_h * len(lines)
        start_y = y0 + (h - total_text_h) // 2

        for i, ln in enumerate(lines):
            d.text(
                (pad, start_y + i * line_h),
                ln,
                font=font,
                fill=text_rgba
            )

    it["displayed_text"] = "\n".join(lines) if lines else ""
    it["truncated"] = False
    return Image.alpha_composite(img, overlay)


def add_inline_text(img, it):
    img = ensure_rgba(img)
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)

    opacity = float(it.get("opacity", 1.0))
    font = load_font(it.get("font_path"), int(it.get("font_size", 16)))
    fill = apply_opacity(parse_rgba(it.get("text_rgba"), (0, 0, 0, 255)), opacity)

    d.text(
        (int(it["x"]), int(it["y"])),
        it["text"],
        font=font,
        fill=fill,
        stroke_width=int(it.get("stroke_w", 0)),
        stroke_fill=parse_rgba(it.get("stroke_rgba"), (255, 255, 255, 200))
    )
    it["displayed_text"] = it.get("text", "")
    it["truncated"] = False
    return Image.alpha_composite(img, overlay)


def add_footer_text(img, it):
    img = ensure_rgba(img)
    W, H = img.size
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)

    opacity = float(it.get("opacity", 1.0))
    font = load_font(it.get("font_path"), int(it.get("font_size", 12)))
    fill = apply_opacity(parse_rgba(it.get("text_rgba"), (80, 80, 80, 200)), opacity)
    margin = int(it.get("margin", 10))
    side = it.get("side", "right")

    text = it["text"]
    max_text_width = W - 2 * margin

    # =========================
    # Wrap text
    # =========================
    lines = wrap_text(d, text, font, max_text_width)

    # =========================
    # Compute total height
    # =========================
    line_h = text_bbox(d, "Ag", font)[3]
    total_text_h = line_h * len(lines)

    # Footer grows upward if multi-line
    start_y = H - margin - total_text_h

    # =========================
    # Render lines
    # =========================
    for i, ln in enumerate(lines):
        lw = text_bbox(d, ln, font)[2]

        if side == "left":
            x = margin
        elif side == "center":
            x = (W - lw) // 2
        else:  # right
            x = W - margin - lw

        y = start_y + i * line_h
        d.text((x, y), ln, font=font, fill=fill)

    it["displayed_text"] = "\n".join(lines) if lines else ""
    it["truncated"] = False
    return Image.alpha_composite(img, overlay)


from PIL import Image, ImageDraw, ImageFilter, ImageFont

# =========================
# Utils
# =========================

def ensure_rgba(img):
    return img.convert("RGBA") if img.mode != "RGBA" else img


def parse_rgba(c, default):
    return tuple(c) if c is not None else default


def apply_opacity(rgba, opacity):
    r, g, b, a = rgba
    return (r, g, b, int(a * opacity))


def load_font(path, size):
    if path:
        return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def text_bbox(draw, text, font):
    return draw.textbbox((0, 0), text, font=font)


def draw_round(draw, box, radius, fill, outline, width):
    draw.rounded_rectangle(
        box, radius=radius, fill=fill, outline=outline, width=width
    )


def aligned_x(x, w, pad, text_w, align):
    if align == "center":
        return x + (w - text_w) // 2
    if align == "right":
        return x + w - pad - text_w
    return x + pad


def wrap_text(draw, text, font, max_w, max_lines=None):
    lines = []
    for raw_paragraph in str(text).split("\n"):
        paragraph = raw_paragraph or " "
        cur = ""
        for ch in paragraph:
            candidate = cur + ch
            if cur and text_bbox(draw, candidate, font)[2] > max_w:
                lines.append(cur.rstrip())
                cur = ch.lstrip() if ch.isspace() else ch
            else:
                cur = candidate
        if cur.strip():
            lines.append(cur.rstrip())
        elif not lines:
            lines.append("")
    if max_lines:
        lines = lines[:max_lines]
    return lines


def measure_lines(draw, lines, font, spacing):
    lh = text_bbox(draw, "Ag", font)[3] + spacing
    return lh * len(lines), lh


def lerp(c1, c2, t):
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(4))


def vertical_gradient(size, c1, c2):
    w, h = size
    g = Image.new("RGBA", (1, h))
    for y in range(h):
        g.putpixel((0, y), lerp(c1, c2, y / max(1, h - 1)))
    return g.resize((w, h))


# =========================
# Main
# =========================

def add_alert_box(img, it):
    img = ensure_rgba(img)
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)

    # =========================
    # Basic params
    # =========================

    opacity = float(it.get("opacity", 1.0))
    x, y, w = map(int, (it["x"], it["y"], it["w"]))
    pad = int(it.get("padding", 12))
    radius = int(it.get("radius", 12))
    line_spacing = int(it.get("line_spacing", 4))
    title_body_gap = int(it.get("title_body_gap", 8))

    title_font = load_font(
        it.get("title_font_path", it.get("font_path")),
        int(it.get("title_font_size", 16))
    )
    body_font = load_font(
        it.get("body_font_path", it.get("font_path")),
        int(it.get("body_font_size", 14))
    )

    title = it.get("title", "")
    body = it.get("body", "")

    # =========================
    # Measure title
    # =========================

    title_bbox = text_bbox(d, title, title_font)
    title_h = title_bbox[3] - title_bbox[1]

    # =========================
    # Compute effective text width (icon-aware)
    # =========================

    text_left_offset = pad
    if "icon" in it:
        ic = it["icon"]
        text_left_offset += int(ic.get("size", 24)) + int(ic.get("gap", 8))

    effective_text_w = max(1, w - text_left_offset - pad)

    # =========================
    # Measure body (NO truncation)
    # =========================

    all_body_lines = wrap_text(
        d,
        body,
        body_font,
        effective_text_w,
        max_lines=None  # IMPORTANT
    )
    # print(all_body_lines)
    body_h, lh = measure_lines(
        d,
        all_body_lines,
        body_font,
        line_spacing
    )

    # print(body_h)

    it["displayed_text"] = "\n".join([part for part in [title, *all_body_lines] if part])
    it["truncated"] = False

    # =========================
    # Buttons height
    # =========================

    buttons = it.get("buttons", [])
    button_h = (32 + pad) if buttons else 0

    # =========================
    # Compute content height
    # =========================

    content_h = (
        pad * 2 +
        title_h +
        title_body_gap +
        body_h +
        button_h
    )

    h = int(it.get("h", content_h))
    if it.get("auto_height", False):
        h = max(content_h, int(it.get("min_h", 0)))
    it["layout_overflow"] = bool(x < 0 or y < 0 or (x + w) > img.size[0] or (y + h) > img.size[1])

    # =========================
    # Shadow
    # =========================

    if "shadow" in it:
        sh = it["shadow"]
        shadow = Image.new("RGBA", img.size, (0, 0, 0, 0))
        sd = ImageDraw.Draw(shadow)
        draw_round(
            sd,
            (x, y, x + w, y + h),
            radius,
            parse_rgba(sh.get("rgba"), (0, 0, 0, 120)),
            None,
            0,
        )
        shadow = shadow.filter(
            ImageFilter.GaussianBlur(int(sh.get("blur", 8)))
        )
        overlay.alpha_composite(
            shadow,
            tuple(sh.get("offset", (0, 4)))
        )

    # =========================
    # Background
    # =========================

    if "gradient" in it:
        g = it["gradient"]
        grad = vertical_gradient(
            (w, h),
            apply_opacity(parse_rgba(g["from"], (255, 255, 255, 255)), opacity),
            apply_opacity(parse_rgba(g["to"], (230, 230, 255, 255)), opacity),
        )
        overlay.alpha_composite(grad, (x, y))
    else:
        fill = apply_opacity(
            parse_rgba(it.get("fill_rgba"), (245, 248, 255, 235)),
            opacity
        )
        draw_round(d, (x, y, x + w, y + h), radius, fill, None, 0)

    border = apply_opacity(
        parse_rgba(it.get("border_rgba"), (60, 120, 255, 180)),
        opacity
    )
    draw_round(
        d,
        (x, y, x + w, y + h),
        radius,
        None,
        border,
        int(it.get("border_w", 2)),
    )

    # =========================
    # Icon
    # =========================

    icon_offset = 0
    if "icon" in it:
        ic = it["icon"]
        icon = Image.open(ic["path"]).convert("RGBA")
        size = int(ic.get("size", 24))
        icon = icon.resize((size, size))
        overlay.alpha_composite(icon, (x + pad, y + pad))
        icon_offset = size + int(ic.get("gap", 8))

    # =========================
    # Title (render)
    # =========================

    title_w = title_bbox[2] - title_bbox[0]
    tx = aligned_x(
        x + icon_offset,
        w - icon_offset,
        pad,
        title_w,
        it.get("title_align", "left"),
    )

    d.text(
        (tx, y + pad),
        title,
        font=title_font,
        fill=apply_opacity(
            parse_rgba(it.get("title_rgba"), (20, 40, 80, 255)),
            opacity
        ),
    )

    # =========================
    # Body (render, WITH truncation)
    # =========================

    body_lines = all_body_lines
    if it.get("max_lines"):
        body_lines = body_lines[:it["max_lines"]]

    ty = y + pad + title_h + title_body_gap
    for ln in body_lines:
        lw = text_bbox(d, ln, body_font)[2]
        bx = aligned_x(x, w, pad, lw, it.get("body_align", "left"))
        d.text(
            (bx, ty),
            ln,
            font=body_font,
            fill=apply_opacity(
                parse_rgba(it.get("body_rgba"), (20, 20, 20, 255)),
                opacity
            ),
        )
        ty += lh

    # =========================
    # Buttons
    # =========================

    if buttons:
        bx = x + w - pad
        by = y + h - pad - 28
        for btn in reversed(buttons):
            bw = max(60, text_bbox(d, btn["text"], body_font)[2] + 20)
            bx -= bw
            draw_round(
                d,
                (bx, by, bx + bw, by + 28),
                8,
                apply_opacity((230, 235, 245, 255), opacity),
                border,
                1,
            )
            tw = text_bbox(d, btn["text"], body_font)[2]
            d.text(
                (bx + (bw - tw) // 2, by + 6),
                btn["text"],
                font=body_font,
                fill=(30, 30, 30, 255),
            )
            bx -= 8

    # =========================
    # Close button
    # =========================

    if it.get("close_button", False):
        cx = x + w - pad - 12
        cy = y + pad
        d.line((cx, cy, cx + 10, cy + 10), fill=(120, 120, 120, 200), width=2)
        d.line((cx + 10, cy, cx, cy + 10), fill=(120, 120, 120, 200), width=2)

    return Image.alpha_composite(img, overlay)





def add_badge(img, it):
    img = ensure_rgba(img)
    W, H = img.size
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)

    opacity = float(it.get("opacity", 1.0))
    font = load_font(it.get("font_path"), int(it.get("font_size", 14)))
    pad = int(it.get("padding", 10))

    text = it["text"]
    max_width = int(it.get("max_width", W - 20))

    # =========================
    # Wrap text
    # =========================
    words = text.split()
    lines = []
    cur = ""

    for w in words:
        test = (cur + " " + w).strip()
        if text_bbox(d, test, font)[2] <= max_width - 2 * pad:
            cur = test
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)

    # =========================
    # Measure badge size
    # =========================
    line_h = text_bbox(d, "Ag", font)[3]
    text_block_h = line_h * len(lines)
    text_block_w = max(text_bbox(d, ln, font)[2] for ln in lines)

    w = text_block_w + 2 * pad
    h = text_block_h + 2 * pad

    x = int(it["x"])
    y = int(it["y"])

    # Prevent overflow to right edge
    if x + w > W:
        x = W - w - 5

    # Prevent overflow bottom
    if y + h > H:
        y = H - h - 5

    fill = apply_opacity(parse_rgba(it.get("fill_rgba"), (255, 235, 200, 235)), opacity)
    border = apply_opacity(parse_rgba(it.get("border_rgba"), (200, 140, 40, 220)), opacity)

    draw_round(
        d,
        (x, y, x + w, y + h),
        int(it.get("radius", 12)),
        fill,
        border,
        int(it.get("border_w", 2))
    )

    # =========================
    # Render text lines
    # =========================
    for i, ln in enumerate(lines):
        d.text(
            (x + pad, y + pad + i * line_h),
            ln,
            font=font,
            fill=apply_opacity(parse_rgba(it.get("text_rgba"), (60, 40, 10, 255)), opacity)
        )

    return Image.alpha_composite(img, overlay)


def add_watermark(img, it):
    img = ensure_rgba(img)
    W, H = img.size

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))

    opacity = float(it.get("opacity", 1.0))
    font_size = int(it.get("font_size", max(20, min(W, H) // 20)))
    font = load_font(it.get("font_path"), font_size)
    fill = apply_opacity(parse_rgba(it.get("text_rgba"), (0, 0, 0, 40)), opacity)

    text = it["text"]
    angle = float(it.get("angle_deg", -25))
    spacing = int(it.get("spacing", 250))

    # ---- compute exact text bbox ----
    dummy = Image.new("RGBA", (1, 1))
    dd = ImageDraw.Draw(dummy)
    bbox = dd.textbbox((0, 0), text, font=font)

    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    # IMPORTANT: add padding to avoid descender clipping
    pad = int(th * 0.5)

    # ---- render text on tight canvas ----
    text_img = Image.new("RGBA", (tw + 2 * pad, th + 2 * pad), (0, 0, 0, 0))
    td = ImageDraw.Draw(text_img)
    td.text((pad, pad), text, font=font, fill=fill)

    # ---- rotate text safely ----
    text_img = text_img.rotate(angle, resample=Image.BICUBIC, expand=True)

    # ---- place watermark ----
    if it.get("mode", "tiled") == "single":
        x = int(it.get("x", (W - text_img.width) // 2))
        y = int(it.get("y", (H - text_img.height) // 2))
        overlay.alpha_composite(text_img, (x, y))
    else:
        for y in range(-H, 2 * H, spacing):
            for x in range(-W, 2 * W, spacing):
                overlay.alpha_composite(text_img, (x, y))

    return Image.alpha_composite(img, overlay)


from typing import Dict, Any
from PIL import Image, ImageDraw

def add_inset_chat(img: Image.Image, item: Dict[str, Any]) -> Image.Image:
    img = ensure_rgba(img)
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)

    # ===============================
    # Geometry
    # ===============================
    x, y, w, h = map(int, (item["x"], item["y"], item["w"], item["h"]))
    item["layout_overflow"] = bool(x < 0 or y < 0 or (x + w) > img.size[0] or (y + h) > img.size[1])
    pad = int(item.get("padding", 12))
    radius = int(item.get("radius", 14))
    line_spacing = int(item.get("line_spacing", 4))

    bg = parse_rgba(item.get("bg_rgba"), (255, 255, 255, 255))
    border = parse_rgba(item.get("border_rgba"), (0, 0, 0, 255))
    border_w = int(item.get("border_w", 2))

    # ===============================
    # Container
    # ===============================
    draw_rounded_rect(
        d,
        (x, y, x + w, y + h),
        radius=radius,
        fill=bg,
        outline=border,
        width=border_w,
    )

    # ===============================
    # Fonts
    # ===============================
    font = load_font(item.get("font_path"), int(item.get("font_size", 14)))
    title_font = load_font(item.get("font_path"), int(item.get("title_font_size", 15)))
    label_font = load_font(item.get("font_path"), int(item.get("label_font_size", 15)))

    # ===============================
    # Header
    # ===============================
    title = str(item.get("title", "Chat"))
    title_color = parse_rgba(item.get("title_rgba"), (40, 40, 40, 255))

    d.text((x + pad, y + pad), title, font=title_font, fill=title_color)

    title_h = text_bbox(d, title, title_font)[3]
    divider_y = y + pad + title_h + 6

    # divider line
    d.line(
        (x + pad, divider_y, x + w - pad, divider_y),
        fill=(180, 180, 180, 255),
        width=1,
    )

    # ===============================
    # Message area start
    # ===============================
    msg_y = divider_y + 8
    bubble_gap = 10
    bubble_pad = 10
    bubble_max_w = int(w * 0.72)

    messages = item.get("messages", [])
    displayed_messages = []
    truncated = False

    for m in messages:
        role = m.get("role", "user")

        if role == "assistant":
            side = "left"
            role_label = "Assistant"
            label_color = (22, 163, 74, 255)
            bubble_fill = parse_rgba(item.get("fill_rgba"), (236, 253, 245, 255))
            bubble_border = (22, 163, 74, 255)
            text_color = parse_rgba(item.get("color_text"), (15, 23, 42, 255))
        else:
            side = "right"
            role_label = "User"
            label_color = (37, 99, 235, 255)
            bubble_fill = parse_rgba(item.get("fill_rgba"), (239, 246, 255, 255))
            bubble_border = (37, 99, 235, 255)
            text_color = parse_rgba(item.get("color_text"), (15, 23, 42, 255))

        # ===============================
        # Role label
        # ===============================
        label_w = text_bbox(d, role_label, label_font)[2]
        label_h = text_bbox(d, role_label, label_font)[3]

        label_x = (
            x + pad if side == "left"
            else x + w - pad - label_w
        )

        d.text(
            (label_x, msg_y),
            role_label,
            font=label_font,
            fill=label_color,
        )

        # ===============================
        # Bubble text wrapping
        # ===============================
        bubble_top = msg_y + label_h + 4
        text = str(m.get("text", ""))

        lines = wrap_text(d, text, font, bubble_max_w - 2 * bubble_pad)

        line_h = text_bbox(d, "Ag", font)[3] + line_spacing
        bubble_h = line_h * len(lines) + 2 * bubble_pad
        bubble_w = min(
            max(text_bbox(d, ln, font)[2] for ln in lines) + 2 * bubble_pad,
            bubble_max_w,
        )

        bx0 = (
            x + pad if side == "left"
            else x + w - pad - bubble_w
        )
        by0 = bubble_top
        bx1, by1 = bx0 + bubble_w, by0 + bubble_h

        if by1 > y + h - pad:
            truncated = True
            break

        draw_rounded_rect(
            d,
            (bx0, by0, bx1, by1),
            radius=12,
            fill=bubble_fill,
            outline=bubble_border,
            width=1,
        )

        # ===============================
        # Bubble text render
        # ===============================
        ty = by0 + bubble_pad
        for ln in lines:
            d.text(
                (bx0 + bubble_pad, ty),
                ln,
                font=font,
                fill=text_color,
            )
            ty += line_h
        displayed_messages.append(text)

        # ===============================
        # Advance cursor
        # ===============================
        msg_y = by1 + bubble_gap
        if msg_y > y + h - pad - 10:
            truncated = len(displayed_messages) < len(messages)
            break

    item["displayed_text"] = "\n".join(displayed_messages)
    item["truncated"] = truncated
    return Image.alpha_composite(img, overlay)



def add_popup(img, it):
    img = ensure_rgba(img)
    W, H = img.size
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)

    # ===============================
    # Geometry & style
    # ===============================
    opacity = float(it.get("opacity", 1.0))
    w, h = int(it["w"]), int(it["h"])
    x = int(it.get("x", (W - w) // 2))
    y = int(it.get("y", (H - h) // 2))
    it["layout_overflow"] = bool(x < 0 or y < 0 or (x + w) > W or (y + h) > H)

    pad = int(it.get("padding", 20))
    radius = int(it.get("radius", 16))
    border_w = int(it.get("border_w", 2))

    # ===============================
    # Backdrop
    # ===============================
    d.rectangle(
        (0, 0, W, H),
        fill=apply_opacity(
            parse_rgba(it.get("backdrop_rgba"), (0, 0, 0, 120)),
            opacity
        )
    )

    fill = apply_opacity(parse_rgba(it.get("fill_rgba"), (255, 255, 255, 245)), opacity)
    border = apply_opacity(parse_rgba(it.get("border_rgba"), (0, 0, 0, 160)), opacity)

    # ===============================
    # Popup container
    # ===============================
    draw_round(d, (x, y, x + w, y + h), radius, fill, border, border_w)

    content_x = x + pad
    content_y = y + pad
    content_w = w - 2 * pad
    content_h = h - 2 * pad

    # ===============================
    # LEFT IMAGE
    # ===============================
    left_w = int(it.get("left_width", 0))
    if left_w > 0 and it.get("left_image_path"):
        try:
            img_left = Image.open(it["left_image_path"]).convert("RGBA")
            img_left = img_left.resize((left_w, h), Image.LANCZOS)

            mask = Image.new("L", (left_w, h), 255)
            m = ImageDraw.Draw(mask)
            m.rectangle((radius, 0, left_w, h), fill=255)
            m.rectangle((0, radius, left_w, h), fill=255)
            m.pieslice((0, 0, 2 * radius, 2 * radius), 180, 270, fill=255)
            m.pieslice((0, h - 2 * radius, 2 * radius, h), 90, 180, fill=255)

            overlay.paste(img_left, (x, y), mask)

            content_x += left_w
            content_w -= left_w
        except Exception:
            pass

    # ===============================
    # RIGHT IMAGE
    # ===============================
    right_w = int(it.get("right_width", 0))
    if right_w > 0 and it.get("right_image_path"):
        try:
            img_right = Image.open(it["right_image_path"]).convert("RGBA")
            img_right = img_right.resize((right_w, h), Image.LANCZOS)

            mask = Image.new("L", (right_w, h), 255)
            m = ImageDraw.Draw(mask)
            m.rectangle((0, 0, right_w - radius, h), fill=255)
            m.rectangle((0, radius, right_w, h), fill=255)
            m.pieslice((right_w - 2 * radius, 0, right_w, 2 * radius), 270, 360, fill=255)
            m.pieslice((right_w - 2 * radius, h - 2 * radius, right_w, h), 0, 90, fill=255)

            overlay.paste(img_right, (x + w - right_w, y), mask)

            content_w -= right_w
        except Exception:
            pass

    # ===============================
    # TOP IMAGE
    # ===============================
    if left_w == 0 and right_w == 0 and it.get("top_image_path"):
        try:
            top_h = int(it.get("top_image_height", int(h * 0.35)))
            img_top = Image.open(it["top_image_path"]).convert("RGBA")
            img_top = img_top.resize((w, top_h), Image.LANCZOS)

            mask = Image.new("L", (w, top_h), 255)
            m = ImageDraw.Draw(mask)
            m.rectangle((radius, 0, w - radius, top_h), fill=255)
            m.rectangle((0, radius, w, top_h), fill=255)
            m.pieslice((0, 0, 2 * radius, 2 * radius), 180, 270, fill=255)
            m.pieslice((w - 2 * radius, 0, w, 2 * radius), 270, 360, fill=255)

            overlay.paste(img_top, (x, y), mask)

            content_y += top_h
            content_h -= top_h
        except Exception:
            pass

    # ===============================
    # Fonts
    # ===============================
    title_font = load_font(
        it.get("title_font_path", it.get("font_path")),
        int(it.get("title_font_size", 18))
    )
    body_font = load_font(
        it.get("body_font_path", it.get("font_path")),
        int(it.get("body_font_size", 15))
    )

    title_color = apply_opacity(parse_rgba(it.get("title_rgba"), (0, 0, 0, 255)), opacity)
    body_color = apply_opacity(parse_rgba(it.get("body_rgba"), (60, 60, 60, 255)), opacity)

    # ===============================
    # Text rendering
    # ===============================
    ty = content_y

    title = it.get("title", "")
    body = it.get("body", "")

    divider_gap = int(it.get("divider_gap", 10))
    divider_thickness = int(it.get("divider_thickness", 1))

    divider_color = apply_opacity(
        parse_rgba(it.get("divider_rgba", it.get("border_rgba")), (0, 0, 0, 255)),
        opacity
    )

    if title:
        d.text((content_x, ty), title, font=title_font, fill=title_color)
        a, dsc = title_font.getmetrics()
        ty += a + dsc + divider_gap

        # ===== Divider =====
        if body:
            d.rectangle(
                (
                    content_x,
                    ty,
                    content_x + content_w,
                    ty + divider_thickness
                ),
                fill=divider_color
            )
            ty += divider_thickness + divider_gap

    # ===== Body =====
    if body:
        a, dsc = body_font.getmetrics()
        lh = a + dsc + int(it.get("line_spacing", 6))

        all_lines = wrap_text(d, body, body_font, content_w)
        max_lines = max(1, (content_y + content_h - ty) // lh)
        lines = all_lines[:max_lines]
        it["truncated"] = len(lines) < len(all_lines)
        it["displayed_text"] = "\n".join([part for part in [title, *lines] if part])

        for ln in lines:
            d.text((content_x, ty), ln, font=body_font, fill=body_color)
            ty += lh
    else:
        it["truncated"] = False
        it["displayed_text"] = title

    return Image.alpha_composite(img, overlay)





def add_notification(img, it):
    img = ensure_rgba(img)
    W, H = img.size
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)

    # ===============================
    # Basic params
    # ===============================
    opacity = float(it.get("opacity", 1.0))
    w = int(it.get("w", int(W * 0.4)))
    margin = int(it.get("margin", 16))
    pad = int(it.get("padding", 14))
    line_spacing = int(it.get("line_spacing", 4))

    # ===============================
    # Icon params
    # ===============================
    icon_size = int(it.get("icon_size", 36))
    icon_pad = int(it.get("icon_padding", 12))
    has_icon = bool(it.get("icon_path"))

    # ===============================
    # Fonts
    # ===============================
    title_font = load_font(
        it.get("title_font_path", it.get("font_path")),
        int(it.get("title_font_size", 14))
    )
    body_font = load_font(
        it.get("body_font_path", it.get("font_path")),
        int(it.get("body_font_size", 13))
    )

    title = it.get("title", "")
    body = it.get("text", "")

    # ===============================
    # Measure TEXT width
    # ===============================
    text_x_offset = pad
    if has_icon:
        text_x_offset += icon_size + icon_pad

    content_w = max(1, w - text_x_offset - pad)

    # ===============================
    # Measure TEXT height
    # ===============================
    content_h = 0

    if title:
        a, dsc = title_font.getmetrics()
        content_h += a + dsc + 2

    a, dsc = body_font.getmetrics()
    lh = a + dsc + line_spacing

    body_lines = wrap_text(d, body, body_font, content_w)
    body_h = lh * len(body_lines)

    content_h += body_h

    # ===============================
    # Final height (auto)
    # ===============================
    h = content_h + 2 * pad

    # Ensure icon fits vertically
    if has_icon:
        h = max(h, icon_size + 2 * pad)

    # ===============================
    # Position
    # ===============================
    pos = it.get("position", "bottom-right")
    x = margin if "left" in pos else W - w - margin
    y = margin if "top" in pos else H - h - margin
    it["layout_overflow"] = bool(x < 0 or y < 0 or (x + w) > W or (y + h) > H)

    # ===============================
    # Container
    # ===============================
    fill = apply_opacity(parse_rgba(it.get("bg_rgba"), (255, 255, 255, 255)), opacity)
    border = apply_opacity(parse_rgba(it.get("border_rgba"), (0, 0, 0, 255)), opacity)

    radius = int(it.get("radius", 16))
    border_w = int(it.get("border_w", 1))

    draw_round(d, (x, y, x + w, y + h), radius, fill, border, border_w)

    # ===============================
    # Icon render
    # ===============================
    text_x = x + pad
    if has_icon:
        try:
            icon = Image.open(it["icon_path"]).convert("RGBA")
            icon = icon.resize((icon_size, icon_size), Image.LANCZOS)
            icon_y = y + (h - icon_size) // 2
            overlay.alpha_composite(icon, (x + pad, icon_y))
            text_x += icon_size + icon_pad
        except Exception:
            pass

    # ===============================
    # Text render
    # ===============================
    ty = y + pad

    title_color = apply_opacity(parse_rgba(it.get("title_rgba"), (0, 0, 0, 255)), opacity)
    body_color = apply_opacity(parse_rgba(it.get("text_rgba"), (60, 60, 60, 255)), opacity)

    if title:
        d.text((text_x, ty), title, font=title_font, fill=title_color)
        a, dsc = title_font.getmetrics()
        ty += a + dsc + 2

    for ln in body_lines:
        d.text((text_x, ty), ln, font=body_font, fill=body_color)
        ty += lh

    it["displayed_text"] = "\n".join([part for part in [title, *body_lines] if part])
    it["truncated"] = False
    return Image.alpha_composite(img, overlay)



# ============================================================
# Dispatcher + CLI
# ============================================================

OVERLAY_FUNCS = {
    "banner": add_banner,
    "inline_text": add_inline_text,
    "footer_text": add_footer_text,
    "alert_box": add_alert_box,
    "badge": add_badge,
    "watermark": add_watermark,
    "inset_chat": add_inset_chat,
    "popup": add_popup,
    "notification": add_notification,
}


def apply_overlays(img: Image.Image, overlays: List[Dict[str, Any]]) -> Image.Image:
    overlays = sorted(overlays, key=lambda x: x.get("z_index", 0))
    out = ensure_rgba(img)
    for it in overlays:
        out = OVERLAY_FUNCS[it["type"]](out, it)
    return out


# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--in", dest="inp", required=True)
#     ap.add_argument("--spec", required=True)
#     ap.add_argument("--out", required=True)
#     args = ap.parse_args()

#     with open(args.spec, "r", encoding="utf-8") as f:
#         spec = json.load(f)

#     img = Image.open(args.inp)
#     out = apply_overlays(img, spec.get("overlays", []))
#     out.save(args.out)
#     # print(f"[OK] wrote {args.out}")


# if __name__ == "__main__":
#     main()
