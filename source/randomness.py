import os
import random


# =========================================================
# COLOR PALETTES (SEPARATED & UI-SAFE)
# =========================================================

TEXT_COLOR_PALETTE = [
    # Neutral dark
    (17, 24, 39),
    (31, 41, 55),
    (55, 65, 81),
    (75, 85, 99),
    (100, 116, 139),
    (148, 163, 184),

    # Slate / Gray
    (33, 37, 41),
    (52, 58, 64),
    (73, 80, 87),
    (108, 117, 125),

    # Blue
    (30, 64, 175),
    (29, 78, 216),
    (37, 99, 235),
    (18, 97, 160),
    (3, 105, 161),

    # Green
    (21, 128, 61),
    (22, 101, 52),
    (20, 83, 45),
    (6, 95, 70),

    # Yellow / Orange (dark tone)
    (133, 77, 14),
    (146, 64, 14),
    (154, 52, 18),

    # Red
    (153, 27, 27),
    (127, 29, 29),
    (136, 19, 55),

    # Purple
    (88, 28, 135),
    (107, 33, 168),
    (126, 34, 206),

    # Indigo
    (49, 46, 129),
    (55, 48, 163),
    (67, 56, 202),
]


# =========================================================
# FILL COLORS – soft UI backgrounds (light / pastel)
# =========================================================

FILL_COLOR_PALETTE = [
    # Neutral
    (248, 249, 250),
    (241, 243, 245),
    (233, 236, 239),
    (222, 226, 230),
    (250, 250, 250),
    (245, 245, 245),

    # Cool gray
    (243, 244, 246),
    (229, 231, 235),
    (209, 213, 219),

    # Blue
    (231, 245, 255),
    (224, 242, 254),
    (219, 234, 254),
    (239, 246, 255),

    # Cyan
    (236, 254, 255),
    (207, 250, 254),

    # Green
    (232, 249, 243),
    (237, 247, 237),
    (220, 252, 231),
    (240, 253, 244),

    # Yellow
    (255, 244, 229),
    (255, 239, 213),
    (254, 249, 195),

    # Orange
    (255, 237, 213),
    (255, 247, 237),

    # Red
    (253, 232, 232),
    (254, 242, 242),

    # Purple / Indigo
    (243, 232, 255),
    (237, 233, 254),
    (245, 243, 255),
]


# =========================================================
# BORDER COLORS – neutral, subtle
# =========================================================

BORDER_COLOR_PALETTE = [
    (220, 220, 220),
    (210, 210, 210),
    (200, 200, 200),
    (190, 190, 190),
    (180, 180, 180),
    (170, 170, 170),
    (160, 160, 160),
    (150, 150, 150),
    (140, 140, 140),
    (130, 130, 130),

    # Slightly tinted (still safe)
    (203, 213, 225),  # slate
    (209, 213, 219),
    (186, 230, 253),  # blue light
    (187, 247, 208),  # green light
    (254, 215, 170),  # orange light
]


# =========================================================
# UTILS
# =========================================================

def random_file_from_dir(folder, exts=(".ttf", ".otf"), fallback=None):
    if not os.path.isdir(folder):
        return fallback
    files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(exts)
    ]
    return random.choice(files) if files else fallback


def rint(a, b):
    return random.randint(a, b)


def rfloat(a, b, nd=2):
    return round(random.uniform(a, b), nd)


def clamp_rand(min_v, max_v):
    if max_v <= min_v:
        return min_v
    return random.randint(min_v, max_v)


def rand_text_rgba(a_min=255, a_max=255):
    r, g, b = random.choice(TEXT_COLOR_PALETTE)
    return [r, g, b, random.randint(a_min, a_max)]


def rand_fill_rgba(a_min=255, a_max=255):
    r, g, b = random.choice(FILL_COLOR_PALETTE)
    return [r, g, b, random.randint(a_min, a_max)]


def rand_border_rgba(a_min=255, a_max=255):
    r, g, b = random.choice(BORDER_COLOR_PALETTE)
    return [r, g, b, random.randint(a_min, a_max)]


# =========================================================
# ALERT BOX
# =========================================================

def random_alert_box(img_w, img_h):
    w = rint(400, min(520, int(img_w * 0.8)))
    min_h = rint(160, 220)

    margin = int(min(img_w, img_h) * 0.04)
    x = clamp_rand(margin, img_w - w - margin)
    y = clamp_rand(margin, img_h - min_h - margin)

    box = {
        "type": "alert_box",
        "z_index": rint(20, 60),
        "opacity": rfloat(1.0, 1.0),

        "x": x,
        "y": y,
        "w": w,

        "auto_height": True,
        "min_h": min_h,

        "title": "",
        "body": "",

        "font_path": random_file_from_dir("assets/fonts", (".ttf", ".otf")),

        "title_font_size": rint(17, 18),
        "title_align": random.choice(["left", "center"]),
        "title_rgba": rand_text_rgba(),

        "body_font_size": rint(15, 17),
        "body_align": "left",
        "body_rgba": rand_text_rgba(),

        "fill_rgba": rand_fill_rgba(),
        "border_rgba": rand_border_rgba(),
        "border_w": rint(1, 3),

        "radius": rint(10, 18),
        "padding": rint(12, 18),
        "line_spacing": rint(6, 8),
        "title_body_gap": rint(11, 13),

        "close_button": random.choice([True, False])
    }

    if random.random() < 1.0:
        icon = random_file_from_dir("assets/alert_icons", (".png", ".jpg", ".webp"))
        if icon:
            box["icon"] = {"path": icon, "size": rint(20, 28), "gap": rint(8, 14)}

    if random.random() < 0.8:
        box["shadow"] = {
            "offset": [rint(-2, 2), rint(4, 8)],
            "blur": rint(10, 18),
            "rgba": [0, 0, 0, rint(90, 140)]
        }

    return box


# =========================================================
# BADGE
# =========================================================


def random_badge(img_w, img_h):
    font_size = rint(15, 17)
    padding = rint(8, 14)

    est_w = font_size * 6 + padding * 2
    est_h = font_size + padding * 2

    margin = int(min(img_w, img_h) * 0.4)
    x = clamp_rand(margin, img_w - est_w - margin)
    y = clamp_rand(margin, img_h - est_h - margin)

    return {
        "type": "badge",
        "z_index": rint(20, 50),
        "opacity": rfloat(1.0, 1.0),

        "x": x,
        "y": y,

        "text": "",
        "font_path": random_file_from_dir("assets/fonts", (".ttf", ".otf")),
        "font_size": font_size,

        "text_rgba": rand_text_rgba(),
        "fill_rgba": rand_fill_rgba(),
        "border_rgba": rand_border_rgba(),
        "border_w": rint(1, 3),

        "padding": padding,
        "radius": rint(8, 16)
    }


# =========================================================
# BANNER
# =========================================================

def random_banner(img_w, img_h):
    return {
        "type": "banner",
        "z_index": rint(5, 20),
        "opacity": rfloat(1.0, 1.0),

        "position": random.choice(["top", "bottom"]),
        "height_pct": rfloat(0.05, 0.12),

        "text": "",
        "font_path": random_file_from_dir("assets/fonts", (".ttf", ".otf")),
        "font_size": rint(15, 16),

        "text_rgba": rand_text_rgba(),
        "fill_rgba": rand_fill_rgba(),
        "border_rgba": rand_border_rgba(),
        "border_w": rint(0, 3),

        "padding": rint(10, 20)
    }


# =========================================================
# FOOTER TEXT
# =========================================================

def random_footer_text(img_w, img_h):
    return {
        "type": "footer_text",
        "z_index": rint(1, 10),
        "opacity": rfloat(1.0, 1.0),

        "text": "",
        "side": random.choice(["left", "right"]),
        "margin": rint(8, 20),
        "border_rgba": rand_border_rgba(),
        "fill_rgba": rand_fill_rgba(),
        "font_path": random_file_from_dir("assets/fonts", (".ttf", ".otf")),
        "font_size": rint(15, 16),

        "text_rgba": rand_text_rgba(160, 220)
    }


# =========================================================
# INLINE TEXT
# =========================================================

def random_inline_text(img_w, img_h):
    font_size = rint(15, 17)

    # text placeholder tạm để ước lượng chiều dài
    approx_chars = random.randint(20, 40)

    # ---- estimate bounding box ----
    est_w = int(font_size * 0.6 * approx_chars)
    est_h = int(font_size * 1.4)

    margin = int(min(img_w, img_h) * 0.03)

    # ---- clamp để text không vượt ảnh ----
    x = clamp_rand(
        margin,
        max(margin, img_w - est_w - margin)
    )
    y = clamp_rand(
        margin,
        max(margin, img_h - est_h - margin)
    )

    inline = {
        "type": "inline_text",
        "z_index": rint(15, 35),
        "opacity": rfloat(1.0, 1.0),

        "x": x,
        "y": y,

        "text": "",

        "font_path": random_file_from_dir("assets/fonts", (".ttf", ".otf")),
        "font_size": font_size,

        "text_rgba": rand_text_rgba()
    }

    # # stroke optional nhưng an toàn hơn
    # if random.random() < 0.7:
    #     inline["stroke_w"] = rint(1, 2)
    #     inline["stroke_rgba"] = rand_border_rgba(180, 240)

    return inline



# =========================================================
# INSET CHAT
# =========================================================

def random_inset_chat(img_w, img_h):
    w = rint(400, min(560, int(img_w * 0.6)))
    h = rint(300, min(320, int(img_h * 0.4)))

    margin = int(min(img_w, img_h) * 0.04)
    x = clamp_rand(margin, img_w - w - margin)
    y = clamp_rand(margin, img_h - h - margin)

    return {
        "type": "inset_chat",
        "z_index": 999,
        "opacity": rfloat(1.0, 1.0),

        "x": x,
        "y": y,
        "w": w,
        "h": h,

        "title": random.choice(["Support Chat", "Live Assistance", "Help Desk", "System Chat"]),
        "font_path": random_file_from_dir("assets/fonts", (".ttf", ".otf")),

        "font_size": rint(15, 16),
        "title_font_size": rint(16, 18),

        "title_rgba": rand_text_rgba(),
        "bg_rgba": rand_fill_rgba(),
        "fill_rgba": rand_fill_rgba(),
        "border_rgba": rand_border_rgba(),
        "border_w": rint(1, 3),
        "color_text": rand_text_rgba(),

        "radius": rint(10, 18),
        "padding": rint(8, 10),
        "line_spacing": rint(3, 6),

        "messages": [
            {"role": "assistant", "text": "", "fill_rgba": rand_fill_rgba()},
            # {"role": "agent", "text": ""}
        ]
    }


# =========================================================
# NOTIFICATION
# =========================================================

def random_notification(img_w, img_h):
    notif = {
        "type": "notification",
        "z_index": rint(15, 35),
        "opacity": rfloat(1.0, 1.0),

        "position": random.choice(["top-left", "top-right", "bottom-left", "bottom-right"]),
        "w": rint(400, 500),

        "title": "",
        "text": "",

        "font_path": random_file_from_dir("assets/fonts", (".ttf", ".otf")),
        "title_font_size": rint(17, 18),
        "body_font_size": rint(15, 17),

        "bg_rgba": rand_fill_rgba(),
        "border_rgba": rand_border_rgba(),
        "title_rgba": rand_text_rgba(),
        "text_rgba": rand_text_rgba(),

        "radius": rint(12, 20),
        "padding": rint(10, 16),
        "line_spacing": rint(3, 6),
        "margin": rint(10, 20)
    }

    if random.random() < 1.0:
        icon = random_file_from_dir("assets/notification_icons", (".png", ".jpg", ".webp"))
        if icon:
            notif["icon_path"] = icon
            notif["icon_size"] = rint(28, 40)

    return notif


# =========================================================
# POPUP
# =========================================================

def random_file_from_dir(folder, exts=(".png", ".jpg", ".jpeg", ".webp")):
    if not os.path.isdir(folder):
        return None
    files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(exts)
    ]
    return random.choice(files) if files else None



def random_popup(img_w: int, img_h: int):
    # ===============================
    # Base geometry
    # ===============================
    w = rint(int(img_w * 0.3), int(img_w * 0.35))
    h = rint(int(img_h * 0.3), int(img_h * 0.45))

    popup = {
        "type": "popup",
        "z_index": random.randint(30, 60),
        "opacity": round(random.uniform(1.0, 1.0), 2),

        "w": w,
        "h": h,

        # semantic text → fill later
        "title": "",
        "body": "",

        "font_path": random_file_from_dir("assets/fonts", (".ttf", ".otf")),
        "title_font_size": random.randint(16, 18),
        "body_font_size": random.randint(15, 16),

        "padding": random.randint(16, 24),
        "line_spacing": random.randint(4, 8),

        "radius": random.randint(12, 20),
        "border_w": random.randint(1, 3),

        # colors
        "backdrop_rgba": [0, 0, 0, random.randint(100, 160)],
        "fill_rgba": rand_fill_rgba(),
        "border_rgba": rand_border_rgba(),
        "divider_rgba": rand_border_rgba(),
        "title_rgba": rand_text_rgba(),
        "body_rgba": rand_text_rgba(),
    }

    # ===============================
    # Image layout (EXCLUSIVE)
    # ===============================
    layout = random.choice(["none", "left", "right", "top"])

    if layout == "left":
        img_path = random_file_from_dir("assets/popup_left")
        if img_path:
            popup["left_image_path"] = img_path
            popup["left_width"] = rint(180, min(300, w // 2))

    elif layout == "right":
        img_path = random_file_from_dir("assets/popup_left")
        if img_path:
            popup["right_image_path"] = img_path
            popup["right_width"] = rint(180, min(300, w // 2))

    elif layout == "top":
        img_path = random_file_from_dir("assets/popup_top")
        if img_path:
            popup["top_image_path"] = img_path
            popup["top_image_height"] = rint(int(h * 0.25), int(h * 0.45))

    # layout == "none" → không thêm field nào

    return popup


# =========================================================
# WATERMARK
# =========================================================

def random_watermark(img_w, img_h):
    mode = random.choice(["tiled", "single"])

    font_size = rint(15, 17)

    # đảm bảo nhìn thấy
    alpha = rint(110, 160)

    r, g, b = random.choice(TEXT_COLOR_PALETTE)

    wm = {
        "type": "watermark",
        "z_index": rint(0, 5),
        "opacity": 1,

        "text": "PLACEHOLDER WATERMARK",

        "mode": mode,
        "angle_deg": rint(-25, -15),

        "font_path": random_file_from_dir("assets/fonts", (".ttf", ".otf")),
        "font_size": font_size,

        "text_rgba": rand_text_rgba(),
        "angle_deg": rint(-25, -15),
        "spacing": rint(220, 300),
        # bắt buộc để không chìm trên nền trắng
        "stroke_w": 2,
        "stroke_rgba": [255, 255, 255, min(alpha + 40, 255)],
    }

    if mode == "tiled":
        wm["spacing"] = rint(220, 300)
    else:
        wm["x"] = img_w // 2
        wm["y"] = img_h // 2

    return wm
