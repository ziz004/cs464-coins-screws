"""
===============================================================================
 Project     : Coins & Screws Counter and Sorter
 Course      : CS464 - Computer Vision
 Instructor  : Dr. Khaled Al Bahnasi
 University  : JIC
 Student     : Ahmed Zaitoun
 Section     : 5
 Student ID  : 231100182
===============================================================================
Core image-analysis module - classical computer vision (OpenCV + NumPy, no training)

Pipeline
 1) Preprocessing & Thresholding : gray -> Gaussian blur -> Otsu (auto light/dark background)
 2) Morphology & Watershed       : open/close + hole filling, then Distance Transform
                                   + marker-based watershed to split touching coins
                                   (a split is accepted only if every piece is round,
                                    so screws are never cut)
 3) Shape features               : circularity = 4*pi*A/P^2 , aspect ratio (minAreaRect)
                                   coin  -> circularity ~1 , aspect ~1
                                   screw -> elongated, high aspect ratio
 4) Coin classification          : diameter px -> mm using PX_PER_MM calibration,
                                   + HSV colour of CENTER vs OUTER RING to recognise
                                   bimetallic (1 SAR / 2 SAR / 1 EUR / 2 EUR), gold/brass,
                                   copper and silver coins
 5) Screw classification         : length from minAreaRect -> short / medium / long,
                                   then counts, sizes and total value drawn on the image
"""
import cv2
import numpy as np

# ------------------------------------------------------------------ coin tables
# material codes: silver | gold | copper | bi_SG (silver centre, gold ring) | bi_GS (gold centre, silver ring)
COIN_TABLES = {
    "SAR": [  # Saudi Arabia, 2016 issue
        {"name": "2 SAR",      "diameter_mm": 25.0, "material": "bi_GS",  "value": 2.00},
        {"name": "1 SAR",      "diameter_mm": 23.0, "material": "bi_SG",  "value": 1.00},
        {"name": "50 Halala",  "diameter_mm": 24.0, "material": "gold",   "value": 0.50},
        {"name": "25 Halala",  "diameter_mm": 21.5, "material": "gold",   "value": 0.25},
        {"name": "10 Halala",  "diameter_mm": 19.0, "material": "silver", "value": 0.10},
        {"name": "5 Halala",   "diameter_mm": 17.5, "material": "silver", "value": 0.05},
        {"name": "1 Halala",   "diameter_mm": 16.0, "material": "silver", "value": 0.01},
    ],
    "EUR": [
        {"name": "2 EUR",   "diameter_mm": 25.75, "material": "bi_GS",  "value": 2.00},
        {"name": "1 EUR",   "diameter_mm": 23.25, "material": "bi_SG",  "value": 1.00},
        {"name": "50 cent", "diameter_mm": 24.25, "material": "gold",   "value": 0.50},
        {"name": "20 cent", "diameter_mm": 22.25, "material": "gold",   "value": 0.20},
        {"name": "10 cent", "diameter_mm": 19.75, "material": "gold",   "value": 0.10},
        {"name": "5 cent",  "diameter_mm": 21.25, "material": "copper", "value": 0.05},
        {"name": "2 cent",  "diameter_mm": 18.75, "material": "copper", "value": 0.02},
        {"name": "1 cent",  "diameter_mm": 16.25, "material": "copper", "value": 0.01},
    ],
}

COLORS = {"coin": (0, 200, 255), "screw": (255, 120, 0), "unknown": (160, 160, 160)}


# ================================================================== 1) threshold
def binarize(gray, blur_k=5):
    k = max(1, blur_k | 1)
    blur = cv2.GaussianBlur(gray, (k, k), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # objects must be white: the image border is assumed to be background
    border = np.concatenate([th[0, :], th[-1, :], th[:, 0], th[:, -1]])
    if border.mean() > 127:
        th = cv2.bitwise_not(th)
    return th


# ================================================================== 2) morphology
def clean_mask(mask, morph_k=5):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_k, morph_k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=2)   # remove specks
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)  # close gaps
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, cnts, -1, 255, cv2.FILLED)              # fill holes / reflections
    return filled


# ================================================================== 3) features
def shape_features(c):
    area = cv2.contourArea(c)
    per = cv2.arcLength(c, True)
    circ = 4 * np.pi * area / (per * per) if per > 0 else 0
    (cx, cy), (w, h), ang = cv2.minAreaRect(c)
    long_side, short_side = max(w, h), max(min(w, h), 1e-6)
    hull_a = cv2.contourArea(cv2.convexHull(c))
    return dict(area=area, perimeter=per, circularity=circ,
                aspect_ratio=long_side / short_side,
                solidity=area / hull_a if hull_a > 0 else 0,
                length_px=long_side, width_px=short_side,
                diameter_px=2 * np.sqrt(area / np.pi), cx=cx, cy=cy)


def _is_round(f, circ_thr):
    return f["circularity"] >= circ_thr and f["aspect_ratio"] < 1.3


# ================================================================== 2b) watershed
def split_touching(blob_mask, sep, circ_thr):
    """Distance transform -> seeds -> marker-based watershed on one blob.
    Returns list of contours, or None if the blob should stay whole."""
    dist = cv2.distanceTransform(blob_mask, cv2.DIST_L2, 5)
    seeds = np.uint8(dist > sep * dist.max()) * 255
    n, markers = cv2.connectedComponents(seeds)
    if n - 1 < 2:
        return None
    # each blob pixel joins its nearest seed (watershed on the distance map)
    d_to_seed = [cv2.distanceTransform(np.uint8(markers != lab) * 255, cv2.DIST_L2, 5)
                 for lab in range(1, n)]
    labels = np.where(blob_mask > 0, np.argmin(np.stack(d_to_seed), axis=0) + 1, 0)
    pieces = []
    for lab in range(1, n):
        m = np.uint8(labels == lab) * 255
        c, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not c:
            return None
        c = max(c, key=cv2.contourArea)
        f = shape_features(c)
        if f["circularity"] < circ_thr * 0.85 or f["aspect_ratio"] > 1.35:
            return None            # pieces not round -> not touching coins (e.g. a screw)
        pieces.append(c)
    return pieces


def find_objects(mask, use_ws, sep, circ_thr, min_area):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = []
    for c in cnts:
        if cv2.contourArea(c) < min_area:
            continue
        if use_ws and not _is_round(shape_features(c), circ_thr):
            blob = np.zeros_like(mask)
            cv2.drawContours(blob, [c], -1, 255, cv2.FILLED)
            pieces = split_touching(blob, sep, circ_thr)
            if pieces:
                out += [(p, True) for p in pieces if cv2.contourArea(p) >= min_area]
                continue
        out.append((c, False))
    return out


# ================================================================== 4) HSV colour
def _region_colour(hsv, region, sat_thr):
    px = hsv[region]
    px = px[px[:, 2] > 40]                        # ignore very dark pixels / shadows
    if len(px) < 10:
        return "silver", 0.0, 0.0
    h, s = float(np.median(px[:, 0])), float(np.median(px[:, 1]))
    if s < sat_thr:
        name = "silver"
    elif h <= 12 or h >= 165:                     # OpenCV hue is 0..179
        name = "copper"
    elif h <= 35:
        name = "gold"
    else:
        name = "silver"
    return name, h, s


def coin_material(img_bgr, contour, sat_thr=60):
    """Compare the colour of the coin CENTER with its OUTER RING in HSV."""
    x0, y0, w, h = cv2.boundingRect(contour)
    roi = img_bgr[y0:y0 + h, x0:x0 + w]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    c = contour - [x0, y0]
    (x, y), r = cv2.minEnclosingCircle(c)
    yy, xx = np.ogrid[:h, :w]
    d = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)
    inside = np.zeros((h, w), np.uint8)
    cv2.drawContours(inside, [c], -1, 1, cv2.FILLED)
    inside = inside.astype(bool)
    center = inside & (d < 0.45 * r)
    ring = inside & (d > 0.72 * r) & (d < 0.92 * r)
    c_name, c_h, c_s = _region_colour(hsv, center, sat_thr)
    r_name, r_h, r_s = _region_colour(hsv, ring, sat_thr)
    if c_name == "silver" and r_name != "silver":
        mat = "bi_SG"
    elif c_name != "silver" and r_name == "silver":
        mat = "bi_GS"
    else:
        mat = r_name
    return mat, dict(center=c_name, ring=r_name, center_h=c_h, center_s=c_s, ring_h=r_h, ring_s=r_s)


def _material_score(found, ref):
    if found == ref:
        return 0.0
    yellow = {"gold", "copper"}
    if found in yellow and ref in yellow:
        return 0.5          # gold vs copper is easy to confuse under bad lighting
    return 2.0


def match_coin(d_mm, mat, table, tol_mm, use_colour=True):
    best, best_score = None, 1e9
    for t in table:
        dd = abs(float(t["diameter_mm"]) - d_mm)
        if dd > tol_mm:
            continue
        score = dd / tol_mm + (_material_score(mat, t.get("material", "")) if use_colour else 0)
        if score < best_score:
            best, best_score = t, score
    return best


# ================================================================== helpers
def size_groups(values, names, prefix, gap_ratio=0.12):
    """1-D clustering: split sorted values where the relative jump is > gap_ratio."""
    if not values:
        return []
    order = np.argsort(values)
    grp, g = {}, 0
    for i, idx in enumerate(order):
        if i and (values[idx] - values[order[i - 1]]) / max(values[order[i - 1]], 1e-6) > gap_ratio:
            g += 1
        grp[idx] = g
    n = g + 1
    out = []
    for i in range(len(values)):
        if n <= len(names):
            pos = 0 if n == 1 else round(grp[i] * (len(names) - 1) / (n - 1))
            out.append(names[pos])
        else:
            out.append(f"{prefix} size {grp[i] + 1}")
    return out


def screw_class(length_mm, short_max, long_min):
    if length_mm < short_max:
        return "short"
    if length_mm >= long_min:
        return "long"
    return "medium"


def calibrate_from_reference(img_bgr, ref_diameter_mm, **kw):
    """PX_PER_MM from the LARGEST coin in the image whose real diameter is known."""
    kw = {k: v for k, v in kw.items() if k in ("min_area", "blur_k", "morph_k", "circ_thr",
                                               "ar_thr", "use_watershed", "ws_sep")}
    _, objs, _ = analyze(img_bgr, px_per_mm=None, **kw)
    coins = [o for o in objs if o["cls"] == "coin"]
    if not coins:
        return None
    return max(o["diameter_px"] for o in coins) / ref_diameter_mm


# ================================================================== main
def analyze(img_bgr, min_area=300, blur_k=5, morph_k=5, circ_thr=0.80, ar_thr=2.0,
            use_watershed=True, ws_sep=0.6, px_per_mm=None, coin_table=None,
            coin_tol_mm=1.0, use_colour=True, sat_thr=60,
            screw_short_max=25.0, screw_long_min=50.0, currency=""):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mask = clean_mask(binarize(gray, blur_k), morph_k)

    objs = []
    for c, split in find_objects(mask, use_watershed, ws_sep, circ_thr, min_area):
        f = shape_features(c)
        if split:
            f["diameter_px"] = 2 * cv2.minEnclosingCircle(c)[1]  # piece lost a sliver at the cut
            cls = "coin"
        elif _is_round(f, circ_thr):
            cls = "coin"
        elif f["aspect_ratio"] >= ar_thr:
            cls = "screw"
        else:
            cls = "unknown"
        f.update(cls=cls, contour=c, split=split, subtype="", value=0.0,
                 material="", diameter_mm=None, length_mm=None)
        objs.append(f)

    coins = [o for o in objs if o["cls"] == "coin"]
    screws = [o for o in objs if o["cls"] == "screw"]

    # ---- 4) coins: colour + size ----
    for o in coins:
        o["material"], o["colour_info"] = coin_material(img_bgr, o["contour"], sat_thr)
    if px_per_mm:
        for o in coins:
            o["diameter_mm"] = o["diameter_px"] / px_per_mm
            t = match_coin(o["diameter_mm"], o["material"], coin_table or [], coin_tol_mm, use_colour)
            if t:
                o["subtype"], o["value"] = t["name"], float(t["value"])
            else:
                o["subtype"] = "unmatched"
    else:
        labs = size_groups([o["diameter_px"] for o in coins],
                           ["small coin", "medium coin", "large coin"], "coin")
        for o, lab in zip(coins, labs):
            o["subtype"] = lab

    # ---- 5) screws: length ----
    if px_per_mm:
        for o in screws:
            o["length_mm"] = o["length_px"] / px_per_mm
            o["subtype"] = screw_class(o["length_mm"], screw_short_max, screw_long_min) + " screw"
    else:
        labs = size_groups([o["length_px"] for o in screws],
                           ["short screw", "medium screw", "long screw"], "screw", 0.15)
        for o, lab in zip(screws, labs):
            o["subtype"] = lab

    assign_ids(objs)
    out = draw_results(img_bgr, objs, currency)
    return out, objs, mask


# ================================================================== drawing
PILL = {"coin": ((0, 185, 245), (20, 20, 20)),      # BGR: amber pill, dark text
        "screw": ((222, 104, 29), (255, 255, 255)),  # BGR: blue pill, white text
        "unknown": ((150, 150, 150), (255, 255, 255))}
FONT = cv2.FONT_HERSHEY_DUPLEX


def assign_ids(objs):
    """Stable IDs per class, ordered top-to-bottom then left-to-right: C1.., S1.., U1.."""
    counters = {"coin": 0, "screw": 0, "unknown": 0}
    for o in sorted(objs, key=lambda o: (round(o["cy"] / 50), o["cx"])):
        counters[o["cls"]] += 1
        o["id"] = {"coin": "C", "screw": "S", "unknown": "U"}[o["cls"]] + str(counters[o["cls"]])


def _label(o):
    """Short on-image label; full sizes are listed in the tables."""
    s = o["id"]
    if o["subtype"] and o["subtype"] != "unmatched":
        s += " " + o["subtype"].replace(" coin", "").replace(" screw", "")
    return s


def _overlap(a, b):
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def _pill(img, text, box, fs, th, bg, fg, top_limit, placed):
    """Readable label (filled box + text) placed above, below or beside the object without overlaps."""
    (tw, tht), base = cv2.getTextSize(text, FONT, fs, th)
    pad = max(4, int(6 * fs))
    H, W = img.shape[:2]
    bw, bhh = tw + 2 * pad, tht + base + 2 * pad
    x, y, w, h = box
    cx = x + w // 2
    cands = [(cx - bw // 2, y - bhh - 4), (cx - bw // 2, y + h + 4),
             (x + w + 4, y + h // 2 - bhh // 2), (x - bw - 4, y + h // 2 - bhh // 2)]
    cands += [(cx - bw // 2, y - bhh - 4 - k * (bhh + 3)) for k in range(1, 4)]
    best = None
    for px, py in cands:
        px = int(min(max(2, px), W - bw - 2))
        py = int(min(max(top_limit + 2, py), H - bhh - 2))
        r = (px, py, px + bw, py + bhh)
        if not any(_overlap(r, q) for q in placed):
            best = r
            break
    if best is None:
        px = int(min(max(2, cx - bw // 2), W - bw - 2)); py = int(min(max(top_limit + 2, y - bhh - 4), H - bhh - 2))
        best = (px, py, px + bw, py + bhh)
    placed.append(best)
    x0, y0, x1, y1 = best
    cv2.rectangle(img, (x0 + 2, y0 + 2), (x1 + 2, y1 + 2), (0, 0, 0), -1)  # shadow
    cv2.rectangle(img, (x0, y0), (x1, y1), bg, -1)
    cv2.putText(img, text, (x0 + pad, y0 + pad + tht), FONT, fs, fg, th, cv2.LINE_AA)


def draw_results(img_bgr, objs, currency="", show=("coin", "screw", "unknown")):
    """Draw outlines + labels for the classes in `show`, with a summary banner."""
    out = img_bgr.copy()
    H, W = out.shape[:2]
    lw = max(2, int(round(max(H, W) / 450)))
    fs = max(0.45, max(H, W) / 2000)
    th = 1 if fs < 0.8 else 2
    bh = int(36 * max(fs, 0.6) + 10)
    sel = [o for o in objs if o["cls"] in show]
    for o in sel:
        bg, fg = PILL[o["cls"]]
        cv2.drawContours(out, [o["contour"]], -1, bg, lw + 1)
        if o["cls"] == "screw":
            box = cv2.boxPoints(cv2.minAreaRect(o["contour"])).astype(np.int32)
            cv2.drawContours(out, [box], -1, (255, 255, 255), max(1, lw - 1))
    placed = []
    for o in sorted(sel, key=lambda o: (o["cy"], o["cx"])):  # labels on top of every outline
        bg, fg = PILL[o["cls"]]
        _pill(out, _label(o), cv2.boundingRect(o["contour"]), fs, th, bg, fg, bh, placed)

    # summary banner
    parts = []
    if "coin" in show:
        parts.append(f"Coins: {sum(o['cls'] == 'coin' for o in objs)}")
    if "screw" in show:
        parts.append(f"Screws: {sum(o['cls'] == 'screw' for o in objs)}")
    total = sum(o["value"] for o in objs)
    if "coin" in show and total > 0:
        parts.append(f"Total: {total:.2f} {currency}")
    cv2.rectangle(out, (0, 0), (W, bh), (63, 27, 11), -1)
    cv2.putText(out, "    ".join(parts), (12, bh - int(12 * max(fs, 0.6))), FONT, max(fs, 0.6) * 1.05,
                (255, 255, 255), th, cv2.LINE_AA)
    return out


# ================================================================== demo image
def make_demo_image(seed=0):
    """Synthetic photo: SAR coins at 4 px/mm (light background) + screws, two coins touching."""
    rng = np.random.default_rng(seed)
    img = np.full((640, 950, 3), 228, np.uint8)
    img = cv2.add(img, rng.integers(0, 10, img.shape, dtype=np.uint8))
    SILVER, GOLD = (150, 150, 155), (40, 160, 205)
    PPM = 4.0

    def coin(x, y, d_mm, centre, ring):
        r = int(round(d_mm * PPM / 2))
        cv2.circle(img, (x, y), r, ring, -1)
        if centre != ring:
            cv2.circle(img, (x, y), int(r * 0.66), centre, -1)
        cv2.circle(img, (x, y), int(r * 0.5), tuple(max(0, v - 25) for v in centre), 2)

    coin(110, 140, 25.0, GOLD, SILVER)     # 2 SAR
    coin(270, 140, 23.0, SILVER, GOLD)     # 1 SAR
    coin(430, 140, 24.0, GOLD, GOLD)       # 50 halala
    coin(110, 320, 21.5, GOLD, GOLD)       # 25 halala
    coin(250, 320, 19.0, SILVER, SILVER)   # 10 halala
    coin(372, 320, 23.0, SILVER, GOLD)     # 1 SAR   } touching
    coin(463, 320, 23.0, SILVER, GOLD)     # 1 SAR   }
    for (cx, cy, L, ang) in [(740, 170, 260, 20), (800, 460, 150, -60),
                             (240, 540, 70, 5), (540, 510, 160, 80)]:
        rect = ((cx, cy), (L, 16), ang)
        cv2.fillPoly(img, [cv2.boxPoints(rect).astype(np.int32)], (85, 85, 90))
        a = np.radians(ang)
        cv2.circle(img, (int(cx - np.cos(a) * L / 2), int(cy - np.sin(a) * L / 2)), 15, (85, 85, 90), -1)
    return cv2.GaussianBlur(img, (3, 3), 0)
