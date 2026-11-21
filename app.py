"""
===============================================================================
 Project     : Coins & Screws Counter and Sorter
 Course      : CS464 - Computer Vision
 Instructor  : Dr. Khaled Al Bahnasi
 University  : JIC
 Student     : Ahmed Zaitoun
 Section     : 5
 Student ID  : 231100182
-------------------------------------------------------------------------------
 Streamlit interface: upload images, detect, count and classify coins and
 screws with classical computer vision (OpenCV, no model training).
===============================================================================
"""
import base64
import os

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import vision as V

PROJECT = {
    "title": "Coins & Screws Counter and Sorter",
    "course": "CS464 - Computer Vision",
    "instructor": "Dr. Khaled Al Bahnasi",
    "university": "JIC",
    "student": "Ahmed Zaitoun",
    "section": "5",
    "student_id": "231100182",
    "email": "ahmed.zaitoun.ps@gmail.com",
    "github": "ziz004",
}
MAIL_ICON = ("<svg width='15' height='15' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2' "
             "stroke-linecap='round' stroke-linejoin='round'><rect x='3' y='5' width='18' height='14' rx='2'/>"
             "<path d='m3 7 9 6 9-6'/></svg>")
GITHUB_ICON = ("<svg width='15' height='15' viewBox='0 0 24 24' fill='currentColor'><path d='M12 .5a11.5 11.5 0 0 0-3.64 "
               "22.41c.58.1.79-.25.79-.56v-2c-3.2.7-3.88-1.37-3.88-1.37-.52-1.33-1.28-1.69-1.28-1.69-1.04-.71.08-.7.08-.7 "
               "1.15.08 1.76 1.19 1.76 1.19 1.03 1.76 2.7 1.25 3.36.96.1-.75.4-1.25.73-1.54-2.56-.29-5.25-1.28-5.25-5.68 "
               "0-1.26.45-2.28 1.19-3.09-.12-.29-.52-1.46.11-3.05 0 0 .97-.31 3.17 1.18a11 11 0 0 1 5.77 0c2.2-1.49 3.17-1.18 "
               "3.17-1.18.63 1.59.23 2.76.11 3.05.74.81 1.19 1.83 1.19 3.09 0 4.41-2.7 5.38-5.27 5.67.41.36.78 1.06.78 2.14v3.17"
               "c0 .31.21.67.8.56A11.5 11.5 0 0 0 12 .5Z'/></svg>")


def contact_links(cls="contact"):
    """Clickable e-mail (opens the mail app) and GitHub profile (opens in a new tab)."""
    return (f"<div class='{cls}'>"
            f"<a href='mailto:{PROJECT['email']}' target='_blank'>{MAIL_ICON}<span>{PROJECT['email']}</span></a>"
            f"<a href='https://github.com/{PROJECT['github']}' target='_blank' rel='noopener'>{GITHUB_ICON}"
            f"<span>github.com/{PROJECT['github']}</span></a></div>")

APP_DIR = os.path.dirname(os.path.abspath(__file__))


def logo_uri():
    """University logo (white SVG) as a data URI: logo.svg, logo2.svg or any .svg next to app.py."""
    names = ["logo.svg", "logo2.svg"] + sorted(f for f in os.listdir(APP_DIR) if f.lower().endswith(".svg"))
    for name in names:
        path = os.path.join(APP_DIR, name)
        if os.path.isfile(path):
            with open(path, "rb") as f:
                return "data:image/svg+xml;base64," + base64.b64encode(f.read()).decode()
    return None


LOGO = logo_uri()

st.set_page_config(page_title="Coin & Screw Counter · CS464", page_icon="🪙", layout="wide",
                   initial_sidebar_state="expanded")

MATERIALS = {"silver": "Silver", "gold": "Gold / brass", "copper": "Copper",
             "bi_SG": "Bimetal · silver core", "bi_GS": "Bimetal · gold core"}
SIZES = {"small coin": "Small coin", "medium coin": "Medium coin", "large coin": "Large coin",
         "short screw": "Short screw", "medium screw": "Medium screw", "long screw": "Long screw",
         "unmatched": "Unmatched"}
SWATCH = {"silver": "#B9BCC2", "gold": "#D9A520", "copper": "#B8673A",
          "bi_SG": "radial-gradient(circle,#B9BCC2 0 44%,#D9A520 46%)",
          "bi_GS": "radial-gradient(circle,#D9A520 0 44%,#B9BCC2 46%)"}
PIPELINE = ["Preprocess · Otsu", "Split touching · Watershed", "Shape · Circularity",
            "Coins · HSV", "Screws · minAreaRect"]
TIPS = [("Shoot straight down", "Keep the camera parallel to the surface so circles stay round."),
        ("Plain background", "A white sheet or single-colour surface keeps Otsu clean."),
        ("Even lighting", "Avoid hard shadows so HSV colours stay accurate."),
        ("No overlaps", "Touching is fine and gets split; stacking is not.")]

# =============================================================== styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500;600&display=swap');
html, body, .stApp { font-family: 'Inter', system-ui, sans-serif; }
.stApp p, .stApp label, .stApp h1, .stApp h2, .stApp h3, .stApp button, .stApp input { font-family: 'Inter', system-ui, sans-serif; }
[data-testid="stIconMaterial"], .material-symbols-rounded { font-family: 'Material Symbols Rounded' !important; }
.stApp { background: #F2F6FC; color: #0B1B3F; }
/* readable on any browser theme (light or dark) */
[data-testid="stMain"] [data-testid="stWidgetLabel"] p, [data-testid="stMain"] [data-testid="stMarkdownContainer"] p,
[data-testid="stMain"] [data-testid="stMarkdownContainer"] strong, [data-testid="stMain"] [data-testid="stCaptionContainer"] { color: #0B1B3F !important; }
[data-testid="stMain"] .stTabs [data-baseweb="tab"] p { color: #5A6A8A !important; font-weight: 600; }
[data-testid="stMain"] .stTabs [aria-selected="true"] p { color: #1D4ED8 !important; }
[data-testid="stMain"] [data-baseweb="select"] > div { background: #FFFFFF !important; border-color: #C3D3F0 !important; }
[data-testid="stMain"] [data-baseweb="select"] * { color: #0B1B3F !important; }
[data-testid="stFileUploaderDropzone"] * { color: #3A4A6B !important; }
section[data-testid="stSidebar"] [data-baseweb="select"] > div { background: #FFFFFF !important; }
section[data-testid="stSidebar"] [data-testid="stExpander"] summary * { color: #E6ECF8 !important; }
.brandbar { display:flex; align-items:center; justify-content:space-between; gap:24px; flex-wrap:wrap;
  background:linear-gradient(120deg,#0B1B3F 0%,#14397F 60%,#1D4ED8 100%); border-radius:22px; padding:22px 28px; margin-bottom:22px;
  box-shadow:0 14px 30px -10px rgba(11,27,63,.35); }
.brandbar img { height:64px; max-width:60%; object-fit:contain; }
.brandbar .bb-txt { display:flex; flex-direction:column; align-items:flex-end; gap:2px; }
.brandbar .bb-txt b { color:#fff; font-size:15px; } .brandbar .bb-txt span { color:#C3D3F0; font-size:13px; }
.contact { display:flex; flex-direction:column; gap:6px; margin-top:10px; }
.contact a { display:flex; align-items:center; gap:8px; text-decoration:none !important; font-size:12.5px;
  color:#FFFFFF !important; background:rgba(255,255,255,.10); border-radius:10px; padding:8px 10px; transition:background .15s; }
.contact a:hover { background:rgba(255,255,255,.20); }
.contact a svg { flex:none; color:#93C5FD; }
.contact a span { color:#FFFFFF !important; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.contact-main { display:flex; flex-wrap:wrap; gap:10px; margin-top:10px; }
.contact-main a { display:inline-flex; align-items:center; gap:8px; text-decoration:none !important; font-size:13px; font-weight:500;
  color:#1D4ED8 !important; background:#FFFFFF; border-radius:999px; padding:7px 14px; box-shadow:0 2px 6px rgba(11,27,63,.08); }
.contact-main a:hover { background:#E0ECFF; }
.contact-main a span { color:#1D4ED8 !important; }
.section-h { display:flex; align-items:center; gap:10px; margin: 22px 0 10px; }
.section-h b { font-size: 20px; color: #0B1B3F; } .section-h span.n { font-family:'IBM Plex Mono',monospace; font-size:12px;
  background:#E3EAF6; color:#1E3A8A; border-radius:999px; padding:2px 9px; }
.badge-c { width:12px; height:12px; border-radius:50%; background:#F5B900; display:inline-block; }
.badge-s { width:12px; height:12px; border-radius:3px; background:#1D68DE; display:inline-block; }
.block-container { padding-top: 2rem; max-width: 1320px; }
section[data-testid="stSidebar"] { background: linear-gradient(180deg, #0B1B3F 0%, #14397F 100%); }
section[data-testid="stSidebar"] * { color: #E6ECF8; }
section[data-testid="stSidebar"] [data-baseweb="select"] *, section[data-testid="stSidebar"] input { color: #0B1B3F !important; }
section[data-testid="stSidebar"] h3 { font-size: 13px; color: #A7B8DB !important; font-weight: 600; margin: 8px 0 0; }
header[data-testid="stHeader"] { background: transparent; }
.hero h1 { margin: 0; font-size: 32px; font-weight: 700; color: #0B1B3F; letter-spacing: -0.5px; }
.hero p { margin: 4px 0 0; color: #5A6A8A; }
.chip { display:inline-block; font-family:'IBM Plex Mono',monospace; font-size:12px; color:#3A4A6B; background:#fff;
        border-radius:999px; padding:4px 11px; margin:8px 6px 0 0; box-shadow:0 2px 6px rgba(11,27,63,.08); }
.tag { display:inline-block; font-family:'IBM Plex Mono',monospace; font-size:12px; color:#3A4A6B; background:#E3EAF6;
        border-radius:6px; padding:2px 8px; }
.kpis { display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:12px; margin: 8px 0 4px; }
.kpi { background:#fff; border-radius:20px; padding:16px 18px; box-shadow:0 1px 2px rgba(11,27,63,.06), 0 8px 24px -4px rgba(11,27,63,.06); }
.kpi.dark { background:linear-gradient(135deg,#0B1B3F 0%,#163F94 55%,#2563EB 100%); box-shadow:0 14px 30px -8px rgba(29,78,216,.35); }
.kpi .l { font-size:13px; color:#5A6A8A; font-weight:500; display:flex; align-items:center; gap:8px; }
.kpi .v { font-family:'IBM Plex Mono',monospace; font-size:32px; font-weight:600; line-height:1.15; color:#0B1B3F; }
.kpi .u { font-size:14px; font-weight:600; color:#5A6A8A; margin-left:6px; }
.kpi .s { font-size:12px; color:#5A6A8A; }
.kpi.dark .l, .kpi.dark .s { color:#C3D3F0; } .kpi.dark .v { color:#fff; } .kpi.dark .u { color:#93C5FD; }
.dot { width:9px; height:9px; border-radius:50%; display:inline-block; }
.card { background:#fff; border-radius:18px; padding:16px 18px; box-shadow:0 1px 2px rgba(11,27,63,.06), 0 8px 24px -4px rgba(11,27,63,.06); }
.sw { width:24px; height:24px; border-radius:50%; border:1px solid #E1E8F3; display:inline-block; flex:none; }
.hrow { display:flex; align-items:center; gap:12px; background:#F2F6FC; border-radius:12px; padding:10px 12px; margin-bottom:8px; }
.hrow .m { margin-left:auto; font-family:'IBM Plex Mono',monospace; font-size:12px; color:#3A4A6B; }
.cand { display:flex; align-items:center; gap:12px; padding:10px 12px; border-radius:12px; margin-bottom:6px; background:#fff;
        box-shadow:0 1px 2px rgba(11,27,63,.06); }
.cand.ok { background:#E8F6EC; box-shadow:none; border:1px solid #BFE3C9; }
.cand .d { margin-left:auto; font-family:'IBM Plex Mono',monospace; font-size:12px; }
.pill { font-size:11px; font-weight:600; border-radius:999px; padding:2px 9px; background:#EDF2FA; color:#3A4A6B; }
.cand.ok .pill { background:#1F8A3B; color:#fff; }
.note { background:#E4EBF8; color:#163A7A; border-radius:14px; padding:12px 14px; font-size:13px; }
.tip { background:#fff; border-radius:18px; padding:16px; height:100%; box-shadow:0 1px 2px rgba(11,27,63,.06), 0 8px 24px -4px rgba(11,27,63,.06); }
.tip b { display:block; margin:6px 0 4px; color:#0B1B3F; } .tip span { color:#5A6A8A; font-size:14px; }
.step { font-size:13px; padding:3px 0; } .step::before { content:'✓ '; color:#93C5FD; font-weight:700; }
.info { background:rgba(255,255,255,.08); border-radius:14px; padding:10px 12px; font-size:12px; line-height:1.7; }
.info b { color:#fff; font-weight:600; } .info span { color:#A7B8DB; }
.stButton button, .stDownloadButton button { border-radius:999px; font-weight:600; min-height:44px; }
.stDownloadButton button { background:linear-gradient(90deg,#0B2A6B,#2563EB); color:#fff; border:0; }
[data-testid="stFileUploaderDropzone"] { border:2px dashed #C3D3F0; border-radius:18px; background:#fff; padding: 28px; }
</style>
""", unsafe_allow_html=True)

# =============================================================== sidebar
with st.sidebar:
    if LOGO:
        st.markdown(f"<img src='{LOGO}' alt='{PROJECT['university']} logo' "
                    "style='width:100%;max-width:260px;display:block;margin:4px 0 18px'>", unsafe_allow_html=True)
    st.markdown("<div style='display:flex;align-items:center;gap:10px;margin-bottom:8px'>"
                "<span style='font-size:26px'>🪙</span><div><div style='font-weight:700;font-size:16px;color:#fff'>Coin &amp; Screw Counter</div>"
                f"<div style='font-family:IBM Plex Mono,monospace;font-size:11px;color:#A7B8DB'>{PROJECT['course']}</div></div></div>",
                unsafe_allow_html=True)
    st.markdown(
        "<div class='info'>"
        f"<span>Instructor</span><br><b>{PROJECT['instructor']}</b><br>"
        f"<span>University</span> <b>{PROJECT['university']}</b><br>"
        f"<span>Student</span><br><b>{PROJECT['student']}</b><br>"
        f"<span>Section</span> <b>{PROJECT['section']}</b> &nbsp;·&nbsp; <span>ID</span> <b>{PROJECT['student_id']}</b>"
        + contact_links() + "</div>", unsafe_allow_html=True)

    st.markdown("### Detection")
    min_area = st.slider("Minimum object area (px)", 50, 5000, 300, 50)
    blur_k = st.slider("Gaussian blur", 1, 21, 5, 2)
    morph_k = st.slider("Morphology kernel", 3, 15, 5, 2)
    circ_thr = st.slider("Coin circularity threshold", 0.50, 0.95, 0.80, 0.01)
    ar_thr = st.slider("Screw aspect-ratio threshold", 1.3, 6.0, 2.0, 0.1)
    use_ws = st.toggle("Split touching coins (Watershed)", True)
    ws_sep = st.slider("Distance-transform threshold", 0.3, 0.9, 0.6, 0.05, disabled=not use_ws)

    st.markdown("### Calibration PX_PER_MM")
    modes = ["Reference coin in image", "Enter PX_PER_MM", "None (relative sizes)"]
    mode = modes.index(st.selectbox("Calibration method", modes))
    currency = st.selectbox("Currency", ["SAR", "EUR"], disabled=mode == 2)
    table = pd.DataFrame(V.COIN_TABLES[currency])
    ref_name, manual_ppm = None, None
    if mode == 0:
        ref_name = st.selectbox("Largest coin in the image",
                                list(table.sort_values("diameter_mm", ascending=False)["name"]))
    elif mode == 1:
        manual_ppm = st.number_input("PX_PER_MM (pixels per mm)", 0.5, 100.0, 4.0, 0.1)
    if mode != 2:
        with st.expander("Coin table (editable)"):
            table = st.data_editor(table, num_rows="dynamic", use_container_width=True)
    tol = st.slider("Diameter tolerance (mm)", 0.3, 4.0, 1.0, 0.1, disabled=mode == 2)
    use_colour = st.toggle("Use HSV colour (centre vs ring)", True)
    sat_thr = st.slider("Saturation threshold: below = silver", 20, 150, 60, 5)

    st.markdown("### Screw classes (mm)")
    c1, c2 = st.columns(2)
    short_max = c1.number_input("Short below", 5.0, 200.0, 25.0, 1.0)
    long_min = c2.number_input("Long from", 5.0, 300.0, 50.0, 1.0)

    st.markdown("### Pipeline")
    st.markdown("".join(f"<div class='step'>{s}</div>" for s in PIPELINE), unsafe_allow_html=True)

# =============================================================== header + input
if LOGO:
    st.markdown(
        "<div class='brandbar'>"
        f"<img src='{LOGO}' alt='{PROJECT['university']} logo'>"
        f"<div class='bb-txt'><b>{PROJECT['course']}</b><span>{PROJECT['instructor']}</span></div>"
        "</div>", unsafe_allow_html=True)
st.markdown(
    f"<div class='hero'><span class='tag'>{PROJECT['course']} · {PROJECT['university']}</span>"
    f"<h1>{PROJECT['title']}</h1>"
    "<p>Count and classify coins and screws with classical computer vision — OpenCV, no model training.</p>"
    + "".join(f"<span class='chip'>{c}</span>" for c in
              ["Otsu", "Morphology", "Watershed", "Circularity", "HSV", "minAreaRect"]) + "</div>",
    unsafe_allow_html=True)
st.write("")

files = st.file_uploader("Drop images here or browse", type=["jpg", "jpeg", "png", "bmp", "webp"],
                         accept_multiple_files=True)
demo_on = st.session_state.get("demo", False)
if st.button("Hide demo image" if demo_on else "Try a demo image"):
    st.session_state.demo = not demo_on
    st.rerun()

images = []
if st.session_state.get("demo"):
    images.append(("demo.png", V.make_demo_image()))
for f in files or []:
    img = cv2.imdecode(np.frombuffer(f.getvalue(), np.uint8), cv2.IMREAD_COLOR)
    if img is not None:
        images.append((f.name, img))

if not images:
    st.markdown("<div style='margin-top:18px'><h3 style='margin:0;color:#0B1B3F'>Start with an image</h3>"
                "<p style='color:#5A6A8A;margin:2px 0 12px'>Upload one or more images, or try the demo image.</p></div>",
                unsafe_allow_html=True)
    cols = st.columns(4)
    for i, (title, body) in enumerate(TIPS):
        cols[i].markdown(f"<div class='tip'><span class='tag'>0{i + 1}</span><b>{title}</b><span>{body}</span></div>",
                         unsafe_allow_html=True)
    st.stop()

base_kw = dict(min_area=min_area, blur_k=blur_k, morph_k=morph_k, circ_thr=circ_thr,
               ar_thr=ar_thr, use_watershed=use_ws, ws_sep=ws_sep)
coin_table = table.dropna().to_dict("records")
all_rows = []


def type_name(o):
    return SIZES.get(o["subtype"], o["subtype"])


def class_name(o):
    return {"coin": "Coin", "screw": "Screw"}.get(o["cls"], "Unknown")


# =============================================================== per image
for name, img in images:
    h, w = img.shape[:2]
    if max(h, w) > 1600:  # speed: shrink very large phone photos
        s = 1600 / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)

    ppm = None
    if ref_name:
        ref_d = float(table.loc[table["name"] == ref_name, "diameter_mm"].iloc[0])
        ppm = V.calibrate_from_reference(img, ref_d, **base_kw)
    elif manual_ppm:
        ppm = manual_ppm

    out, objs, mask = V.analyze(img, **base_kw, px_per_mm=ppm, coin_table=coin_table, coin_tol_mm=tol,
                                use_colour=use_colour, sat_thr=sat_thr, screw_short_max=short_max,
                                screw_long_min=long_min, currency=currency if ppm else "")
    coins = [o for o in objs if o["cls"] == "coin"]
    screws = [o for o in objs if o["cls"] == "screw"]
    unknown = [o for o in objs if o["cls"] == "unknown"]
    known = [o for o in coins if o["value"] > 0]
    cnt = lambda k: sum(k in o["subtype"] for o in screws)

    st.divider()
    st.markdown(f"<span class='tag'>{name}</span>", unsafe_allow_html=True)
    if ref_name and not ppm:
        st.warning("No coin found to calibrate this image.")

    kp = [
        ("", "Coins", len(coins), "", f"{sum(o['split'] for o in coins)} split by Watershed", "#1D4ED8"),
        ("", "Screws", len(screws), "", f"{cnt('short')} short · {cnt('medium')} medium · {cnt('long')} long", "#60A5FA"),
        ("", "Unknown", len(unknown), "", "Outside the shape rules", "#9AA8C2"),
        ("", "Scale", f"{ppm:.2f}" if ppm else "—", "px/mm" if ppm else "",
         f"From {ref_name}" if ref_name else ("Not calibrated" if not ppm else "Manual"), None),
        ("dark", "Total", f"{sum(o['value'] for o in coins):.2f}" if ppm else "—", currency if ppm else "",
         f"From {len(known)} recognised coins", None),
    ]
    st.markdown("<div class='kpis'>" + "".join(
        f"<div class='kpi {c}'><div class='l'>" + (f"<span class='dot' style='background:{d}'></span>" if d else "") +
        f"{l}</div><div><span class='v'>{v}</span><span class='u'>{u}</span></div><div class='s'>{s}</div></div>"
        for c, l, v, u, s, d in kp) + "</div>", unsafe_allow_html=True)

    left, right = st.columns([2.1, 1], gap="large")
    with left:
        t_all, t_coins, t_screws, t_orig, t_mask = st.tabs(["All objects", "Coins only", "Screws only", "Original", "Binary mask"])
        t_all.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB),
                    caption="Amber = coin · blue = screw (white box = minAreaRect) · grey = unknown",
                    use_container_width=True)
        t_coins.image(cv2.cvtColor(V.draw_results(img, objs, currency if ppm else "", show=("coin",)), cv2.COLOR_BGR2RGB),
                      use_container_width=True)
        t_screws.image(cv2.cvtColor(V.draw_results(img, objs, show=("screw",)), cv2.COLOR_BGR2RGB),
                       use_container_width=True)
        t_orig.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)
        t_mask.image(mask, use_container_width=True)

    with right:
        st.markdown("**Inspect an object**")
        ordered = sorted(objs, key=lambda o: (o["cls"], int(o["id"][1:])))
        if ordered:
            ids = [o["id"] for o in ordered]
            pick = st.selectbox("Choose an object", ids, key=f"pick_{name}",
                                format_func=lambda i: f"{i} · {type_name(next(o for o in ordered if o['id'] == i))}")
            o = next(x for x in ordered if x["id"] == pick)
            x, y, bw, bh = cv2.boundingRect(o["contour"])
            p = 14
            crop = out[max(0, y - p):y + bh + p, max(0, x - p):x + bw + p]
            ci, cm = st.columns([1, 1.4])
            ci.image(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), use_container_width=True)
            cm.markdown(f"<span class='tag'>{o['id']}</span>"
                        f"<div style='font-size:20px;font-weight:700;margin-top:4px;color:#0B1B3F'>{type_name(o)}</div>"
                        f"<div style='color:#5A6A8A;font-size:13px'>{class_name(o)}</div>", unsafe_allow_html=True)
            met = []
            if o["cls"] == "coin":
                met.append(("Diameter", f"{o['diameter_mm']:.2f} mm" if o["diameter_mm"] else f"{o['diameter_px']:.1f} px"))
                if ppm:
                    met.append(("Value", f"{o['value']:.2f} {currency}"))
                met.append(("Watershed", "Split" if o["split"] else "Single"))
            elif o["cls"] == "screw":
                met.append(("Length", f"{o['length_mm']:.1f} mm" if o["length_mm"] else f"{o['length_px']:.1f} px"))
            met += [("Area", f"{int(o['area'])} px"), ("Circularity", f"{o['circularity']:.2f}"),
                    ("Aspect", f"{o['aspect_ratio']:.2f}")]
            st.markdown("<div class='card' style='display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px 16px;padding:14px'>" +
                        "".join(f"<div><div style='font-size:12px;color:#5A6A8A'>{k}</div>"
                                f"<div style='font-family:IBM Plex Mono,monospace;font-weight:600;color:#0B1B3F'>{v}</div></div>"
                                for k, v in met) + "</div>", unsafe_allow_html=True)

            if o["cls"] == "coin" and o.get("colour_info"):
                c = o["colour_info"]
                st.markdown("<div style='font-weight:700;margin:14px 0 8px;color:#0B1B3F'>HSV colour analysis</div>" +
                            "".join(f"<div class='hrow'><span class='sw' style='background:{SWATCH[n]}'></span>"
                                    f"<div><div style='font-size:12px;color:#5A6A8A'>{lab}</div>"
                                    f"<div style='font-weight:600'>{MATERIALS[n]}</div></div>"
                                    f"<span class='m'>H {hh:.0f} · S {ss:.0f}</span></div>"
                                    for lab, n, hh, ss in [("Centre", c["center"], c["center_h"], c["center_s"]),
                                                           ("Outer ring", c["ring"], c["ring_h"], c["ring_s"])]),
                            unsafe_allow_html=True)
                if ppm and o["diameter_mm"]:
                    cands = sorted(coin_table, key=lambda r: abs(float(r["diameter_mm"]) - o["diameter_mm"]))[:3]
                    html = "<div style='font-weight:700;margin:14px 0 8px;color:#0B1B3F'>Classification decision</div>"
                    for r in cands:
                        dd = abs(float(r["diameter_mm"]) - o["diameter_mm"])
                        ok = r["name"] == o["subtype"]
                        tag = "Chosen" if ok else ("Out of tolerance" if dd > tol else
                                                   ("Colour mismatch" if r.get("material") != o["material"] else ""))
                        html += (f"<div class='cand {'ok' if ok else ''}'><div><div style='font-weight:600'>{r['name']} · {r['diameter_mm']} mm</div>"
                                 f"<div style='font-size:12px;color:#5A6A8A'>{MATERIALS.get(r.get('material', ''), '')}</div></div>"
                                 f"<span class='d'>Δ {dd:.2f} mm</span>" + (f"<span class='pill'>{tag}</span>" if tag else "") + "</div>")
                    if cands and cands[0]["name"] != o["subtype"] and o["subtype"] != "unmatched":
                        html += (f"<div class='note'>By diameter the nearest is {cands[0]['name']}, but the centre/ring "
                                 f"colours match {o['subtype']} — colour decided it.</div>")
                    st.markdown(html, unsafe_allow_html=True)

    rows = []
    for o in sorted(objs, key=lambda o: (o["cls"], int(o["id"][1:]))):
        rows.append({
            "image": name, "ID": o["id"], "Class": class_name(o), "Type": type_name(o),
            "Colour": MATERIALS.get(o["material"], ""),
            "Diameter (mm)": round(o["diameter_mm"], 2) if o["diameter_mm"] else None,
            "Length (mm)": round(o["length_mm"], 1) if o["length_mm"] else None,
            "Circularity": round(o["circularity"], 3), "Aspect": round(o["aspect_ratio"], 2),
            "Area (px)": int(o["area"]),
            "Value": o["value"] if o["cls"] == "coin" and ppm else None,
        })
    all_rows += rows

    col_c, col_s = st.columns(2, gap="large")
    # ---------------- coins ----------------
    with col_c:
        st.markdown(f"<div class='section-h'><span class='badge-c'></span><b>Coins</b><span class='n'>{len(coins)}</span></div>",
                    unsafe_allow_html=True)
        if coins:
            cs = pd.DataFrame([{"Type": type_name(o), "Value": o["value"]} for o in coins])
            summary = cs.groupby("Type").agg(Count=("Type", "size"), Subtotal=("Value", "sum")).reset_index()
            summary = summary.sort_values("Subtotal", ascending=False)
            if not ppm:
                summary = summary.drop(columns=["Subtotal"])
            st.dataframe(summary, use_container_width=True, hide_index=True)
            st.dataframe(pd.DataFrame([{
                "ID": o["id"], "Type": type_name(o), "Colour": MATERIALS.get(o["material"], ""),
                "Diameter": f"{o['diameter_mm']:.2f} mm" if o["diameter_mm"] else f"{o['diameter_px']:.1f} px",
                "Circularity": round(o["circularity"], 2),
                **({"Value": o["value"]} if ppm else {}),
            } for o in sorted(coins, key=lambda o: int(o["id"][1:]))]), use_container_width=True, hide_index=True)
        else:
            st.info("No coins detected.")
    # ---------------- screws ----------------
    with col_s:
        st.markdown(f"<div class='section-h'><span class='badge-s'></span><b>Screws</b><span class='n'>{len(screws)}</span></div>",
                    unsafe_allow_html=True)
        if screws:
            ss = pd.DataFrame([{"Class": type_name(o), "Length": o["length_mm"] or o["length_px"]} for o in screws])
            unit = "mm" if ppm else "px"
            summary = ss.groupby("Class").agg(Count=("Class", "size"),
                                              **{f"Min ({unit})": ("Length", "min"), f"Max ({unit})": ("Length", "max")}).reset_index()
            st.dataframe(summary.round(1), use_container_width=True, hide_index=True)
            st.dataframe(pd.DataFrame([{
                "ID": o["id"], "Class": type_name(o),
                "Length": f"{o['length_mm']:.1f} mm" if o["length_mm"] else f"{o['length_px']:.1f} px",
                "Width": f"{o['width_px'] / ppm:.1f} mm" if ppm else f"{o['width_px']:.1f} px",
                "Aspect": round(o["aspect_ratio"], 2),
            } for o in sorted(screws, key=lambda o: int(o["id"][1:]))]), use_container_width=True, hide_index=True)
        else:
            st.info("No screws detected.")
    if unknown:
        st.markdown(f"<div class='section-h'><b>Unknown objects</b><span class='n'>{len(unknown)}</span></div>",
                    unsafe_allow_html=True)
        st.dataframe(pd.DataFrame([{"ID": o["id"], "Circularity": round(o["circularity"], 2),
                                    "Aspect": round(o["aspect_ratio"], 2), "Area (px)": int(o["area"])} for o in unknown]),
                     use_container_width=True, hide_index=True)

    ok, buf = cv2.imencode(".png", out)
    st.download_button("Download analysed image", buf.tobytes(), f"result_{name}.png", "image/png", key=f"dl_{name}")

if all_rows:
    st.divider()
    big = pd.DataFrame(all_rows)
    if len(images) > 1:
        st.markdown("**All images**")
        st.dataframe(big.groupby(["image", "Class"]).size().unstack(fill_value=0), use_container_width=True)
    st.download_button("Download results CSV", big.to_csv(index=False).encode("utf-8"), "results.csv", "text/csv")

st.markdown(
    f"<div style='margin-top:28px;color:#5A6A8A;font-size:12px'>{PROJECT['title']} · {PROJECT['course']} · "
    f"{PROJECT['instructor']} · {PROJECT['university']} · {PROJECT['student']} · Section {PROJECT['section']} · "
    f"ID {PROJECT['student_id']}</div>" + contact_links("contact-main"), unsafe_allow_html=True)
