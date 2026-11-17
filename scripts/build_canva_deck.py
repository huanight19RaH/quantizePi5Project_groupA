#!/usr/bin/env python3
"""Build Team_A's 28-slide 16:9 editable PPTX presentation for native Canva import."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml import parse_xml
from pptx.oxml.ns import nsdecls, qn
from pptx.util import Inches, Pt

SLIDE_W = Inches(13.333333)
SLIDE_H = Inches(7.5)

THEMES: dict[str, dict[str, str]] = {
    "editorial_light": {
        "background": "#F8FAFC",
        "surface": "#FFFFFF",
        "surface_alt": "#EEF2F6",
        "primary": "#0F172A",
        "accent": "#2563EB",
        "accent_soft": "#DBEAFE",
        "winner_bg": "#ECFDF5",
        "winner_fg": "#065F46",
        "alert_bg": "#FFF1F2",
        "alert_fg": "#9F1239",
        "text_primary": "#0F172A",
        "text_secondary": "#475569",
        "text_on_dark": "#FFFFFF",
        "border": "#CBD5E1",
        "font_heading": "Montserrat",
        "font_body": "Inter",
    },
    "executive_dark": {
        "background": "#0B1120",
        "surface": "#162033",
        "surface_alt": "#1E293B",
        "primary": "#F8FAFC",
        "accent": "#38BDF8",
        "accent_soft": "#0C4A6E",
        "winner_bg": "#064E3B",
        "winner_fg": "#A7F3D0",
        "alert_bg": "#4C0519",
        "alert_fg": "#FECDD3",
        "text_primary": "#F8FAFC",
        "text_secondary": "#CBD5E1",
        "text_on_dark": "#FFFFFF",
        "border": "#334155",
        "font_heading": "Montserrat",
        "font_body": "Inter",
    },
}

PLACEHOLDER_PATTERN = re.compile(
    r"(\bTBD\b|\bTODO\b|\blorem\s+ipsum\b|\[insert[^\]]*\]|\[placeholder[^\]]*\])",
    re.IGNORECASE,
)


def _rgb(hex_color: str) -> RGBColor:
    clean = hex_color.strip().lstrip("#")
    if len(clean) != 6:
        raise ValueError(f"Invalid hex color: {hex_color}")
    return RGBColor(int(clean[0:2], 16), int(clean[2:4], 16), int(clean[4:6], 16))


def _set_solid_fill(shape: Any, hex_color: str, opacity: float = 1.0) -> None:
    shape.fill.solid()
    shape.fill.fore_color.rgb = _rgb(hex_color)
    if opacity < 1.0:
        alpha_val = max(0, min(100000, int(round(opacity * 100000))))
        solid_fill = shape._element.spPr.find(qn("a:solidFill"))
        if solid_fill is not None:
            srgb = solid_fill.find(qn("a:srgbClr"))
            if srgb is not None:
                for old in srgb.findall(qn("a:alpha")):
                    srgb.remove(old)
                srgb.append(parse_xml(f'<a:alpha {nsdecls("a")} val="{alpha_val}"/>'))


def _set_border(shape: Any, hex_color: str | None, width_pt: float = 1.0) -> None:
    if not hex_color:
        shape.line.fill.background()
        return
    shape.line.color.rgb = _rgb(hex_color)
    shape.line.width = Pt(width_pt)


def _add_rect(
    slide: Any,
    left: float,
    top: float,
    width: float,
    height: float,
    fill_hex: str,
    opacity: float = 1.0,
    border_hex: str | None = None,
    border_pt: float = 1.0,
    rounded: bool = False,
) -> Any:
    shape_type = MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE
    shape = slide.shapes.add_shape(shape_type, Inches(left), Inches(top), Inches(width), Inches(height))
    if rounded and hasattr(shape, "adjustments") and len(shape.adjustments) > 0:
        shape.adjustments[0] = 0.035
    _set_solid_fill(shape, fill_hex, opacity=opacity)
    _set_border(shape, border_hex, width_pt=border_pt)
    return shape


def _add_textbox(
    slide: Any,
    left: float,
    top: float,
    width: float,
    height: float,
    text: str | list[str],
    font_name: str,
    font_size_pt: float,
    color_hex: str,
    bold: bool = False,
    italic: bool = False,
    align: PP_ALIGN = PP_ALIGN.LEFT,
    valign: MSO_ANCHOR = MSO_ANCHOR.TOP,
    line_spacing: float = 1.18,
    bullet_prefix: str = "",
    space_after_pt: float = 5.0,
    warnings: list[str] | None = None,
    slide_idx: int = 0,
    field_label: str = "text",
) -> Any:
    box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = valign
    tf.margin_left = Inches(0.05)
    tf.margin_right = Inches(0.05)
    tf.margin_top = Inches(0.03)
    tf.margin_bottom = Inches(0.03)

    lines = text if isinstance(text, list) else [text]
    lines = [str(x).strip() for x in lines if str(x).strip()] or [""]

    for i, line in enumerate(lines):
        if PLACEHOLDER_PATTERN.search(line):
            raise ValueError(f"Slide {slide_idx} ({field_label}) contains forbidden placeholder: {line!r}")
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = line_spacing
        p.space_after = Pt(space_after_pt if i < len(lines) - 1 else 0)
        run = p.add_run()
        run.text = f"{bullet_prefix}{line}" if bullet_prefix else line
        run.font.name = font_name
        run.font.size = Pt(font_size_pt)
        run.font.bold = bold
        run.font.italic = italic
        run.font.color.rgb = _rgb(color_hex)

    if warnings is not None and lines != [""]:
        usable_w_pt = max(36.0, (width - 0.1) * 72.0)
        usable_h_pt = max(14.0, (height - 0.06) * 72.0)
        chars_per_line = max(10.0, usable_w_pt / (font_size_pt * 0.53))
        total_lines = sum(
            max(1, math.ceil(len(f"{bullet_prefix}{ln}" if bullet_prefix else ln) / chars_per_line))
            for ln in lines
        )
        est_height_pt = total_lines * font_size_pt * line_spacing + max(0, len(lines) - 1) * space_after_pt
        if est_height_pt > usable_h_pt * 1.15:
            warnings.append(
                f"Slide {slide_idx} '{field_label}' overflow risk ({est_height_pt:.0f}pt > {usable_h_pt:.0f}pt)"
            )
    return box


def _validate_image(path_str: str, slide_idx: int) -> tuple[Path, int, int]:
    img_path = Path(path_str)
    if not img_path.is_file():
        raise FileNotFoundError(f"Slide {slide_idx}: image file not found: {img_path}")
    with Image.open(img_path) as probe:
        probe.verify()
    with Image.open(img_path) as img:
        w, h = img.size
    if w < 300 or h < 200:
        raise ValueError(f"Slide {slide_idx}: image '{img_path.name}' ({w}x{h}) too small.")
    return img_path, w, h


def _add_real_image(
    slide: Any,
    image_path: str,
    left: float,
    top: float,
    width: float,
    height: float,
    fit: str = "contain",
    slide_idx: int = 0,
) -> dict[str, Any]:
    """Place verified raster chart or real photo cleanly with 'contain' or non-destructive 'cover' crop."""
    img_path, nat_w, nat_h = _validate_image(image_path, slide_idx)
    nat_ratio = nat_w / nat_h
    box_ratio = width / height
    if fit == "contain":
        if nat_ratio >= box_ratio:
            draw_w = width
            draw_h = width / nat_ratio
            draw_left = left
            draw_top = top + (height - draw_h) / 2.0
        else:
            draw_h = height
            draw_w = height * nat_ratio
            draw_left = left + (width - draw_w) / 2.0
            draw_top = top
        slide.shapes.add_picture(str(img_path), Inches(draw_left), Inches(draw_top), Inches(draw_w), Inches(draw_h))
    else:
        pic = slide.shapes.add_picture(str(img_path), Inches(left), Inches(top), Inches(width), Inches(height))
        if nat_ratio > box_ratio:
            crop = (1.0 - box_ratio / nat_ratio) / 2.0
            pic.crop_left = crop
            pic.crop_right = crop
        elif nat_ratio < box_ratio:
            crop = (1.0 - nat_ratio / box_ratio) / 2.0
            pic.crop_top = crop
            pic.crop_bottom = crop
    return {"path": str(img_path), "width": nat_w, "height": nat_h}


def _add_header(
    slide: Any,
    spec: dict[str, Any],
    theme: dict[str, str],
    warnings: list[str],
    slide_idx: int,
    max_width: float = 11.73,
) -> float:
    y = 0.40
    kicker = (spec.get("kicker") or "").strip()
    if kicker:
        _add_textbox(
            slide, 0.8, y, max_width, 0.28, kicker.upper(),
            theme["font_body"], 10.5, theme["accent"], bold=True,
            warnings=warnings, slide_idx=slide_idx, field_label="kicker",
        )
        y += 0.27
    title = (spec.get("title") or "").strip()
    if not title:
        raise ValueError(f"Slide {slide_idx}: missing required 'title'.")
    _add_textbox(
        slide, 0.8, y, max_width, 0.56, title,
        theme["font_heading"], 23, theme["text_primary"], bold=True, line_spacing=1.10,
        warnings=warnings, slide_idx=slide_idx, field_label="title",
    )
    y += 0.56
    subtitle = (spec.get("subtitle") or "").strip()
    if subtitle:
        _add_textbox(
            slide, 0.8, y, max_width, 0.46, subtitle,
            theme["font_body"], 12.2, theme["text_secondary"], line_spacing=1.15,
            warnings=warnings, slide_idx=slide_idx, field_label="subtitle",
        )
        y += 0.47
    return y + 0.08


def _add_footer(slide: Any, slide_idx: int, total: int, deck_title: str, theme: dict[str, str], on_dark: bool = False) -> None:
    color = "#E2E8F0" if on_dark else theme["text_secondary"]
    if deck_title:
        _add_textbox(slide, 0.8, 7.06, 10.0, 0.28, deck_title, theme["font_body"], 9.5, color, slide_idx=slide_idx, field_label="footer_title")
    _add_textbox(slide, 11.3, 7.06, 1.23, 0.28, f"{slide_idx} / {total}", theme["font_body"], 9.5, color, align=PP_ALIGN.RIGHT, slide_idx=slide_idx, field_label="footer_page")


def _render_hero_cover(slide: Any, spec: dict[str, Any], theme: dict[str, str], warnings: list[str], idx: int, total: int, deck_title: str, used: list[dict[str, Any]]) -> None:
    used.append(_add_real_image(slide, spec["image_path"], 0.0, 0.0, 13.333333, 7.5, fit=spec.get("image_fit", "cover"), slide_idx=idx))
    _add_rect(slide, 0.0, 0.0, 13.333333, 7.5, "#0B1120", opacity=float(spec.get("scrim_opacity", 0.72)))
    _add_rect(slide, 0.85, 1.75, 0.11, 3.65, theme["accent"])
    if spec.get("kicker"):
        _add_textbox(slide, 1.18, 1.72, 10.6, 0.35, spec["kicker"].upper(), theme["font_body"], 12.5, "#38BDF8", bold=True, warnings=warnings, slide_idx=idx, field_label="kicker")
    _add_textbox(slide, 1.18, 2.15, 11.1, 1.50, spec["title"], theme["font_heading"], 36, "#FFFFFF", bold=True, line_spacing=1.12, warnings=warnings, slide_idx=idx, field_label="title")
    if spec.get("subtitle"):
        _add_textbox(slide, 1.18, 3.78, 10.8, 1.25, spec["subtitle"], theme["font_body"], 16, "#E2E8F0", line_spacing=1.28, warnings=warnings, slide_idx=idx, field_label="subtitle")
    if spec.get("meta"):
        _add_rect(slide, 1.18, 5.75, 10.8, 0.56, "#1E293B", opacity=0.88, border_hex="#334155", rounded=True)
        _add_textbox(slide, 1.38, 5.83, 10.4, 0.40, spec["meta"], theme["font_body"], 11.5, "#CBD5E1", valign=MSO_ANCHOR.MIDDLE, warnings=warnings, slide_idx=idx, field_label="meta")


def _render_chart_hero(slide: Any, spec: dict[str, Any], theme: dict[str, str], warnings: list[str], idx: int, total: int, deck_title: str, used: list[dict[str, Any]]) -> None:
    _add_rect(slide, 0.0, 0.0, 13.333333, 7.5, theme["background"])
    top_y = _add_header(slide, spec, theme, warnings, idx)
    cards = spec.get("cards") or []
    card_zone_h = 1.25 if cards else 0.0
    gap_below = 0.16 if cards else 0.0
    chart_h = 6.90 - top_y - card_zone_h - gap_below

    _add_rect(slide, 0.8, top_y, 11.73, chart_h, theme["surface"], border_hex=theme["border"], border_pt=1.0, rounded=True)
    used.append(_add_real_image(slide, spec["image_path"], 0.90, top_y + 0.06, 11.53, chart_h - 0.12, fit=spec.get("image_fit", "contain"), slide_idx=idx))

    if cards:
        cy = top_y + chart_h + gap_below
        n = len(cards)
        gap = 0.24
        cw = (11.73 - gap * (n - 1)) / n
        for ci, c in enumerate(cards):
            cx = 0.8 + ci * (cw + gap)
            hl = c.get("highlight")
            bg = theme["winner_bg"] if hl == "winner" else theme["alert_bg"] if hl == "alert" else theme["surface"]
            fg_tag = theme["winner_fg"] if hl == "winner" else theme["alert_fg"] if hl == "alert" else theme["accent"]
            _add_rect(slide, cx, cy, cw, card_zone_h, bg, border_hex=theme["border"], rounded=True)
            _add_rect(slide, cx, cy + 0.14, 0.07, card_zone_h - 0.28, fg_tag)
            if c.get("title"):
                _add_textbox(slide, cx + 0.18, cy + 0.10, cw - 0.28, 0.30, c["title"], theme["font_heading"], 11.5, fg_tag, bold=True, warnings=warnings, slide_idx=idx, field_label=f"card_{ci+1}_t")
            _add_textbox(slide, cx + 0.18, cy + 0.40, cw - 0.28, card_zone_h - 0.48, c.get("body", ""), theme["font_body"], 10.5, theme["text_primary"], line_spacing=1.16, warnings=warnings, slide_idx=idx, field_label=f"card_{ci+1}_b")
    _add_footer(slide, idx, total, deck_title, theme)


def _render_chart_split(slide: Any, spec: dict[str, Any], theme: dict[str, str], warnings: list[str], idx: int, total: int, deck_title: str, used: list[dict[str, Any]], chart_side: str) -> None:
    _add_rect(slide, 0.0, 0.0, 13.333333, 7.5, theme["background"])
    top_y = _add_header(slide, spec, theme, warnings, idx)
    panel_h = 6.88 - top_y
    chart_w = 7.45
    info_w = 4.00
    if chart_side == "left":
        chart_x, info_x = 0.8, 0.8 + chart_w + 0.28
    else:
        info_x, chart_x = 0.8, 0.8 + info_w + 0.28

    _add_rect(slide, chart_x, top_y, chart_w, panel_h, theme["surface"], border_hex=theme["border"], rounded=True)
    used.append(_add_real_image(slide, spec["image_path"], chart_x + 0.08, top_y + 0.08, chart_w - 0.16, panel_h - 0.16, fit=spec.get("image_fit", "contain"), slide_idx=idx))

    _add_rect(slide, info_x, top_y, info_w, panel_h, theme["surface"], border_hex=theme["border"], rounded=True)
    _add_rect(slide, info_x, top_y, info_w, 0.07, theme["accent"])
    iy = top_y + 0.18

    metrics = spec.get("metrics") or []
    if metrics:
        mcols = min(2, len(metrics))
        mw = (info_w - 0.48 - 0.16 * (mcols - 1)) / mcols
        for mi, m in enumerate(metrics):
            r, col_i = divmod(mi, mcols)
            mx = info_x + 0.20 + col_i * (mw + 0.16)
            my = iy + r * 0.92
            _add_rect(slide, mx, my, mw, 0.82, theme["surface_alt"], rounded=True)
            _add_textbox(slide, mx + 0.10, my + 0.06, mw - 0.20, 0.36, str(m["value"]), theme["font_heading"], 18, theme["accent"], bold=True, warnings=warnings, slide_idx=idx, field_label=f"m_{mi+1}_v")
            _add_textbox(slide, mx + 0.10, my + 0.44, mw - 0.20, 0.32, str(m["label"]), theme["font_body"], 9.5, theme["text_secondary"], line_spacing=1.10, warnings=warnings, slide_idx=idx, field_label=f"m_{mi+1}_l")
        iy += math.ceil(len(metrics) / mcols) * 0.92 + 0.10

    if spec.get("panel_title"):
        _add_textbox(slide, info_x + 0.20, iy, info_w - 0.40, 0.48, spec["panel_title"], theme["font_heading"], 13.0, theme["text_primary"], bold=True, line_spacing=1.12, warnings=warnings, slide_idx=idx, field_label="panel_title")
        iy += 0.50

    bullets = spec.get("bullets") or []
    if bullets:
        rem_h = max(1.2, (top_y + panel_h) - iy - 0.15)
        _add_textbox(slide, info_x + 0.20, iy, info_w - 0.40, rem_h, bullets, theme["font_body"], 11.2, theme["text_primary"], line_spacing=1.22, bullet_prefix="•  ", space_after_pt=7.0, warnings=warnings, slide_idx=idx, field_label="bullets")

    _add_footer(slide, idx, total, deck_title, theme)


def _render_dual_visuals(slide: Any, spec: dict[str, Any], theme: dict[str, str], warnings: list[str], idx: int, total: int, deck_title: str, used: list[dict[str, Any]]) -> None:
    _add_rect(slide, 0.0, 0.0, 13.333333, 7.5, theme["background"])
    top_y = _add_header(slide, spec, theme, warnings, idx)
    items = spec.get("visuals") or []
    takeaway = (spec.get("takeaway") or "").strip()
    bot_h = 0.96 if takeaway else 0.0
    gap_bot = 0.16 if takeaway else 0.0
    box_h = 6.88 - top_y - bot_h - gap_bot
    w = (11.73 - 0.28) / 2.0

    for vi, item in enumerate(items[:2]):
        vx = 0.8 + vi * (w + 0.28)
        _add_rect(slide, vx, top_y, w, box_h, theme["surface"], border_hex=theme["border"], rounded=True)
        img_top = top_y + 0.44 if item.get("title") else top_y + 0.08
        if item.get("title"):
            _add_textbox(slide, vx + 0.16, top_y + 0.08, w - 0.32, 0.32, item["title"], theme["font_heading"], 12.5, theme["accent"], bold=True, warnings=warnings, slide_idx=idx, field_label=f"vis_{vi+1}_t")
        img_h = (top_y + box_h - 0.44 - img_top) if item.get("caption") else (top_y + box_h - 0.10 - img_top)
        used.append(_add_real_image(slide, item["image_path"], vx + 0.12, img_top, w - 0.24, img_h, fit=item.get("image_fit", "contain"), slide_idx=idx))
        if item.get("caption"):
            _add_textbox(slide, vx + 0.16, top_y + box_h - 0.40, w - 0.32, 0.32, item["caption"], theme["font_body"], 10.0, theme["text_secondary"], italic=True, warnings=warnings, slide_idx=idx, field_label=f"vis_{vi+1}_cap")

    if takeaway:
        ty = top_y + box_h + gap_bot
        _add_rect(slide, 0.8, ty, 11.73, bot_h, theme["surface_alt"], border_hex=theme["border"], rounded=True)
        _add_rect(slide, 0.8, ty, 0.08, bot_h, theme["accent"])
        _add_textbox(slide, 1.02, ty + 0.10, 11.35, bot_h - 0.20, takeaway, theme["font_body"], 11.5, theme["text_primary"], bold=False, valign=MSO_ANCHOR.MIDDLE, warnings=warnings, slide_idx=idx, field_label="takeaway")
    _add_footer(slide, idx, total, deck_title, theme)


def _render_editable_table(slide: Any, spec: dict[str, Any], theme: dict[str, str], warnings: list[str], idx: int, total: int, deck_title: str, used: list[dict[str, Any]]) -> None:
    _add_rect(slide, 0.0, 0.0, 13.333333, 7.5, theme["background"])
    top_y = _add_header(slide, spec, theme, warnings, idx)
    cols: list[dict[str, Any]] = spec["columns"]
    rows: list[dict[str, Any]] = spec["rows"]
    sidebar = spec.get("sidebar")
    takeaway = (spec.get("takeaway") or "").strip()

    bot_h = 0.72 if takeaway else 0.0
    gap_bot = 0.14 if takeaway else 0.0
    table_h = 6.88 - top_y - bot_h - gap_bot
    table_w = 8.45 if sidebar else 11.73

    total_weight = sum(float(c.get("weight", 1.0)) for c in cols)
    col_widths = [table_w * float(c.get("weight", 1.0)) / total_weight for c in cols]

    hdr_h = min(0.52, table_h * 0.14)
    row_h = (table_h - hdr_h) / max(1, len(rows))
    font_sz = 10.5 if len(rows) <= 7 else 9.6

    # Table Header
    _add_rect(slide, 0.8, top_y, table_w, hdr_h, theme["primary"])
    cx = 0.8
    for ci, col in enumerate(cols):
        cw = col_widths[ci]
        align = PP_ALIGN.LEFT if ci == 0 else PP_ALIGN.CENTER
        _add_textbox(slide, cx + 0.04, top_y + 0.04, cw - 0.08, hdr_h - 0.08, str(col["label"]), theme["font_heading"], font_sz, theme["text_on_dark"], bold=True, align=align, valign=MSO_ANCHOR.MIDDLE, warnings=warnings, slide_idx=idx, field_label=f"th_{ci}")
        cx += cw

    # Table Rows
    ry = top_y + hdr_h
    for ri, row in enumerate(rows):
        hl = row.get("highlight")
        if hl == "winner":
            bg, fg, is_b = theme["winner_bg"], theme["winner_fg"], True
        elif hl == "alert":
            bg, fg, is_b = theme["alert_bg"], theme["alert_fg"], True
        else:
            bg = theme["surface"] if ri % 2 == 0 else theme["surface_alt"]
            fg, is_b = theme["text_primary"], False
        _add_rect(slide, 0.8, ry, table_w, row_h, bg, border_hex=theme["border"], border_pt=0.5)
        cx = 0.8
        for ci, val in enumerate(row["cells"]):
            cw = col_widths[ci]
            align = PP_ALIGN.LEFT if ci == 0 else PP_ALIGN.CENTER
            _add_textbox(slide, cx + 0.04, ry + 0.02, cw - 0.08, row_h - 0.04, str(val), theme["font_body"], font_sz, fg, bold=(is_b or ci == 0), align=align, valign=MSO_ANCHOR.MIDDLE, line_spacing=1.08, warnings=warnings, slide_idx=idx, field_label=f"r{ri}_c{ci}")
            cx += cw
        ry += row_h

    if sidebar:
        sx = 0.8 + table_w + 0.25
        sw = 12.53 - sx
        _add_rect(slide, sx, top_y, sw, table_h, theme["surface"], border_hex=theme["border"], rounded=True)
        _add_rect(slide, sx, top_y, sw, 0.07, theme["accent"])
        sy = top_y + 0.16
        if sidebar.get("image_path"):
            ih = 2.15
            used.append(_add_real_image(slide, sidebar["image_path"], sx + 0.14, sy, sw - 0.28, ih, fit="contain", slide_idx=idx))
            sy += ih + 0.14
        if sidebar.get("title"):
            _add_textbox(slide, sx + 0.16, sy, sw - 0.32, 0.34, sidebar["title"], theme["font_heading"], 13, theme["accent"], bold=True, warnings=warnings, slide_idx=idx, field_label="sb_title")
            sy += 0.36
        if sidebar.get("bullets"):
            _add_textbox(slide, sx + 0.16, sy, sw - 0.32, max(1.0, top_y + table_h - sy - 0.12), sidebar["bullets"], theme["font_body"], 10.8, theme["text_primary"], line_spacing=1.2, bullet_prefix="•  ", space_after_pt=6.0, warnings=warnings, slide_idx=idx, field_label="sb_bullets")

    if takeaway:
        ty = top_y + table_h + gap_bot
        _add_rect(slide, 0.8, ty, 11.73, bot_h, theme["winner_bg"], border_hex=theme["border"], rounded=True)
        _add_textbox(slide, 1.00, ty + 0.08, 11.33, bot_h - 0.16, takeaway, theme["font_body"], 11.2, theme["winner_fg"], bold=True, valign=MSO_ANCHOR.MIDDLE, warnings=warnings, slide_idx=idx, field_label="takeaway")

    _add_footer(slide, idx, total, deck_title, theme)


def _render_cards_grid(slide: Any, spec: dict[str, Any], theme: dict[str, str], warnings: list[str], idx: int, total: int, deck_title: str, used: list[dict[str, Any]]) -> None:
    _add_rect(slide, 0.0, 0.0, 13.333333, 7.5, theme["background"])
    top_y = _add_header(slide, spec, theme, warnings, idx)
    cards = spec["cards"]
    n = len(cards)
    gap = 0.26
    has_bot_img = bool(spec.get("image_path"))
    cw = (11.73 - gap * (n - 1)) / n if not has_bot_img else (7.95 - gap * (n - 1)) / n
    ch = 6.86 - top_y
    for ci, card in enumerate(cards):
        cx = 0.8 + ci * (cw + gap)
        hl = card.get("highlight")
        bg = theme["winner_bg"] if hl == "winner" else theme["alert_bg"] if hl == "alert" else theme["surface"]
        ac = theme["winner_fg"] if hl == "winner" else theme["alert_fg"] if hl == "alert" else theme["accent"]
        _add_rect(slide, cx, top_y, cw, ch, bg, border_hex=theme["border"], rounded=True)
        _add_rect(slide, cx, top_y, cw, 0.07, ac)
        iy = top_y + 0.18
        if card.get("image_path"):
            ih = 1.95
            used.append(_add_real_image(slide, card["image_path"], cx + 0.14, iy, cw - 0.28, ih, fit=card.get("image_fit", "contain"), slide_idx=idx))
            iy += ih + 0.14
        if card.get("tag"):
            _add_textbox(slide, cx + 0.18, iy, cw - 0.36, 0.26, card["tag"].upper(), theme["font_body"], 9.5, ac, bold=True, warnings=warnings, slide_idx=idx, field_label=f"c{ci}_tag")
            iy += 0.28
        if card.get("title"):
            _add_textbox(slide, cx + 0.18, iy, cw - 0.36, 0.56, card["title"], theme["font_heading"], 14.5 if n <= 3 else 13.0, theme["text_primary"], bold=True, line_spacing=1.12, warnings=warnings, slide_idx=idx, field_label=f"c{ci}_t")
            iy += 0.58
        body = card.get("bullets") or card.get("body") or ""
        if body:
            _add_textbox(slide, cx + 0.18, iy, cw - 0.36, max(1.0, top_y + ch - iy - 0.14), body, theme["font_body"], 11.5 if n <= 3 else 10.8, theme["text_secondary"] if isinstance(body, str) else theme["text_primary"], line_spacing=1.22, bullet_prefix="•  " if isinstance(body, list) else "", space_after_pt=6.0, warnings=warnings, slide_idx=idx, field_label=f"c{ci}_b")

    if has_bot_img:
        px = 0.8 + 7.95 + 0.26
        pw = 12.53 - px
        _add_rect(slide, px, top_y, pw, ch, theme["surface"], border_hex=theme["border"], rounded=True)
        used.append(_add_real_image(slide, spec["image_path"], px + 0.12, top_y + 0.12, pw - 0.24, ch - 0.68, fit=spec.get("image_fit", "contain"), slide_idx=idx))
        if spec.get("image_caption"):
            _add_textbox(slide, px + 0.14, top_y + ch - 0.50, pw - 0.28, 0.42, spec["image_caption"], theme["font_body"], 9.5, theme["text_secondary"], italic=True, align=PP_ALIGN.CENTER, warnings=warnings, slide_idx=idx, field_label="img_cap")
    _add_footer(slide, idx, total, deck_title, theme)


def _render_comparison(slide: Any, spec: dict[str, Any], theme: dict[str, str], warnings: list[str], idx: int, total: int, deck_title: str, used: list[dict[str, Any]]) -> None:
    _add_rect(slide, 0.0, 0.0, 13.333333, 7.5, theme["background"])
    top_y = _add_header(slide, spec, theme, warnings, idx)
    takeaway = (spec.get("takeaway") or "").strip()
    bot_h = 0.78 if takeaway else 0.0
    gap_bot = 0.14 if takeaway else 0.0
    col_h = 6.88 - top_y - bot_h - gap_bot
    cols = [spec["left_col"], spec["right_col"]]
    gap = 0.32
    cw = (11.73 - gap) / 2.0
    for ci, col in enumerate(cols):
        cx = 0.8 + ci * (cw + gap)
        hdr_bg = theme["text_secondary"] if ci == 0 else theme["accent"]
        _add_rect(slide, cx, top_y, cw, col_h, theme["surface"], border_hex=theme["border"], rounded=True)
        _add_rect(slide, cx, top_y, cw, 0.52, hdr_bg)
        _add_textbox(slide, cx + 0.22, top_y + 0.08, cw - 0.44, 0.36, col["title"], theme["font_heading"], 15, theme["text_on_dark"], bold=True, valign=MSO_ANCHOR.MIDDLE, warnings=warnings, slide_idx=idx, field_label=f"col_{ci}_t")
        _add_textbox(slide, cx + 0.24, top_y + 0.68, cw - 0.48, col_h - 0.84, col["bullets"], theme["font_body"], 12.8, theme["text_primary"], line_spacing=1.28, bullet_prefix="•  ", space_after_pt=8.0, warnings=warnings, slide_idx=idx, field_label=f"col_{ci}_b")
    if takeaway:
        ty = top_y + col_h + gap_bot
        _add_rect(slide, 0.8, ty, 11.73, bot_h, theme["winner_bg"], border_hex=theme["border"], rounded=True)
        _add_textbox(slide, 1.0, ty + 0.08, 11.33, bot_h - 0.16, takeaway, theme["font_body"], 11.5, theme["winner_fg"], bold=True, valign=MSO_ANCHOR.MIDDLE, warnings=warnings, slide_idx=idx, field_label="takeaway")
    _add_footer(slide, idx, total, deck_title, theme)


def _render_timeline(slide: Any, spec: dict[str, Any], theme: dict[str, str], warnings: list[str], idx: int, total: int, deck_title: str, used: list[dict[str, Any]]) -> None:
    _add_rect(slide, 0.0, 0.0, 13.333333, 7.5, theme["background"])
    top_y = _add_header(slide, spec, theme, warnings, idx)
    steps = spec["steps"]
    n = len(steps)
    gap = 0.20
    sw = (11.73 - gap * (n - 1)) / n
    sh = 6.86 - top_y
    _add_rect(slide, 0.8 + sw / 2, top_y + 0.22, 11.73 - sw, 0.05, theme["border"])
    for si, st in enumerate(steps):
        sx = 0.8 + si * (sw + gap)
        bw = 1.05
        _add_rect(slide, sx + (sw - bw) / 2, top_y, bw, 0.44, theme["accent"], rounded=True)
        _add_textbox(slide, sx + (sw - bw) / 2, top_y, bw, 0.44, str(st.get("step") or f"P{si}"), theme["font_heading"], 11.5, theme["text_on_dark"], bold=True, align=PP_ALIGN.CENTER, valign=MSO_ANCHOR.MIDDLE, slide_idx=idx, field_label=f"st_{si}_b")
        cy = top_y + 0.58
        _add_rect(slide, sx, cy, sw, sh - 0.58, theme["surface"], border_hex=theme["border"], rounded=True)
        _add_textbox(slide, sx + 0.14, cy + 0.14, sw - 0.28, 0.56, st["title"], theme["font_heading"], 13.0, theme["text_primary"], bold=True, line_spacing=1.12, warnings=warnings, slide_idx=idx, field_label=f"st_{si}_t")
        _add_textbox(slide, sx + 0.14, cy + 0.74, sw - 0.28, sh - 1.44, st["body"], theme["font_body"], 10.8, theme["text_secondary"], line_spacing=1.22, warnings=warnings, slide_idx=idx, field_label=f"st_{si}_d")
    _add_footer(slide, idx, total, deck_title, theme)


def build_deck(spec: dict[str, Any], out_path: Path) -> dict[str, Any]:
    slides_spec = spec.get("slides") or []
    theme = dict(THEMES[str(spec.get("theme_preset") or "editorial_light")])
    prs = Presentation()
    prs.slide_width = SLIDE_W
    prs.slide_height = SLIDE_H
    blank = prs.slide_layouts[6]

    deck_title = str(spec.get("title") or "").strip()
    total = len(slides_spec)
    warnings: list[str] = []
    used: list[dict[str, Any]] = []

    renderers = {
        "hero_cover": _render_hero_cover,
        "chart_hero": _render_chart_hero,
        "chart_split_left": lambda sl, s, t, w, i, tot, dt, u: _render_chart_split(sl, s, t, w, i, tot, dt, u, "left"),
        "chart_split_right": lambda sl, s, t, w, i, tot, dt, u: _render_chart_split(sl, s, t, w, i, tot, dt, u, "right"),
        "dual_visuals": _render_dual_visuals,
        "editable_table": _render_editable_table,
        "cards_grid": _render_cards_grid,
        "comparison": _render_comparison,
        "timeline": _render_timeline,
    }

    for i, s in enumerate(slides_spec, start=1):
        layout = s["layout"]
        if layout not in renderers:
            raise ValueError(f"Slide {i}: unknown layout '{layout}'")
        slide = prs.slides.add_slide(blank)
        renderers[layout](slide, s, theme, warnings, i, total, deck_title, used)

        notes: list[str] = []
        if s.get("speaker_notes"):
            notes.append(str(s["speaker_notes"]).strip())
        if s.get("image_source"):
            notes.append(f"Visual source: {s['image_source']}")
        for vi, v in enumerate(s.get("visuals") or [], start=1):
            if v.get("image_source"):
                notes.append(f"Panel {vi} source: {v['image_source']}")
        if notes:
            slide.notes_slide.notes_text_frame.text = "\n\n".join(notes)

    if not used:
        raise ValueError("Deck must embed verified charts/photos.")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(out_path))
    return {
        "output_pptx": str(out_path.resolve()),
        "slide_count": total,
        "visuals_placed": len(used),
        "warnings": warnings,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Build Team_A 28-slide editable Canva deck.")
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    report = build_deck(spec, Path(args.out))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
