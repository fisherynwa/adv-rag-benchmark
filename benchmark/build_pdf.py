"""Render the fictional Kerrowan annual report to PDF, one content page per PDF page.

If a page's content overflows, the PDF gets more than 25 pages and the check in verify.py fails,
so page placement is guaranteed rather than assumed.
"""
import sys
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer, Table,
                                TableStyle, PageBreak)
from content_a import PAGES_A
from content_b import PAGES_B
from content_c import PAGES_C
from content_extra import EXTRA

PAGES = PAGES_A + PAGES_B + PAGES_C
for _i, _pg in enumerate(PAGES, 1):
    _pg["blocks"] = _pg["blocks"] + EXTRA.get(_i, [])
INK, MUTED, RULE, ACCENT = (colors.HexColor(c) for c in ("#1f2328", "#5b6470", "#d0d7de", "#0b4f6c"))

H1 = ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=17, leading=21, textColor=ACCENT, spaceAfter=7)
H2 = ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=11.5, leading=14, textColor=INK,
                    spaceBefore=6, spaceAfter=3)
P = ParagraphStyle("p", fontName="Helvetica", fontSize=9.8, leading=13.4, textColor=INK, spaceAfter=5.5)
CELL = ParagraphStyle("cell", fontName="Helvetica", fontSize=8.8, leading=11, textColor=INK)
CELL_H = ParagraphStyle("cellh", parent=CELL, fontName="Helvetica-Bold")


def esc(t):
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(20 * mm, 12 * mm, "Kerrowan Marine Systems plc  |  Annual Report 2025")
    canvas.drawRightString(190 * mm, 12 * mm, str(doc.page))
    canvas.restoreState()


def table(rows):
    data = [[Paragraph(esc(c), CELL_H if i == 0 else CELL) for c in r] for i, r in enumerate(rows)]
    t = Table(data, hAlign="LEFT", repeatRows=1)
    t.setStyle(TableStyle([
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, INK),
        ("LINEBELOW", (0, 1), (-1, -1), 0.3, RULE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 2.2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.2),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    return t


def build(out_path):
    doc = BaseDocTemplate(out_path, pagesize=A4, title="Kerrowan Marine Systems plc Annual Report 2025",
                          author="Kerrowan Marine Systems plc (fictional benchmark document)")
    frame = Frame(20 * mm, 18 * mm, 170 * mm, 262 * mm, id="body", leftPadding=0, rightPadding=0,
                  topPadding=0, bottomPadding=0)
    doc.addPageTemplates([PageTemplate(id="page", frames=[frame], onPage=footer)])
    story = []
    for i, page in enumerate(PAGES):
        for kind, content in page["blocks"]:
            if kind == "h1":
                story.append(Paragraph(esc(content), H1))
            elif kind == "h2":
                story.append(Paragraph(esc(content), H2))
            elif kind == "p":
                story.append(Paragraph(esc(content), P))
            elif kind == "table":
                story += [table(content), Spacer(1, 6)]
        if i < len(PAGES) - 1:
            story.append(PageBreak())
    doc.build(story)
    return len(PAGES)


if __name__ == "__main__":
    n = build(sys.argv[1] if len(sys.argv) > 1 else "kerrowan_annual_report_2025.pdf")
    print(f"rendered {n} content pages")
