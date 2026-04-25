"""Generate the sample.pdf fixture used by smoke tests.

Run from the repo root:
    python3 tests/fixtures/_generate_sample.py

Produces tests/fixtures/sample.pdf — a tiny one-page PDF containing a
heading, a paragraph mentioning Acme Corp Q1 2024 revenue, and a small
2-row, 3-column table. Reproducible so the fixture can be regenerated
from version control if needed.
"""
from __future__ import annotations

from pathlib import Path

import fitz  # PyMuPDF

OUTPUT = Path(__file__).parent / "sample.pdf"


def build_pdf(output: Path) -> None:
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)  # US Letter

    # Heading
    page.insert_text(
        (72, 90),
        "Acme Corp Annual Report",
        fontname="helv",
        fontsize=18,
    )

    # Paragraph
    page.insert_text(
        (72, 140),
        "Acme Corp had Q1 2024 revenue of $100 million.",
        fontname="helv",
        fontsize=12,
    )

    # Table: 2 rows (header + 1 data) x 3 cols (Quarter | Revenue | Cost)
    cell_w = 120
    cell_h = 24
    x0 = 72
    y0 = 200

    headers = ["Quarter", "Revenue", "Cost"]
    data_row = ["Q1", "100", "60"]

    # Draw cell borders + text
    for col_idx, value in enumerate(headers):
        rect = fitz.Rect(
            x0 + col_idx * cell_w,
            y0,
            x0 + (col_idx + 1) * cell_w,
            y0 + cell_h,
        )
        page.draw_rect(rect, color=(0, 0, 0), width=0.7)
        page.insert_text(
            (rect.x0 + 6, rect.y0 + 16),
            value,
            fontname="helv",
            fontsize=11,
        )

    for col_idx, value in enumerate(data_row):
        rect = fitz.Rect(
            x0 + col_idx * cell_w,
            y0 + cell_h,
            x0 + (col_idx + 1) * cell_w,
            y0 + 2 * cell_h,
        )
        page.draw_rect(rect, color=(0, 0, 0), width=0.7)
        page.insert_text(
            (rect.x0 + 6, rect.y0 + 16),
            value,
            fontname="helv",
            fontsize=11,
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(output), deflate=True)
    doc.close()


if __name__ == "__main__":
    build_pdf(OUTPUT)
    print(f"Wrote {OUTPUT} ({OUTPUT.stat().st_size} bytes)")
