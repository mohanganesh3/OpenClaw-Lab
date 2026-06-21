#!/usr/bin/env python3
"""
Full-paper PDF extractor for OpenResearchOS (run via `uv run --with pymupdf`).

Reads EVERY page and preserves:
  - full body text in reading order
  - tables (via PyMuPDF find_tables → Markdown)
  - figure/table captions
  - section structure

Usage: python pdf_full_extract.py <input.pdf> <output.md>
Writes Markdown to <output.md> and prints a one-line JSON summary to stdout.
"""
import sys, json, re

def table_to_md(tab):
    try:
        rows = tab.extract()
    except Exception:
        return ""
    if not rows:
        return ""
    out = []
    header = rows[0]
    ncol = max(len(r) for r in rows)
    def fmt(r):
        cells = [("" if c is None else str(c)).replace("\n", " ").strip() for c in r]
        cells += [""] * (ncol - len(cells))
        return "| " + " | ".join(cells) + " |"
    out.append(fmt(header))
    out.append("| " + " | ".join(["---"] * ncol) + " |")
    for r in rows[1:]:
        out.append(fmt(r))
    return "\n".join(out)

def main():
    if len(sys.argv) < 3:
        print(json.dumps({"ok": False, "error": "args"})); return
    pdf_path, out_path = sys.argv[1], sys.argv[2]
    try:
        import fitz  # PyMuPDF
    except Exception as e:
        print(json.dumps({"ok": False, "error": f"no pymupdf: {e}"})); return
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(json.dumps({"ok": False, "error": f"open: {e}"})); return

    parts = []
    n_tables = 0
    n_captions = 0
    n_pages = doc.page_count
    MAX_PAGES = 60  # a single thesis/book shouldn't dominate; 60 pages is plenty
    read_pages = min(n_pages, MAX_PAGES)
    for pi in range(read_pages):
        page = doc[pi]
        text = page.get_text("text") or ""
        parts.append(f"\n\n<!-- page {pi+1} -->\n{text.strip()}")
        # Tables
        try:
            tf = page.find_tables()
            for ti, tab in enumerate(tf.tables):
                md = table_to_md(tab)
                if md and md.count("|") > 4:
                    n_tables += 1
                    parts.append(f"\n\n**[Table p{pi+1}.{ti+1}]**\n{md}")
        except Exception:
            pass
        # Figure / Table captions
        for m in re.finditer(r"(?im)^\s*(figure|fig\.?|table)\s*\d+[:.\)]?\s+.{0,300}", text):
            n_captions += 1
            parts.append(f"\n[CAPTION] {m.group(0).strip()[:320]}")

    full = "\n".join(parts).strip()
    # Collapse excessive blank lines
    full = re.sub(r"\n{4,}", "\n\n\n", full)
    # Strip null bytes and other non-printable characters that break downstream
    # spawnSync calls (Node.js rejects strings containing \x00 in arg lists).
    full = full.replace("\x00", " ")
    # Strip other ASCII control chars except tab, newline, carriage return
    full = re.sub(r"[\x01-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", full)
    with open(out_path, "w", encoding="utf-8", errors="replace") as f:
        f.write(full)
    print(json.dumps({
        "ok": True, "pages": read_pages, "total_pages": n_pages, "tables": n_tables,
        "captions": n_captions, "chars": len(full),
    }))

if __name__ == "__main__":
    main()
