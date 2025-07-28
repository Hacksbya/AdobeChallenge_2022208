#!/usr/bin/env python3
import json
from pathlib import Path
from collections import Counter, defaultdict
from itertools import groupby
import fitz  # PyMuPDF

BIN_HEIGHT = 5.0     # points to bucket spans into lines
X_THRESH   = 10.0    # points to merge split-heading lines by indentation

def extract_outline(pdf_path):
    doc = fitz.open(pdf_path)
    all_lines = []

    # 1) Extract every text span, bucket by (page, y-bin)
    for pg in doc:
        spans = []
        for block in pg.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    txt = span["text"].strip()
                    if not txt:
                        continue
                    spans.append({
                        "text": txt,
                        "x": span["bbox"][0],
                        "y": span["bbox"][1],
                        "size": span["size"],
                        "bold": bool(span["flags"] & 2),
                        "page": pg.number + 1
                    })

        bins = defaultdict(list)
        for s in spans:
            key = (s["page"], int(s["y"] // BIN_HEIGHT))
            bins[key].append(s)

        # 2) Merge spans in each bin into one "line" record
        for (page, _), grp in bins.items():
            grp.sort(key=lambda s: s["x"])
            tokens = [s["text"] for s in grp]

            # a) prefer single-token fragments if available
            singles = [t for t in tokens if " " not in t]
            toks = singles or tokens

            # b) drop any toks that are strict substrings of others
            filt = [t for t in toks if not any(t != o and t in o for o in toks)]
            toks = filt or toks

            # c) collapse consecutive duplicates
            uniq = [t for t,_ in groupby(toks)]
            clean_text = " ".join(uniq)

            all_lines.append({
                "tokens": tokens,         # raw tokens for possible H1 rebuild
                "text": clean_text,       # cleaned fallback
                "size": max(s["size"] for s in grp),
                "bold": any(s["bold"] for s in grp),
                "page": page,
                "x": sum(s["x"] for s in grp) / len(grp),
                "y": sum(s["y"] for s in grp) / len(grp)
            })

    if not all_lines:
        return "", []

    # 3) Title: metadata or largest clean_text
    meta = doc.metadata.get("title", "").strip()
    if meta and len(meta) > 5:
        title = meta
    else:
        best = max(all_lines, key=lambda L: (L["size"], len(L["text"])))
        title = best["text"]

    # 4) Determine body font size (most common)
    body_size = Counter(L["size"] for L in all_lines).most_common(1)[0][0]

    # 5) Filter heading candidates
    cands = [
        L for L in all_lines
        if L["size"] > body_size + 1
           or (L["bold"] and len(L["tokens"]) <= 7)
    ]

    # 6) Map the top‑3 sizes to H1, H2, H3
    top_sizes = sorted({L["size"] for L in cands}, reverse=True)[:3]
    size2lvl = {sz: f"H{idx+1}" for idx, sz in enumerate(top_sizes)}

    # 7) Build raw outline entries
    raw = []
    for L in sorted(cands, key=lambda L: (L["page"], L["y"])):
        lvl = size2lvl.get(L["size"]) or ("H3" if L["bold"] else None)
        if lvl:
            raw.append({
                "level": lvl,
                "tokens": L["tokens"],
                "text":   L["text"],
                "page":   L["page"],
                "x":      L["x"],
                "y":      L["y"]
            })

    # 8) Merge split-heading lines of same level & page by indentation
    merged = []
    for e in raw:
        if not merged:
            merged.append(e)
            continue
        prev = merged[-1]
        if (
            e["level"] == prev["level"]
            and e["page"] == prev["page"]
            and abs(e["x"] - prev["x"]) < X_THRESH
        ):
            # continuation of the same heading
            prev["tokens"] += e["tokens"]
            prev["text"]   += " " + e["text"]
        else:
            merged.append(e)

    # 9) Final outline: use raw tokens to rebuild H1 exactly, else cleaned text
    outline = []
    for e in merged:
        if e["level"] == "H1":
            # rebuild full H1 from tokens, collapsing exact duplicates
            seq = [t for t,_ in groupby(e["tokens"])]
            text = " ".join(seq)
        else:
            text = e["text"]
        outline.append({
            "level": e["level"],
            "text":  text,
            "page":  e["page"]
        })

    return title, outline

def main():
    IN  = Path("input")
    OUT = Path("output")
    OUT.mkdir(exist_ok=True)
    for pdf in sorted(IN.glob("*.pdf")):
        title, outline = extract_outline(str(pdf))
        # Build final JSON object matching schema
        result = {
            "title": title,
            "outline": [
                {"level": o["level"], "text": o["text"], "page": o["page"]}
                for o in outline
            ]
        }
        with open(OUT / f"{pdf.stem}.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"Processed {pdf.name} → {pdf.stem}.json")

if __name__ == "__main__":
    main()
