"""PDF sanitisation (A4): native text, scanned pages and mixed pages.

Genuine redaction
-----------------
Findings are applied with PyMuPDF redaction annotations followed by ``apply_redactions`` using
``PDF_REDACT_TEXT_REMOVE`` (glyphs are removed from the content stream) and
``PDF_REDACT_IMAGE_PIXELS`` (the pixels of any image under the box are overwritten). A black
rectangle drawn on top would NOT be accepted as sanitisation - verification re-extracts the
output with independent tools and fails the artifact if the target is still recoverable.

Coordinates
-----------
Regions are stored in ``pdf_unrot_tl`` (PyMuPDF text/annotation space: unrotated page, origin
top-left, points). OCR boxes travel render_px -> (prepare steps) -> pdf_unrot_tl through an
explicit TransformChain (see coords.pdf_page_chain), so rotated pages are handled correctly.
"""
from __future__ import annotations

import io
import json
import re
import time

try:
    import pymupdf
except Exception:                                  # pragma: no cover - older installs
    import fitz as pymupdf                         # type: ignore

from PIL import Image

from .context import RunContext
from .coords import Rect, pdf_page_chain, rect_overlap_fraction, rect_pad, rect_union, rect_norm
from .detect import finding_to_region, make_replacement, scan_text
from .imaging import (FaceDetectorUnavailable, OcrTimeout, OcrUnavailable, detect_faces, ocr_oriented, ocr_words,
                      prepare_for_ocr, span_to_boxes, words_to_text)
from .regions import Region, group_regions, make_region_id, merge_line_rects
from .safety import DeadlineExceeded, InputRejected

SPACE = "pdf_unrot_tl"
MIN_NATIVE_CHARS = 25
IMAGE_AREA_MIN = 0.01


def open_pdf(path: str, rc_limits) -> "pymupdf.Document":
    try:
        doc = pymupdf.open(path)
    except Exception as exc:
        raise InputRejected("malformed", f"PDF could not be opened: {exc}") from exc
    if doc.needs_pass or doc.is_encrypted:
        doc.close()
        raise InputRejected("encrypted", "PDF is encrypted / password-protected")
    if not doc.is_pdf:
        doc.close()
        raise InputRejected("unsupported_type", "document is not a PDF")
    if doc.page_count > rc_limits.max_pages:
        n = doc.page_count
        doc.close()
        raise InputRejected("too_many_pages", f"{n} pages exceeds limit {rc_limits.max_pages}")
    if doc.page_count == 0:
        doc.close()
        raise InputRejected("malformed", "PDF has no pages")
    return doc


# --------------------------------------------------------------------------- native text
def page_text_and_chars(page) -> tuple[str, list[tuple[Rect | None, int]], list[tuple[str, Rect]], int]:
    """-> (text, per-char (rect|None, line_no), words[(text, rect)], white_text_chars).

    No mediabox clipping: off-page text is still text the file carries.
    """
    d = page.get_text("rawdict", flags=pymupdf.TEXT_PRESERVE_WHITESPACE)
    text: list[str] = []
    cmap: list[tuple[Rect | None, int]] = []
    words: list[tuple[str, Rect]] = []
    white = 0
    line_no = 0
    for b in d.get("blocks", []):
        if b.get("type") != 0:
            continue
        for ln in b.get("lines", []):
            cur: list[tuple[str, Rect]] = []
            for sp in ln.get("spans", []):
                is_white = sp.get("color") == 0xFFFFFF
                for ch in sp.get("chars", []):
                    c = ch["c"]
                    r = tuple(ch["bbox"])
                    text.append(c)
                    cmap.append((r, line_no))
                    white += is_white and not c.isspace()
                    if c.isspace():
                        if cur:
                            words.append(("".join(x for x, _ in cur), rect_union(y for _, y in cur)))
                            cur = []
                    else:
                        cur.append((c, r))
            if cur:
                words.append(("".join(x for x, _ in cur), rect_union(y for _, y in cur)))
            text.append("\n")
            cmap.append((None, line_no))
            line_no += 1
        text.append("\n")
        cmap.append((None, line_no))
    return "".join(text), cmap, words, white


def _native_regions(rc: RunContext, page, pno: int, text: str, cmap) -> list[Region]:
    out: list[Region] = []
    unit = f"page:{pno + 1}"
    size = (page.mediabox.width, page.mediabox.height)
    for f in scan_text(rc.detect, text, "pdf"):
        by_line: dict[int, list[Rect]] = {}
        for i in range(f.start, min(f.end, len(cmap))):
            r, ln = cmap[i]
            if r is not None:
                by_line.setdefault(ln, []).append(r)
        rects = [rect_union(v) for _k, v in sorted(by_line.items())]
        if not rects:
            continue
        reg = finding_to_region(rc.detect, f, unit_id=unit, space=SPACE, rects=rects, source_method="native_text",
                                anchor={"page": pno + 1}, page_size=size)
        out.append(reg)
    return out


# --------------------------------------------------------------------------- OCR
def _render(page, dpi: int, max_pixels: int) -> Image.Image:
    rect = page.rect
    est = (rect.width * dpi / 72) * (rect.height * dpi / 72)
    if est > max_pixels:                              # lower the DPI instead of refusing the page
        dpi = max(72, int(dpi * (max_pixels / est) ** 0.5))
    pix = page.get_pixmap(dpi=dpi, alpha=False)
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples), dpi


def _ocr_regions(rc: RunContext, page, pno: int, native_text_chars: int) -> tuple[list[Region], list[Region]]:
    """OCR (and optional face detection) on one rendered page. Returns (text_regions, face_regions)."""
    texts: list[Region] = []
    faces: list[Region] = []
    unit = f"page:{pno + 1}"
    left = rc.ocr_budget_left()
    if left <= 0.5:
        rc.counters["ocr_skipped_budget"] += 1
        rc.payloads.add("scanned_images", "failed", 1, "OCR time budget exhausted before this page")
        return texts, faces
    img, dpi = _render(page, rc.limits.ocr_dpi, rc.limits.max_image_pixels)
    w, h = page.mediabox.width, page.mediabox.height
    page_chain = pdf_page_chain(w, h, page.rotation, dpi)
    scanned = native_text_chars < MIN_NATIVE_CHARS
    prep = None
    words = None
    chain_desc: list = []
    try:
        if rc.policy.ocr:
            if not rc.ocr_ready():
                rc.payloads.add("scanned_images", "unsupported", 1, "OCR engine unavailable")
            else:
                t0 = time.monotonic()
                try:
                    prep, words, _rot = ocr_oriented(img, origin_space="render_px", timeout=min(rc.limits.ocr_timeout_per_image_s, left),
                                                     engine=rc.options.ocr_engine, auto_deskew=scanned)
                    chain_desc = prep.chain.describe() + page_chain.inverse().describe()
                except (OcrTimeout, OcrUnavailable) as exc:
                    rc.counters["images_ocr_failed"] += 1
                    rc.payloads.add("scanned_images", "failed", 1, f"OCR failed on page {pno + 1}: {exc}")
                    words = None
                finally:
                    rc.charge_ocr(t0)
                if words is not None:
                    rc.counters["images_ocr"] += 1
                    for wd in words:
                        wd.box_original = prep.to_original(wd.box)
                    text, spans = words_to_text(words)
                    for f in scan_text(rc.detect, text, "pdf"):
                        boxes_render = span_to_boxes(words, spans, f.start, f.end, use_original=True)
                        rects = [page_chain.map_rect(b, back=True) for b in boxes_render]
                        if not rects:
                            continue
                        reg = finding_to_region(rc.detect, f, unit_id=unit, space=SPACE, rects=rects, source_method="ocr",
                                                anchor={"page": pno + 1, "dpi": dpi}, chain=chain_desc,
                                                page_size=(w, h))
                        texts.append(reg)
        if rc.policy.faces and ("FACE.REGION" in rc.detect.registry.ids()) and rc.detect.registry.get("FACE.REGION").enabled:
            if prep is None:
                prep = prepare_for_ocr(img, origin_space="render_px")
                chain_desc = prep.chain.describe() + page_chain.inverse().describe()
            try:
                boxes = detect_faces(prep.image, rc.options.face_detector)
                rc.payloads.add("face_regions", "sanitised" if boxes else "inspected", len(boxes), "")
                rule = rc.detect.registry.get("FACE.REGION")
                for k, b in enumerate(boxes):
                    rect = page_chain.map_rect(prep.to_original(b), back=True)
                    rid = make_region_id(rc.doc_id, unit, rule.id, f"face-{k}-{round(rect[0])}-{round(rect[1])}")
                    faces.append(Region(region_id=rid, document_id=rc.doc_id, unit_id=unit, space=SPACE, rects=[rect],
                                        anchor={"page": pno + 1}, category=rule.category, rule_id=rule.id,
                                        entity_type=rule.entity_type, severity=rule.severity, hard_block=False,
                                        confidence=rule.base_confidence, source_method="face_detector", action="REDACT",
                                        reason="face region (detection only, no identification)", chain=chain_desc,
                                        page_size=(w, h), text_hash="", text_shape="face", text_len=0))
            except FaceDetectorUnavailable as exc:
                rc.payloads.add("face_regions", "unsupported", 1, f"face detector unavailable: {exc}")
    finally:
        if prep is not None:
            prep.close()
        img.close()
    return texts, faces


def _dedupe_native_ocr(native: list[Region], ocr: list[Region]) -> list[Region]:
    """Fold OCR findings that describe text already found natively. Returns the OCR-only regions."""
    only: list[Region] = []
    for o in ocr:
        ob = o.bbox
        host = None
        for n in native:
            nb = n.bbox
            if rect_overlap_fraction(ob, rect_pad(nb, 2.0)) >= 0.5 or rect_overlap_fraction(nb, rect_pad(ob, 2.0)) >= 0.5:
                host = n
                break
        if host is None:
            only.append(o)
        else:
            host.source_method = "native_text+ocr"
            host.merged_from = sorted(set(host.merged_from) | {o.region_id})
            if o.rule_id != host.rule_id:
                host.also_matched = sorted(set(host.also_matched) | {o.rule_id})
            host.rects = host.rects + [r for r in o.rects if rect_overlap_fraction(r, rect_pad(host.bbox, 1.0)) < 0.6]
    return only


# --------------------------------------------------------------------------- non-page payloads
def _replace_spans(text: str, findings, keep_last: int) -> str:
    out, pos = [], 0
    for f in sorted(findings, key=lambda x: x.start):
        if f.action in ("REDACT", "MASK", "GENERALISE"):
            out.append(text[pos:f.start])
            out.append(make_replacement(f.action, f.rule.category, f.raw, keep_last))
            pos = f.end
    out.append(text[pos:])
    return "".join(out)


def _scan_value(rc: RunContext, value: str, unit: str, method: str, anchor: dict) -> tuple[list[Region], str]:
    """Detect inside one short string (metadata value, annotation, field). Returns regions and cleaned text."""
    regs = []
    fs = scan_text(rc.detect, value, "pdf")
    for f in fs:
        regs.append(finding_to_region(rc.detect, f, unit_id=unit, space="text_anchor", source_method=method, anchor=anchor))
    return regs, _replace_spans(value, fs, rc.policy.mask_keep_last)


def scan_pdf(rc: RunContext, doc) -> dict:
    """Inspect every page and payload. Returns a plan dict consumed by apply_pdf."""
    plan = {"input_path": getattr(doc, "name", None), "pages": [], "metadata": {}, "xmp": None, "annots": [], "widgets": [], "links": [], "toc": None,
            "attachments": [], "words_before": {}, "changed_rects": {}}
    rc.counters["units_total"] += doc.page_count
    has_native_regions = False
    for pno in range(doc.page_count):
        rc.deadline.check(f"page {pno + 1}")
        page = doc[pno]
        text, cmap, words, white = page_text_and_chars(page)
        native_chars = sum(1 for c in text if not c.isspace())
        plan["words_before"][pno] = words
        if white:
            rc.payloads.add("hidden_text", "inspected", white, "white-coloured text present (scanned like any native text)")
        regs = _native_regions(rc, page, pno, text, cmap)
        # images on page ---------------------------------------------------------
        infos = []
        try:
            infos = page.get_image_info(xrefs=True)
        except Exception:
            rc.warn(f"image_info_failed:page {pno + 1}")
        area = page.rect.width * page.rect.height or 1.0
        img_frac = sum(max(0.0, (i["bbox"][2] - i["bbox"][0]) * (i["bbox"][3] - i["bbox"][1])) for i in infos) / area
        rc.counters["images_total"] += len(infos)
        if infos:
            rc.payloads.add("embedded_images", "inspected", len(infos))
        needs_ocr = (img_frac >= IMAGE_AREA_MIN) or native_chars < MIN_NATIVE_CHARS and (infos or native_chars == 0)
        ocr_regs: list[Region] = []
        face_regs: list[Region] = []
        wants_faces = rc.policy.faces and bool(infos)
        if (needs_ocr and rc.policy.ocr) or wants_faces:
            if rc.counters["images_ocr"] + rc.counters["images_ocr_failed"] >= rc.limits.max_ocr_images:
                rc.payloads.add("scanned_images", "failed", 1, "max OCR image count reached")
            else:
                ocr_regs, face_regs = _ocr_regions(rc, page, pno, native_chars)
        elif needs_ocr and not rc.policy.ocr and infos:
            rc.payloads.add("scanned_images", "unsupported", len(infos), "OCR disabled by policy")
        only_ocr = _dedupe_native_ocr(regs, ocr_regs)
        regs.extend(only_ocr)
        regs.extend(face_regs)
        if needs_ocr and infos and rc.policy.ocr and rc.ocr_ready():
            rc.payloads.add("scanned_images", "sanitised" if any(r.source_method.endswith("ocr") for r in regs) else "inspected", len(infos))
        regs = group_regions(regs)
        # thumbnails ------------------------------------------------------------
        try:
            if doc.xref_get_key(page.xref, "Thumb")[0] != "null":
                plan.setdefault("thumb_pages", []).append(pno)
                rc.payloads.add("thumbnails", "sanitised", 1, "page thumbnail removed")
        except Exception:
            pass
        # annotations / widgets / links ---------------------------------------
        for an in (page.annots() or []):
            info = an.info or {}
            content = " ".join(str(info.get(k) or "") for k in ("content", "title", "subject"))
            rs, cleaned = _scan_value(rc, info.get("content") or "", f"page:{pno + 1}:annot:{an.xref}", "annotation", {"page": pno + 1, "xref": an.xref})
            rc.payloads.add("annotations", "inspected", 1)
            if rs:
                regs.extend(rs)
                plan["annots"].append((pno, an.xref, cleaned, rs))
                plan["changed_rects"].setdefault(pno, []).append(tuple(an.rect))
        for w in (page.widgets() or []):
            val = w.field_value if isinstance(w.field_value, str) else ""
            rc.payloads.add("form_fields", "inspected", 1)
            if val:
                rs, cleaned = _scan_value(rc, val, f"page:{pno + 1}:field:{w.xref}", "form_field", {"page": pno + 1, "xref": w.xref})
                if rs:
                    regs.extend(rs)
                    plan["widgets"].append((pno, w.xref, cleaned, rs))
                    plan["changed_rects"].setdefault(pno, []).append(tuple(w.rect))
        for lk in page.get_links():
            uri = lk.get("uri")
            if uri:
                rs, cleaned = _scan_value(rc, uri, f"page:{pno + 1}:link", "link_target", {"page": pno + 1})
                rc.payloads.add("annotations", "inspected", 1, "link targets")
                if rs:
                    regs.extend(rs)
                    plan["links"].append((pno, lk, cleaned, rs))
                    plan["changed_rects"].setdefault(pno, []).append(tuple(lk["from"]))
        plan["pages"].append({"pno": pno, "regions": regs, "size": (page.mediabox.width, page.mediabox.height),
                              "rotation": page.rotation})
        rc.regions.extend(regs)
        rc.counters["units_inspected"] += 1
        has_native_regions = has_native_regions or bool(regs)
    # metadata ------------------------------------------------------------------
    meta = dict(doc.metadata or {})
    clean_meta = {}
    n_meta = 0
    for k in ("title", "author", "subject", "keywords", "creator", "producer"):
        v = meta.get(k) or ""
        if not v:
            continue
        n_meta += 1
        rs, cleaned = _scan_value(rc, v, f"meta:info:{k}", "metadata", {"field": k})
        if rc.policy.strip_author_metadata and k in ("author", "creator"):
            cleaned = ""
            rc.payloads.add("metadata", "sanitised", 1, f"{k} blanked by policy")
        if rs or cleaned != v:
            clean_meta[k] = cleaned
            rc.regions.extend(rs)
            plan["metadata"].setdefault("regions", []).extend(rs)
    rc.payloads.add("metadata", "inspected", n_meta)
    plan["metadata"]["values"] = clean_meta
    try:
        xmp = doc.get_xml_metadata() or ""
    except Exception:
        xmp = ""
    if xmp.strip():
        fs = scan_text(rc.detect, xmp, "pdf")
        regs = [finding_to_region(rc.detect, f, unit_id="meta:xmp", space="text_anchor", source_method="metadata", anchor={"field": "xmp"}) for f in fs]
        rc.payloads.add("metadata", "inspected", 1, "XMP packet")
        if regs:
            plan["xmp"] = _replace_spans(xmp, fs, rc.policy.mask_keep_last)
            rc.regions.extend(regs)
            plan["metadata"].setdefault("regions", []).extend(regs)
    # bookmarks -----------------------------------------------------------------
    toc = doc.get_toc(simple=False)
    if toc:
        new_toc, changed = [], False
        for ent in toc:
            rs, cleaned = _scan_value(rc, ent[1], f"toc:{len(new_toc)}", "bookmark", {"index": len(new_toc)})
            if rs:
                changed = True
                rc.regions.extend(rs)
            new_toc.append([ent[0], cleaned, ent[2]] + ([ent[3]] if len(ent) > 3 else []))
        rc.payloads.add("bookmarks", "inspected", len(toc))
        if changed:
            plan["toc"] = new_toc
            rc.payloads.add("bookmarks", "sanitised", len(toc))
    # attachments / javascript --------------------------------------------------
    n_emb = doc.embfile_count()
    if n_emb:
        rc.payloads.add("embedded_objects", "unsupported", n_emb, "attached files are listed but not sanitised")
        for name in doc.embfile_names():
            try:
                blob = doc.embfile_get(name)
            except Exception:
                continue
            plan["attachments"].append(name)
            try:
                txt = blob[:200_000].decode("utf-8")
            except UnicodeDecodeError:
                continue
            for f in scan_text(rc.detect, txt, "pdf"):
                reg = finding_to_region(rc.detect, f, unit_id=f"attachment:{name}", space="text_anchor", source_method="attachment",
                                        anchor={"name": name})
                reg.action, reg.verification_status, reg.reason = "FLAG_REVIEW", "review_required", "sensitive text inside an attached file that cannot be sanitised"
                rc.regions.append(reg)
    else:
        rc.payloads.add("embedded_objects", "inspected", 0, "no attachments")
    js = 0
    try:
        for x in range(1, doc.xref_length()):
            obj = doc.xref_object(x, compressed=False)
            if "/JavaScript" in obj or re.search(r"/JS\b", obj):
                js += 1
    except Exception:
        pass
    rc.payloads.add("javascript_macros", "preserved" if js else "inspected", js, "document contains JavaScript actions (preserved, flagged)" if js else "")
    if js:
        rc.warn(f"pdf_javascript_present:{js}")
    return plan


# --------------------------------------------------------------------------- apply
def _fit_font(text: str, rect: Rect) -> float:
    h = rect[3] - rect[1]
    w = rect[2] - rect[0]
    fs = max(4.0, min(12.0, h * 0.8))
    while fs > 4.0 and pymupdf.get_text_length(text, fontname="helv", fontsize=fs) > w:
        fs -= 0.5
    return fs


def apply_pdf(rc: RunContext, doc, plan: dict, out_path: str) -> None:
    pad_native, pad_ocr = 0.4, 1.2
    for page_info in plan["pages"]:
        pno = page_info["pno"]
        page = doc[pno]
        added = 0
        for reg in page_info["regions"]:
            if reg.action not in ("REDACT", "MASK", "GENERALISE") or reg.space != SPACE or not reg.rects:
                continue                     # text-anchored regions (annotations, fields, links) are rewritten below
            pad = pad_ocr if "ocr" in reg.source_method or reg.source_method == "face_detector" else pad_native
            bounds = (page.mediabox.x0 - 1e4, page.mediabox.y0 - 1e4, page.mediabox.x1 + 1e4, page.mediabox.y1 + 1e4)
            rects = [rect_pad(r, pad, bounds) for r in reg.rects]
            use_text = reg.action in ("MASK", "GENERALISE") and bool(reg.replacement)
            if use_text:
                fs = _fit_font(reg.replacement, rects[0])
                if fs <= 4.0 and pymupdf.get_text_length(reg.replacement, fontname="helv", fontsize=4.0) > (rects[0][2] - rects[0][0]):
                    use_text = False
                    reg.action = "REDACT"
                    reg.reason += "; replacement did not fit, fell back to REDACT"
                    reg.replacement = make_replacement("REDACT", reg.category, reg.raw)
            for k, rr in enumerate(rects):
                try:
                    if use_text and k == 0:
                        page.add_redact_annot(pymupdf.Rect(rr), text=reg.replacement, fontname="helv",
                                              fontsize=_fit_font(reg.replacement, rr), fill=(1, 1, 1), text_color=(0, 0, 0))
                    else:
                        page.add_redact_annot(pymupdf.Rect(rr), fill=(0, 0, 0))
                    added += 1
                except Exception as exc:
                    reg.verification_status, reg.reason = "failed", f"redaction annotation failed: {exc}"
        if added:
            page.apply_redactions(images=pymupdf.PDF_REDACT_IMAGE_PIXELS, graphics=pymupdf.PDF_REDACT_LINE_ART_NONE,
                                  text=pymupdf.PDF_REDACT_TEXT_REMOVE)
    for pno in plan.get("thumb_pages", []):
        try:
            doc.xref_set_key(doc[pno].xref, "Thumb", "null")
        except Exception:
            rc.warn(f"thumbnail_removal_failed:page {pno + 1}")
    for pno, xref, cleaned, rs in plan["annots"]:
        page = doc[pno]
        for an in page.annots():
            if an.xref == xref:
                try:
                    an.set_info(content=cleaned)
                    an.update()
                    for r in rs:
                        r.action = r.action if r.action != "NONE" else "REDACT"
                except Exception:
                    page.delete_annot(an)
                    for r in rs:
                        r.reason += "; annotation removed (could not rewrite)"
                break
    for pno, xref, cleaned, rs in plan["widgets"]:
        for w in doc[pno].widgets():
            if w.xref == xref:
                try:
                    w.field_value = cleaned
                    w.update()
                except Exception:
                    for r in rs:
                        r.verification_status, r.reason = "failed", "form field could not be rewritten"
                break
    for pno, lk, cleaned, rs in plan["links"]:
        try:
            lk2 = dict(lk)
            lk2["uri"] = cleaned
            doc[pno].update_link(lk2)
        except Exception:
            for r in rs:
                r.verification_status, r.reason = "failed", "link target could not be rewritten"
    if plan["metadata"].get("values"):
        meta = dict(doc.metadata or {})
        meta.update(plan["metadata"]["values"])
        doc.set_metadata(meta)
        rc.payloads.add("metadata", "sanitised", len(plan["metadata"]["values"]))
    if plan["xmp"] is not None:
        doc.set_xml_metadata(plan["xmp"])
        rc.payloads.add("metadata", "sanitised", 1, "XMP")
    if plan["toc"] is not None:
        doc.set_toc(plan["toc"])
    doc.save(out_path, garbage=4, deflate=True, clean=True)
    # mark text-anchored metadata regions as applied (verification confirms)
