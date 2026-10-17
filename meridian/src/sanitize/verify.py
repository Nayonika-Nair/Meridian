"""Independent verification of the OUTPUT artifact (A12).

The artifact is reopened from disk (never from the objects the sanitiser held) and inspected with
different code paths than the ones that wrote it:

  PDF     PyMuPDF text, pypdf text, decoded content-stream strings, pdfium render + OCR
  OOXML   zip + lxml text of EVERY part, raw bytes of every zip member (UTF-8 and UTF-16), per-unit
          text hashes against the expected result, formatting digests, media hashes, OCR of media
  image   Pillow decode, EXIF, OCR of the re-opened pixels, pixel comparison outside redactions

Occurrence-aware by design: a value that was NOT detected as sensitive at another location (for
example an invoice number equal to a labelled account number) is not treated as a leak. Hard-block
values (government IDs, cards, secrets) are different: the exact same string anywhere in the
output counts as a leak.

Verification results distinguish: detection (what the rules found), sanitisation (what was applied),
verification (what the independent checks could confirm) and unsupported/unchecked content.
"""
from __future__ import annotations

import difflib
import hashlib
import io
import re
import time
import zipfile
from collections import Counter

from lxml import etree
from PIL import Image

from .context import RunContext
from .coords import pdf_page_chain, rect_overlap_fraction, rect_pad
from .detect import finding_to_region, norm_value, scan_text
from .imaging import OcrTimeout, OcrUnavailable, ocr_oriented, ocr_words, open_image, prepare_for_ocr, words_to_text, span_to_boxes
from .regions import Region
from .safety import sha256_bytes
from .textunits import parse_xml, run_format_digest

ACTING = ("REDACT", "MASK", "GENERALISE")


def _chk(name: str, status: str, detail: str = "", **extra) -> dict:
    return {"name": name, "status": status, "detail": detail, **extra}


def _lcs(a: str, b: str) -> int:
    if not a or not b:
        return 0
    return difflib.SequenceMatcher(None, a, b, autojunk=False).find_longest_match(0, len(a), 0, len(b)).size


def _leaks(raw: str, text: str) -> bool:
    t, r = norm_value(text), norm_value(raw)
    if len(r) < 4:
        return False
    if r in t:
        return True
    return _lcs(r, t) >= max(6, int(0.8 * len(r)))


def _acting(regions: list[Region]) -> list[Region]:
    return [r for r in regions if r.action in ACTING and r.verification_status != "failed"]


def _residual(rc: RunContext, text: str, fmt: str, label: str, regions: list[Region]) -> int:
    """Re-run the detector over output text. Any acting-rule finding is a residual leak (tracked as a region)."""
    n = 0
    for f in scan_text(rc.detect, text, fmt):
        if f.action in ACTING:
            n += 1
            reg = finding_to_region(rc.detect, f, unit_id=f"verify:{label}", space="text_anchor",
                                    source_method="verification_rescan", anchor={"channel": label})
            reg.verification_status, reg.reason = "failed", f"residual match in output ({label})"
            regions.append(reg)
    return n


def _hard_block_presence(raws: list[str], blobs: dict[str, str]) -> list[str]:
    hits = []
    for raw in raws:
        nr = norm_value(raw)
        for label, txt in blobs.items():
            if nr in norm_value(txt):
                hits.append(label)
    return hits


# --------------------------------------------------------------------------- OCR on a rendered/decoded image
def _ocr_image(rc: RunContext, img: Image.Image, *, origin: str = "original_px", exif_orient=None):
    """-> (words, text, spans, prep) or None when OCR cannot run."""
    if not (rc.policy.ocr and rc.ocr_ready()) or rc.ocr_budget_left() <= 0.5:
        return None
    t0 = time.monotonic()
    try:
        prep, words, _rot = ocr_oriented(img, origin_space=origin, exif_orient=exif_orient,
                                         min_long_side=1400 if origin != "render_px" else 0,
                                         timeout=min(rc.limits.ocr_timeout_per_image_s, rc.ocr_budget_left()),
                                         engine=rc.options.ocr_engine)
    except (OcrTimeout, OcrUnavailable):
        return None
    finally:
        rc.charge_ocr(t0)
    for w in words:
        w.box_original = prep.to_original(w.box)
    text, spans = words_to_text(words)
    return words, text, spans, prep


def _local_ocr_leak(words, rects, raw: str) -> bool:
    from .coords import rect_norm
    buf = []
    for w in words:
        b = w.box_original or w.box
        if any(rect_overlap_fraction(b, rect_pad(r, 2.0)) >= 0.3 for r in rects):
            buf.append(w.text)
    return _leaks(raw, " ".join(buf))


# --------------------------------------------------------------------------- PDF
_PDF_DATE = re.compile(r"D:\d{8,14}[^)\s]*")          # PDF date strings are digits by construction, never card numbers


def _printable_ratio(b: bytes) -> float:
    if not b:
        return 0.0
    return sum(1 for c in b if 32 <= c < 127 or c in (9, 10, 13)) / len(b)


def _pdf_stream_strings(doc) -> str:
    """Every object dictionary plus the decoded string operands of every stream.

    Catches text that is no longer drawn but is still carried by the file (old content, hidden objects, XMP,
    embedded text). Textual streams (XMP, embedded plain text) are included verbatim.
    """
    parts: list[str] = []
    str_re = re.compile(rb"\((?:\\.|[^\\()])*\)|<[0-9A-Fa-f\s]{4,}>")
    for x in range(1, doc.xref_length()):
        try:
            parts.append(doc.xref_object(x, compressed=False))
        except Exception:
            continue
        try:
            if not doc.xref_is_stream(x):
                continue
            data = doc.xref_stream(x) or b""
        except Exception:
            continue
        try:
            if "/EmbeddedFile" in doc.xref_object(x, compressed=False):
                continue                      # attachments are reported as unsupported payloads, not scanned here
        except Exception:
            pass
        if _printable_ratio(data[:4096]) > 0.92 and b"BT" not in data[:2048]:
            parts.append(data[:2_000_000].decode("latin-1", "ignore"))
        for m in str_re.finditer(data):
            s = m.group(0)
            if s.startswith(b"<"):
                try:
                    s = bytes.fromhex(re.sub(rb"\s", b"", s[1:-1]).decode())
                except Exception:
                    continue
            else:
                s = s[1:-1]
            parts.append(s.decode("latin-1", "ignore"))
    return "\n".join(parts)


def _render_pdfium(path: str, pno: int, dpi: int) -> Image.Image | None:
    try:
        import pypdfium2 as pdfium
    except Exception:
        return None
    pdf = pdfium.PdfDocument(path)
    try:
        page = pdf[pno]
        bm = page.render(scale=dpi / 72.0)
        return bm.to_pil().convert("RGB")
    finally:
        pdf.close()


def verify_pdf(rc: RunContext, out_path: str, plan: dict, regions: list[Region]) -> list[dict]:
    import pymupdf
    checks = []
    acting = _acting(regions)
    # 1. structure --------------------------------------------------------------------------------------
    try:
        doc = pymupdf.open(out_path)
    except Exception as exc:
        return [_chk("structure_valid", "failed", f"output PDF cannot be opened: {exc}")]
    try:
        if doc.needs_pass or doc.page_count != len(plan["pages"]):
            checks.append(_chk("structure_valid", "failed", f"page count {doc.page_count} vs {len(plan['pages'])} expected"))
            return checks
        checks.append(_chk("structure_valid", "passed", f"{doc.page_count} pages, reopened with PyMuPDF"))
        # 2. local text check -----------------------------------------------------------------------------
        leaks = 0
        page_words: dict[int, list] = {}
        for pno in range(doc.page_count):
            page = doc[pno]
            page_words[pno] = page.get_text("words")
        for r in acting:
            if r.space != "pdf_unrot_tl" or not r.rects:
                continue
            pno = r.anchor.get("page", 1) - 1
            buf = []
            for w in page_words.get(pno, []):
                wb = tuple(w[:4])
                if any(rect_overlap_fraction(wb, rect_pad(rr, 1.0)) >= 0.3 for rr in r.rects):
                    buf.append(w[4])
            if _leaks(r.raw, " ".join(buf)):
                r.verification_status, r.reason = "failed", "target text still extractable from the redacted area"
                leaks += 1
        checks.append(_chk("target_text_removed_native", "failed" if leaks else "passed",
                           f"{leaks} region(s) still extractable" if leaks else f"{len([r for r in acting if r.space == 'pdf_unrot_tl'])} redacted area(s) yield no target text"))
        # 3. residual scan over independent extractions ----------------------------------------------------
        texts = {"pymupdf": "\n".join(doc[p].get_text() for p in range(doc.page_count))}
        try:
            from pypdf import PdfReader
            rd = PdfReader(out_path)
            texts["pypdf"] = "\n".join((pg.extract_text() or "") for pg in rd.pages)
        except Exception as exc:
            checks.append(_chk("independent_extractor_pypdf", "warning", f"pypdf unavailable or failed: {exc}"))
        texts["streams"] = _PDF_DATE.sub("", _pdf_stream_strings(doc))
        meta = " ".join(str(v) for k, v in (doc.metadata or {}).items() if v and k not in ("creationDate", "modDate", "format", "encryption")) \
            + " " + (doc.get_xml_metadata() or "")
        texts["metadata"] = meta
        resid = 0
        for label, txt in texts.items():
            resid += _residual(rc, txt, "pdf", label, regions)
        checks.append(_chk("residual_rescan", "failed" if resid else "passed",
                           f"{resid} acting-rule match(es) remain in output text channels" if resid else
                           f"no acting-rule matches in {', '.join(texts)}"))
        hb = [r.raw for r in acting if r.hard_block]
        hits = _hard_block_presence(hb, texts)
        checks.append(_chk("hard_block_values_absent", "failed" if hits else "passed",
                           f"hard-block value still present in {sorted(set(hits))}" if hits else
                           f"{len(hb)} hard-block value(s) absent from all text channels"))
        if hits:
            for r in acting:
                if r.hard_block:
                    r.verification_status, r.reason = "failed", "hard-block value found in output"
        # 4. OCR recheck -------------------------------------------------------------------------------------
        need_pages = sorted({p["pno"] for p in plan["pages"] if p["regions"]} | {r.anchor.get("page", 1) - 1 for r in acting if r.space == "pdf_unrot_tl"})
        has_images = any(doc[p["pno"]].get_image_info() for p in plan["pages"])
        ocr_done, ocr_leaks, ocr_resid, ocr_incomplete = 0, 0, 0, False
        if not rc.policy.ocr:
            checks.append(_chk("ocr_recheck", "skipped", "OCR disabled by policy"))
        elif not rc.ocr_ready():
            checks.append(_chk("ocr_recheck", "incomplete" if (need_pages or has_images) else "skipped", "OCR engine unavailable: visual recoverability not checked"))
        else:
            dpi = min(rc.limits.ocr_dpi, 200)
            for pno in need_pages:
                img = _render_pdfium(out_path, pno, dpi)
                if img is None:
                    pix = doc[pno].get_pixmap(dpi=dpi, alpha=False)
                    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                res = _ocr_image(rc, img, origin="render_px")
                if res is None:
                    ocr_incomplete = True
                    img.close()
                    continue
                words, text, spans, prep = res
                ocr_done += 1
                page = doc[pno]
                chain = pdf_page_chain(page.mediabox.width, page.mediabox.height, page.rotation, dpi)
                ocr_resid += _residual(rc, text, "pdf", f"ocr:page{pno + 1}", regions)
                for r in acting:
                    if r.space == "pdf_unrot_tl" and r.anchor.get("page", 1) - 1 == pno and r.rects:
                        # words are in render px; bring region rects into render px for the overlap test
                        rr = [chain.map_rect(x) for x in r.rects]
                        if _local_ocr_leak(words, rr, r.raw):
                            r.verification_status, r.reason = "failed", "target still readable by OCR in the rendered output"
                            ocr_leaks += 1
                prep.close()
                img.close()
            bad = ocr_leaks or ocr_resid
            checks.append(_chk("ocr_recheck", "failed" if bad else ("incomplete" if ocr_incomplete else "passed"),
                               f"{ocr_done} page render(s) OCR'd; {ocr_leaks} readable region(s), {ocr_resid} residual match(es)"
                               + ("; OCR budget/timeout prevented some pages" if ocr_incomplete else "")))
        # 5. unrelated content preserved ----------------------------------------------------------------------
        lost = total = 0
        for pi in plan["pages"]:
            pno = pi["pno"]
            before = plan["words_before"].get(pno, [])
            rects = [rect_pad(rr, 2.0) for r in acting if r.space == "pdf_unrot_tl" and r.anchor.get("page", 1) - 1 == pno for rr in r.rects]
            rects += [rect_pad(tuple(rr), 2.0) for rr in plan.get("changed_rects", {}).get(pno, [])]   # rewritten annotations / fields / links
            keep = Counter(w for w, b in before if not any(rect_overlap_fraction(b, rc_) > 0 for rc_ in rects))
            after = Counter(w[4] for w in page_words.get(pno, []))
            total += sum(keep.values())
            lost += sum((keep - after).values())
        ratio = lost / total if total else 0.0
        checks.append(_chk("unrelated_content_preserved", "failed" if ratio > 0.02 else "passed",
                           f"{lost} of {total} words outside redactions missing ({ratio:.1%})", lost=lost, total=total))
        # 5b. visual preservation: pages are rendered before/after with the independent renderer; pixels outside
        #     the redaction boxes must be unchanged (covers scanned pages whose "unrelated content" is an image)
        if plan.get("input_path") and need_pages:
            import numpy as np
            worst, compared = 0.0, 0
            for pno in need_pages:
                a = _render_pdfium(plan["input_path"], pno, 80)
                b = _render_pdfium(out_path, pno, 80)
                if a is None or b is None or a.size != b.size:
                    continue
                page = doc[pno]
                ch = pdf_page_chain(page.mediabox.width, page.mediabox.height, page.rotation, 80)
                mask = np.ones((a.size[1], a.size[0]), dtype=bool)
                for r in acting:
                    if r.space == "pdf_unrot_tl" and r.anchor.get("page", 1) - 1 == pno:
                        for rr in r.rects:
                            x0, y0, x1, y1 = [int(v) for v in ch.map_rect(rect_pad(rr, 3.0))]
                            mask[max(0, y0):max(0, y1) + 1, max(0, x0):max(0, x1) + 1] = False
                ga = np.asarray(a.convert("L"), dtype="int16")
                gb = np.asarray(b.convert("L"), dtype="int16")
                if mask.any():
                    worst = max(worst, float(np.abs(ga - gb)[mask].mean()))
                    compared += 1
            if compared:
                checks.append(_chk("visual_content_preserved", "failed" if worst > 6.0 else "passed",
                                   f"{compared} page render(s) compared outside redaction boxes; worst mean difference {worst:.2f} (tolerance 6.0)"))
        # 6. metadata -----------------------------------------------------------------------------------------
        checks.append(_chk("metadata_checked", "failed" if _residual_count(rc, texts["metadata"], "pdf") else "passed",
                           "Info dictionary and XMP re-scanned"))
    finally:
        doc.close()
    return checks


def _residual_count(rc, text, fmt) -> int:
    return sum(1 for f in scan_text(rc.detect, text, fmt) if f.action in ACTING)


# --------------------------------------------------------------------------- OOXML
def _zip_text(zf: zipfile.ZipFile) -> dict[str, str]:
    out = {}
    for name in zf.namelist():
        if not name.lower().endswith((".xml", ".rels")):
            continue
        try:
            root = parse_xml(zf.read(name))
        except Exception:
            out[name] = zf.read(name).decode("utf-8", "ignore")
            continue
        chunks = []
        for el in root.iter():
            if isinstance(el.tag, str):
                if el.text and el.text.strip():
                    chunks.append(el.text)
                for v in el.attrib.values():
                    if isinstance(v, str) and len(v) > 3:
                        chunks.append(v)
        out[name] = "\n".join(chunks)
    return out


def verify_ooxml(rc: RunContext, out_path: str, plan: dict, regions: list[Region], family: str,
                 original_media: dict[str, str]) -> list[dict]:
    from .ooxml import (DOCX_PARTS, PPTX_PARTS, _build_units, _classify, open_package, part_root)
    checks = []
    acting = _acting(regions)
    try:
        with zipfile.ZipFile(out_path) as zf:
            bad = zf.testzip()
            names = zf.namelist()
            zip_texts = _zip_text(zf)
            raw_members = {n: zf.read(n) for n in names if not n.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff", ".webp"))
                           and zf.getinfo(n).file_size < 20_000_000}
            media = {n: zf.read(n) for n in names if "/media/" in n}
    except Exception as exc:
        return [_chk("structure_valid", "failed", f"output package unreadable: {exc}")]
    try:
        obj = open_package(out_path, family)
    except Exception as exc:
        return [_chk("structure_valid", "failed", f"output cannot be reopened with the library: {exc}")]
    checks.append(_chk("structure_valid", "failed" if bad else "passed", f"zip ok, {len(names)} members, reopened with python-{family}" if not bad else f"corrupt member {bad}"))
    # unit-by-unit expectation --------------------------------------------------------------------------------
    pkg = obj.part.package
    units_after = {}
    for part in sorted(pkg.iter_parts(), key=lambda p: str(p.partname)):
        name = str(part.partname)
        payload, mode = _classify(family, name)
        if not mode:
            continue
        root, _g = part_root(part)
        for u in _build_units(mode, root, name):
            units_after[u.unit_id] = (hashlib.sha256(u.text.encode()).hexdigest(), run_format_digest(u.container) if u.container is not None else "")
    exp = plan["units"]
    missing = [u for u in exp if u not in units_after]
    extra = [u for u in units_after if u not in exp]
    wrong_text = [u for u, v in exp.items() if u in units_after and units_after[u][0] != v["text_after_sha"]]
    wrong_fmt = [u for u, v in exp.items() if u in units_after and not v.get("touched") and v["digest_before"] and units_after[u][1] != v["digest_before"]]
    ok = not (missing or extra or wrong_text)
    checks.append(_chk("target_text_removed_exact_occurrences", "passed" if ok else "failed",
                       f"{len(exp)} text unit(s) match the expected post-replacement text" if ok else
                       f"{len(wrong_text)} unit(s) differ from expectation, {len(missing)} missing, {len(extra)} unexpected"))
    for u in wrong_text:
        for r in acting:
            if r.unit_id == u:
                r.verification_status, r.reason = "failed", "output text differs from the expected replacement"
    checks.append(_chk("unrelated_content_preserved", "failed" if (wrong_text or wrong_fmt or missing) else "passed",
                       f"{len(exp) - len(wrong_text) - len(missing)} of {len(exp)} units text-identical to expectation; "
                       f"{len(wrong_fmt)} untouched unit(s) changed run formatting"))
    # residual rescan over every part ------------------------------------------------------------------------------
    resid = 0
    for label, txt in zip_texts.items():
        resid += _residual(rc, txt, family, label, regions)
    checks.append(_chk("residual_rescan", "failed" if resid else "passed",
                       f"{resid} acting-rule match(es) remain across {len(zip_texts)} XML part(s)" if resid else f"no acting-rule matches in {len(zip_texts)} XML part(s)"))
    # hard-block raw presence incl. UTF-16 and non-text members ------------------------------------------------------
    hb = [r.raw for r in acting if r.hard_block]
    hits = _hard_block_presence(hb, {k: v for k, v in zip_texts.items()})
    for raw in hb:
        for n, data in raw_members.items():
            if raw.encode("utf-8") in data or raw.encode("utf-16-le") in data:
                hits.append(n)
    checks.append(_chk("hard_block_values_absent", "failed" if hits else "passed",
                       f"hard-block value found in {sorted(set(hits))[:5]}" if hits else f"{len(hb)} hard-block value(s) absent from all parts and raw members"))
    if hits:
        for r in acting:
            if r.hard_block:
                r.verification_status, r.reason = "failed", "hard-block value found in output package"
    # embedded copies + OCR of media --------------------------------------------------------------------------------------
    changed_shas = {sha for sha, regs in plan["img_regions"].items() if any(r in acting for r in regs)}
    still = [n for n, data in media.items() if sha256_bytes(data) in changed_shas]
    checks.append(_chk("embedded_copies_removed", "failed" if still else "passed",
                       f"original image bytes still in package: {still}" if still else
                       f"{len(changed_shas)} original image(s) with findings no longer present in the package"))
    if still:
        for sha, regs in plan["img_regions"].items():
            for r in regs:
                if r in acting:
                    r.verification_status, r.reason = "failed", "unsanitised copy of the image remains in the package"
    img_regions = [r for r in acting if r.handle and r.handle[0] == "image"]
    if not rc.policy.ocr and media:
        checks.append(_chk("ocr_recheck", "skipped", "OCR disabled by policy"))
    elif media and not rc.ocr_ready():
        checks.append(_chk("ocr_recheck", "incomplete", "OCR engine unavailable: embedded images not re-checked"))
    elif media:
        done, leaks, rres, incomplete = 0, 0, 0, False
        ref_by_sha_part = {}
        for r in img_regions:
            ref = r.handle[1]
            ref_by_sha_part.setdefault(str(ref.target.partname), []).append(r)
        for n, data in sorted(media.items()):
            try:
                im = open_image(io.BytesIO(data), rc.limits.max_image_pixels)
            except Exception:
                continue
            res = _ocr_image(rc, im, exif_orient=im.getexif().get(0x0112) if hasattr(im, "getexif") else None)
            if res is None:
                incomplete = True
                im.close()
                continue
            words, text, spans, prep = res
            done += 1
            rres += _residual(rc, text, "image", f"ocr:{n}", regions)
            prep.close()
            im.close()
        checks.append(_chk("ocr_recheck", "failed" if (leaks or rres) else ("incomplete" if incomplete else "passed"),
                           f"{done} embedded image(s) re-OCR'd; {rres} residual match(es)" + ("; some images could not be checked" if incomplete else "")))
        # local check of each sanitised occurrence in its NEW image
        for r in img_regions:
            ref = r.handle[1]
        # (the residual rescan above is the recoverability test; per-region geometry check follows)
        _ooxml_image_local(rc, obj, family, img_regions)
    else:
        checks.append(_chk("ocr_recheck", "skipped", "no embedded images"))
    # thumbnail --------------------------------------------------------------------------------------------------------
    if acting:
        thumbs = [n for n in names if "thumbnail" in n.lower()]
        checks.append(_chk("thumbnail_removed", "failed" if thumbs else "passed", f"thumbnail still present: {thumbs}" if thumbs else "no package thumbnail"))
    checks.append(_chk("office_rendering", "skipped",
                       "Office layout is not rendered: visual recoverability is checked only for embedded images"))
    return checks


def _ooxml_image_local(rc, obj, family, img_regions: list[Region]) -> None:
    """For every sanitised occurrence, re-OCR the image the occurrence NOW points to and test its region rects."""
    cache = {}
    for r in img_regions:
        if r.verification_status == "failed":
            continue
        ref = r.handle[1]
        try:
            rel = ref.owner.rels.get(ref.rid)
            blob = rel.target_part.blob
        except Exception:
            continue
        sha = sha256_bytes(blob)
        if sha not in cache:
            try:
                im = open_image(io.BytesIO(blob), rc.limits.max_image_pixels)
            except Exception:
                cache[sha] = None
                continue
            cache[sha] = _ocr_image(rc, im, exif_orient=im.getexif().get(0x0112))
            im.close()
        res = cache[sha]
        if res is None:
            continue
        words, text, spans, prep = res
        if _local_ocr_leak(words, r.rects, r.raw):
            r.verification_status, r.reason = "failed", "target still readable by OCR in the sanitised image"


# --------------------------------------------------------------------------- stand-alone images
def verify_image(rc: RunContext, out_path: str, src_img: Image.Image, regions: list[Region]) -> list[dict]:
    checks = []
    acting = _acting(regions)
    try:
        im = open_image(out_path, rc.limits.max_image_pixels)
    except Exception as exc:
        return [_chk("structure_valid", "failed", f"output image cannot be decoded: {exc}")]
    try:
        same = im.size == src_img.size
        checks.append(_chk("structure_valid", "passed" if same else "failed", f"{im.size[0]}x{im.size[1]} {im.format}; dimensions {'unchanged' if same else 'CHANGED'}"))
        # EXIF
        ex = im.getexif()
        from .image_ops import SENSITIVE_EXIF
        bad = [SENSITIVE_EXIF[t] for t in ex.keys() if t in SENSITIVE_EXIF]
        try:
            if 0x8825 in ex and ex.get_ifd(0x8825):
                bad.append("GPS")
        except Exception:
            pass
        checks.append(_chk("metadata_checked", "failed" if bad else "passed", f"sensitive EXIF remains: {sorted(set(bad))}" if bad else "no sensitive EXIF tags"))
        # pixels outside regions unchanged
        import numpy as np
        a = np.asarray(src_img.convert("RGB"), dtype="int16")
        b = np.asarray(im.convert("RGB"), dtype="int16") if same else None
        if b is not None:
            mask = np.ones(a.shape[:2], dtype=bool)
            for r in acting:
                for rect in r.rects:
                    x0, y0, x1, y1 = [int(round(v)) for v in rect_pad(rect, 3.0)]
                    mask[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = False
            diff = float(np.abs(a - b)[mask].mean()) if mask.any() else 0.0
            tol = 6.0 if (im.format or "").upper() in ("JPEG", "WEBP") else 0.5
            checks.append(_chk("unrelated_content_preserved", "passed" if diff <= tol else "failed", f"mean pixel difference outside redactions {diff:.2f} (tolerance {tol})"))
        # OCR recheck
        if not rc.policy.ocr:
            checks.append(_chk("ocr_recheck", "skipped", "OCR disabled by policy"))
        elif not rc.ocr_ready():
            checks.append(_chk("ocr_recheck", "incomplete", "OCR engine unavailable: output not re-read"))
        else:
            res = _ocr_image(rc, im, exif_orient=ex.get(0x0112))
            if res is None:
                checks.append(_chk("ocr_recheck", "incomplete", "OCR budget/timeout prevented the recheck"))
            else:
                words, text, spans, prep = res
                rres = _residual(rc, text, "image", "ocr:image", regions)
                leaks = 0
                for r in acting:
                    if r.rects and _local_ocr_leak(words, r.rects, r.raw):
                        r.verification_status, r.reason = "failed", "target still readable by OCR"
                        leaks += 1
                prep.close()
                checks.append(_chk("ocr_recheck", "failed" if (rres or leaks) else "passed",
                                   f"OCR re-read the output: {leaks} readable region(s), {rres} residual match(es)"))
        faces = [r for r in acting if r.source_method == "face_detector"]
        if faces:
            gray = np.asarray(im.convert("L"))
            bad = 0
            for r in faces:
                x0, y0, x1, y1 = [int(round(v)) for v in r.bbox]
                zone = gray[max(0, y0):max(0, y1), max(0, x0):max(0, x1)]
                if zone.size and zone.mean() > 40:
                    r.verification_status, r.reason = "failed", "face region not blanked in the output"
                    bad += 1
            checks.append(_chk("face_regions_blanked", "failed" if bad else "passed", f"{len(faces)} face region(s) checked for blanking; {bad} not blanked"))
        checks.append(_chk("target_text_removed", "failed" if any(r.verification_status == "failed" for r in acting) else "passed",
                           f"{len(acting)} acting region(s) re-tested"))
    finally:
        im.close()
    return checks


# --------------------------------------------------------------------------- final statuses
def finalise_statuses(regions: list[Region]) -> dict:
    """Give every region a terminal verification status and return counts."""
    counts = Counter()
    for r in regions:
        if r.action in ACTING:
            if r.verification_status in ("pending", "verified"):
                r.verification_status = "verified"
        elif r.action == "FLAG_REVIEW":
            r.verification_status = "review_required"
            r.reason = r.reason or "flagged for human review"
        else:
            if r.verification_status == "pending":
                r.verification_status = "not_applicable"
        counts[r.verification_status] += 1
    return dict(counts)


def overall_status(checks: list[dict]) -> str:
    st = [c["status"] for c in checks]
    if "failed" in st:
        return "failed"
    if "incomplete" in st:
        return "incomplete"
    return "passed"
