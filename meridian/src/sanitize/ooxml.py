"""DOCX and PPTX sanitisation (A3, A5) on top of python-docx / python-pptx.

What is inspected (and, where the capability registry says so, sanitised)
------------------------------------------------------------------------
paragraphs & runs, tables, headers/footers, footnotes/endnotes, comments, text boxes and
alternate-content fallbacks, tracked-change deletions, hidden (vanish) text, field instructions,
hyperlink targets, shape alt-text, slide shapes/groups/notes/masters, document properties,
custom XML, charts/diagrams (text), embedded images (OCR + faces), package thumbnails, and a
census of embedded objects that cannot be sanitised (OLE, workbooks, macros, altChunk, media).

Embedded images use copy-on-write per OCCURRENCE: the sanitised bytes become a new media part and
only the referencing element (``a:blip`` / ``v:imagedata``) of the affected occurrence is re-pointed.
Other occurrences of the same underlying image keep the original part untouched.
"""
from __future__ import annotations

import hashlib
import io
import re
import zipfile
from dataclasses import dataclass, field

from lxml import etree
from PIL import Image

from .context import RunContext
from .detect import finding_to_region, scan_text
from .image_ops import ImageScan, inspect_image, render_sanitised
from .imaging import open_image
from .regions import Region
from .safety import InputRejected, sha256_bytes
from .textunits import (A, A_NS, R_NS, W, W_NS, TextUnit, all_text_nodes, attr_units, drawing_units, parse_xml,
                        run_format_digest, serialise_xml, value_units, word_units)

P_NS = "http://schemas.openxmlformats.org/presentationml/2006/main"
C_NS = "http://schemas.openxmlformats.org/drawingml/2006/chart"
WP_NS = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
V_NS = "urn:schemas-microsoft-com:vml"
O_NS = "urn:schemas-microsoft-com:office:office"
MC_NS = "http://schemas.openxmlformats.org/markup-compatibility/2006"
PIC_NS = "http://schemas.openxmlformats.org/drawingml/2006/picture"

DOCX_PARTS = [
    (r"^/word/document\.xml$", "body_text", "word"),
    (r"^/word/(header|footer)\d*\.xml$", "headers_footers", "word"),
    (r"^/word/(footnotes|endnotes)\.xml$", "footnotes_endnotes", "word"),
    (r"^/word/comments\w*\.xml$", "comments", "word"),
    (r"^/word/charts/chart\d+\.xml$", "charts_diagrams", "chart"),
    (r"^/word/diagrams/(data|drawing)\d*\.xml$", "charts_diagrams", "drawing"),
    (r"^/customXml/item\d+\.xml$", "custom_xml", "generic"),
    (r"^/docProps/(app|custom)\.xml$", "metadata", "generic"),
]
PPTX_PARTS = [
    (r"^/ppt/slides/slide\d+\.xml$", "body_text", "pptx"),
    (r"^/ppt/(slideLayouts|slideMasters)/[^/]+\.xml$", "masters_layouts", "pptx"),
    (r"^/ppt/(notesSlides|notesMasters|handoutMasters)/[^/]+\.xml$", "notes", "pptx"),
    (r"^/ppt/comments/[^/]+\.xml$", "comments", "pcomments"),
    (r"^/ppt/charts/chart\d+\.xml$", "charts_diagrams", "chart"),
    (r"^/ppt/diagrams/(data|drawing)\d*\.xml$", "charts_diagrams", "drawing"),
    (r"^/ppt/tags/[^/]+\.xml$", "custom_xml", "generic"),
    (r"^/customXml/item\d+\.xml$", "custom_xml", "generic"),
    (r"^/docProps/(app|custom)\.xml$", "metadata", "generic"),
]
EMBED_RE = re.compile(r"(/embeddings/|/activeX/|vbaProject\.bin$|/oleObject|\.ole$)", re.I)
CORE_ATTRS = ("author", "category", "comments", "content_status", "identifier", "keywords", "language",
              "last_modified_by", "subject", "title", "version")
IMAGE_CT_OK = ("image/png", "image/jpeg", "image/gif", "image/bmp", "image/tiff", "image/webp", "image/x-png", "image/pjpeg")


def open_package(path: str, family: str):
    try:
        if family == "docx":
            import docx
            return docx.Document(path)
        import pptx
        return pptx.Presentation(path)
    except Exception as exc:
        raise InputRejected("malformed", f"{family.upper()} could not be opened: {exc}") from exc


def _package(obj):
    return obj.part.package


def part_root(part):
    """(root element, is_generic). XmlParts expose a live element; other parts are parsed from their blob."""
    el = getattr(part, "_element", None)           # python-docx and python-pptx XmlPart keep the live tree here
    if el is not None:
        return el, False
    return parse_xml(part.blob), True


def commit_part(part, root) -> None:
    part._blob = serialise_xml(root)


def _classify(family: str, name: str):
    for pat, payload, mode in (DOCX_PARTS if family == "docx" else PPTX_PARTS):
        if re.match(pat, name):
            return payload, mode
    return None, None


def _build_units(mode: str, root, name: str):
    if mode == "word":
        yield from word_units(root, name)
        yield from attr_units(root, name, [(f"{{{WP_NS}}}docPr", "descr"), (f"{{{WP_NS}}}docPr", "title")], "alt")
    elif mode == "pptx":
        yield from drawing_units(root, name)
        yield from attr_units(root, name, [(f"{{{P_NS}}}cNvPr", "descr"), (f"{{{P_NS}}}cNvPr", "title")], "alt")
    elif mode == "pcomments":
        yield from value_units(root, name, [f"{{{P_NS}}}text"], "c")
        yield from drawing_units(root, name)
    elif mode == "drawing":
        yield from drawing_units(root, name)
    elif mode == "chart":
        yield from drawing_units(root, name)
        yield from value_units(root, name, [f"{{{C_NS}}}v"], "v")
    elif mode == "generic":
        yield from all_text_nodes(root, name)


def _count(root, ns: str, tag: str) -> int:
    return sum(1 for _ in root.iter(f"{{{ns}}}{tag}"))


# --------------------------------------------------------------------------- image references
@dataclass
class ImageRef:
    owner: object                  # package part containing the element
    owner_name: str
    elem: object                   # a:blip or v:imagedata
    attr: str                      # {R_NS}embed or {R_NS}id / {O_NS}relid
    rid: str
    target: object                 # image part
    index: int
    kind: str                      # blip | vml

    @property
    def occurrence_id(self) -> str:
        return f"{self.owner_name}#img{self.index}"


def collect_image_refs(parts, roots) -> tuple[list[ImageRef], list[str]]:
    refs, notes = [], []
    for part in parts:
        name = str(part.partname)
        root = roots.get(name)
        if root is None:
            continue
        i = 0
        for el in root.iter(f"{{{A_NS}}}blip", f"{{{V_NS}}}imagedata"):
            if etree.QName(el).localname == "blip":
                attr_options = [f"{{{R_NS}}}embed"]
                kind = "blip"
            else:
                attr_options = [f"{{{R_NS}}}id", f"{{{O_NS}}}relid"]
                kind = "vml"
            if el.get(f"{{{R_NS}}}link") and not any(el.get(a) for a in attr_options):
                notes.append(f"linked_image_not_embedded:{name}")
                continue
            for attr in attr_options:
                rid = el.get(attr)
                if not rid:
                    continue
                rel = part.rels.get(rid)
                if rel is None or rel.is_external:
                    notes.append(f"external_image:{name}")
                    continue
                refs.append(ImageRef(part, name, el, attr, rid, rel.target_part, i, kind))
                i += 1
                break
    return refs, notes


# --------------------------------------------------------------------------- scan
def scan_ooxml(rc: RunContext, obj, family: str) -> dict:
    pkg = _package(obj)
    parts = sorted(pkg.iter_parts(), key=lambda p: str(p.partname))
    plan: dict = {"src_path": getattr(rc, "src_path", None), "units": {}, "unit_objs": {}, "generic": {}, "roots": {}, "rels": [], "core": [], "images": {},
                  "thumb": False, "family": family}
    payload_units: dict[str, int] = {}
    payload_regions: dict[str, int] = {}
    embedded = 0
    for part in parts:
        rc.deadline.check(str(part.partname))
        name = str(part.partname)
        ct = getattr(part, "content_type", "") or ""
        if EMBED_RE.search(name) or (name.startswith(("/ppt/media/", "/word/media/")) and not ct.startswith("image/")):
            embedded += 1
            rc.payloads.add("embedded_objects", "unsupported", 1, f"{name.rsplit('/', 1)[-1]} cannot be inspected")
            continue
        payload, mode = _classify(family, name)
        if not mode:
            continue
        try:
            root, generic = part_root(part)
        except Exception as exc:
            rc.payloads.add(payload, "failed", 1, f"{name} unreadable: {exc}")
            continue
        plan["roots"][name] = root
        if generic:
            plan["generic"][name] = (part, root)
        n_units = 0
        for unit in _build_units(mode, root, name):
            n_units += 1
            rc.counters["units_total"] += 1
            rc.counters["units_inspected"] += 1
            plan["unit_objs"][unit.unit_id] = unit
            findings = scan_text(rc.detect, unit.text, family)
            method = f"{family}_text" if unit.kind in ("visible", "value", "attr") else f"{family}_{unit.kind}"
            for f in findings:
                reg = finding_to_region(rc.detect, f, unit_id=unit.unit_id, space="text_anchor", source_method=method,
                                        anchor={"part": name, "kind": unit.kind})
                reg.handle = (unit, f.start, f.end)
                if unit.covers_hidden(f.start, f.end):
                    reg.reason += "; hidden (vanish) text"
                    rc.payloads.add("hidden_text", "sanitised", 1, "hidden text rewritten")
                if unit.kind == "deleted":
                    reg.reason += "; tracked-change deletion"
                    rc.payloads.add("hidden_text", "sanitised", 1, "tracked deletions rewritten")
                rc.regions.append(reg)
                payload_regions[payload] = payload_regions.get(payload, 0) + 1
        payload_units[payload] = payload_units.get(payload, 0) + n_units
        # structural census (inspected, never silently skipped)
        if family == "docx" and name == "/word/document.xml":
            rc.payloads.add("tables", "sanitised" if payload_regions.get("body_text") else "inspected", _count(root, W_NS, "tbl"))
            rc.payloads.add("text_boxes", "inspected", _count(root, W_NS, "txbxContent") + _count(root, MC_NS, "AlternateContent"))
            rc.payloads.add("hyperlinks", "inspected", _count(root, W_NS, "hyperlink"))
            rc.payloads.add("field_codes", "inspected", _count(root, W_NS, "instrText"))
            hid = _count(root, W_NS, "vanish") + _count(root, W_NS, "delText")
            rc.payloads.add("hidden_text", "inspected", hid, "vanish runs and tracked deletions are scanned")
            alt = _count(root, W_NS, "altChunk")
            if alt:
                embedded += alt
                rc.payloads.add("embedded_objects", "unsupported", alt, "altChunk (embedded alternate content) cannot be inspected")
            ole = _count(root, O_NS, "OLEObject") + _count(root, W_NS, "object")
            if ole:
                embedded += ole
                rc.payloads.add("embedded_objects", "unsupported", ole, "OLE objects cannot be inspected")
        if family == "docx" and payload == "comments":
            authors = {a.get(W("author")) for a in root.iter(W("comment")) if a.get(W("author"))}
            rc.payloads.add("comments", "preserved" if authors else "inspected", 0,
                            f"{len(authors)} comment author name(s) preserved" if authors else "")
        if family == "pptx" and payload == "body_text":
            if root.get("show") == "0":
                rc.payloads.add("hidden_text", "inspected", 1, f"hidden slide {name.rsplit('/', 1)[-1]} scanned like any slide")
            ole = _count(root, P_NS, "oleObj")
            if ole:
                embedded += ole
                rc.payloads.add("embedded_objects", "unsupported", ole, "OLE objects on slides cannot be inspected")
            rc.payloads.add("tables", "inspected", _count(root, A_NS, "tbl"))
        if family == "pptx" and name == "/ppt/commentAuthors.xml":
            rc.payloads.add("comments", "preserved", 0, "comment author list preserved")
    for payload, n in payload_units.items():
        rc.payloads.add(payload, "sanitised" if payload_regions.get(payload) else "inspected", n)
    orphans = _orphan_members(plan.get("src_path"), parts)
    if orphans:
        rc.warn(f"orphan_package_members_dropped:{len(orphans)} unreferenced zip member(s) are not carried into the output")
        rc.payloads.add("embedded_objects", "inspected", 0, f"{len(orphans)} unreferenced package member(s) dropped on save")
    if family == "pptx":
        for part in parts:
            if str(part.partname) == "/ppt/commentAuthors.xml":
                rc.payloads.add("comments", "preserved", 0, "comment author list preserved (policy)")
    # hyperlink targets -------------------------------------------------------------------------------
    for part in parts:
        for rid, rel in list(part.rels.items()):
            if rel.is_external and rel.reltype.endswith("/hyperlink"):
                fs = scan_text(rc.detect, rel.target_ref, family)
                rc.payloads.add("hyperlinks", "inspected", 1, "external link targets")
                for f in fs:
                    reg = finding_to_region(rc.detect, f, unit_id=f"rel:{part.partname}:{rid}", space="text_anchor",
                                            source_method="link_target", anchor={"part": str(part.partname), "rid": rid})
                    reg.handle = ("rel", rel, f)
                    rc.regions.append(reg)
                    plan["rels"].append((rel, reg, f))
            elif rel.is_external and rel.reltype.endswith("/oleObject"):
                embedded += 1
                rc.payloads.add("embedded_objects", "unsupported", 1, "linked OLE object")
    # core properties -----------------------------------------------------------------------------------
    cp = obj.core_properties
    n_core = 0
    for attr in CORE_ATTRS:
        v = getattr(cp, attr, None)
        if isinstance(v, str) and v:
            n_core += 1
            fs = scan_text(rc.detect, v, family)
            blank = rc.policy.strip_author_metadata and attr in ("author", "last_modified_by")
            for f in fs:
                reg = finding_to_region(rc.detect, f, unit_id=f"meta:core:{attr}", space="text_anchor", source_method="metadata", anchor={"field": attr})
                reg.handle = ("core", attr, v, fs)
                rc.regions.append(reg)
            if fs or blank:
                plan["core"].append((attr, v, fs, blank))
    rc.payloads.add("metadata", "inspected", n_core, "core properties")
    # thumbnails ------------------------------------------------------------------------------------------
    rels = getattr(pkg, "rels", None)
    if rels is None:
        rels = getattr(pkg, "_rels", {})
    plan["pkg_rels"] = rels
    thumbs = [rid for rid, r in list(rels.items()) if r.reltype.endswith("/thumbnail")]
    plan["thumb_rids"] = thumbs
    rc.payloads.add("thumbnails", "inspected", len(thumbs), "package thumbnail present" if thumbs else "")
    # images -----------------------------------------------------------------------------------------------
    refs, notes = collect_image_refs(parts, plan["roots"])
    for n in dict.fromkeys(notes):
        rc.warn(n)
        rc.payloads.add("embedded_images", "unsupported", 1, n.split(":")[0].replace("_", " "))
    plan["image_refs"] = refs
    rc.counters["images_total"] += len(refs)
    cache: dict[str, ImageScan | None] = {}
    img_regions_by_media: dict[str, list[Region]] = {}
    for ref in refs:
        blob = ref.target.blob
        sha = sha256_bytes(blob)
        ctype = getattr(ref.target, "content_type", "")
        if ctype not in IMAGE_CT_OK:
            rc.payloads.add("embedded_images", "unsupported", 1, f"{ctype or 'unknown'} images cannot be rasterised for OCR")
            continue
        if sha not in cache:
            try:
                if rc.counters["images_ocr"] + rc.counters["images_ocr_failed"] >= rc.limits.max_ocr_images:
                    raise ValueError("max OCR image count reached")
                im = open_image(io.BytesIO(blob), rc.limits.max_image_pixels)
                try:
                    cache[sha] = inspect_image(rc, im, unit_id=f"media:{sha[:12]}", space_label=str(ref.target.partname))
                finally:
                    im.close()
            except Exception as exc:
                cache[sha] = None
                rc.payloads.add("embedded_images", "failed", 1, f"image {str(ref.target.partname).rsplit('/', 1)[-1]}: {exc}")
        sc = cache[sha]
        if sc is None:
            continue
        if sc.ocr_status in ("failed", "unavailable") and rc.policy.ocr:
            rc.payloads.add("embedded_images", "failed" if sc.ocr_status == "failed" else "unsupported", 1,
                            f"image text not inspected ({sc.ocr_status})")
        mine = []
        for t in sc.regions:
            c = Region(**{**t.__dict__, "region_id": _occ_id(t.region_id, ref), "unit_id": ref.occurrence_id,
                          "anchor": {**t.anchor, "occurrence": ref.occurrence_id, "media": str(ref.target.partname)},
                          "merged_from": list(t.merged_from), "also_matched": list(t.also_matched)})
            c.handle = ("image", ref, sha)
            mine.append(c)
            rc.regions.append(c)
        if sc.exif_flags:
            rc.payloads.add("metadata", "inspected", 1, f"embedded image EXIF present ({', '.join(sc.exif_flags)}) - removed only on images that are re-encoded")
        img_regions_by_media.setdefault(sha, []).extend(mine)
        rc.payloads.add("embedded_images", "sanitised" if mine else "inspected", 1)
    plan["img_cache"] = cache
    plan["img_regions"] = img_regions_by_media
    if not refs:
        rc.payloads.add("embedded_images", "inspected", 0, "no embedded images")
    if embedded == 0:
        rc.payloads.add("embedded_objects", "inspected", 0, "none found")
    # fingerprints before changes ---------------------------------------------------------------------------
    for uid, u in plan["unit_objs"].items():
        plan["units"][uid] = {"digest_before": run_format_digest(u.container) if u.container is not None else ""}
    return plan


def _occ_id(rid: str, ref: ImageRef) -> str:
    return "R-" + hashlib.sha256(f"{rid}|{ref.occurrence_id}".encode()).hexdigest()[:16]


# --------------------------------------------------------------------------- apply
def _acting(r: Region) -> bool:
    return r.action in ("REDACT", "MASK", "GENERALISE") and r.verification_status != "failed"


def replace_image_occurrence(pkg, ref: ImageRef, new_bytes: bytes | None, rt_image: str) -> None:
    """Copy-on-write for ONE occurrence. ``None`` removes the picture element's image reference only."""
    if new_bytes is None:
        raise ValueError("removing a picture element is not performed automatically")
    img_part = pkg.get_or_add_image_part(io.BytesIO(new_bytes))
    new_rid = ref.owner.relate_to(img_part, rt_image)
    ref.elem.set(ref.attr, new_rid)
    old = ref.rid
    ref.rid = new_rid
    if old != new_rid and not _still_referenced(ref.owner, old):
        try:
            ref.owner.rels.pop(old)
        except Exception:
            pass


def _still_referenced(part, rid: str) -> bool:
    root, _g = part_root(part)
    for el in root.iter():
        for k, v in el.attrib.items():
            if v == rid and (k.startswith(f"{{{R_NS}}}") or k.startswith(f"{{{O_NS}}}")):
                return True
    return False


def apply_ooxml(rc: RunContext, obj, plan: dict, out_path: str) -> None:
    family = plan["family"]
    pkg = _package(obj)
    if family == "docx":
        from docx.opc.constants import RELATIONSHIP_TYPE as RT
    else:
        from pptx.opc.constants import RELATIONSHIP_TYPE as RT
    # 0. the EXPECTED post-edit text of every unit is derived from the original text + regions, never from what
    #    the editor claims to have done, so verification can catch an edit that silently did not happen
    simulate_expected(plan, rc.regions)
    # 1. text replacements, right-to-left inside each unit so earlier offsets stay valid ------------------
    by_unit: dict[str, list[Region]] = {}
    for r in rc.regions:
        if r.handle and isinstance(r.handle[0], TextUnit) and _acting(r):
            by_unit.setdefault(r.unit_id, []).append(r)
    touched = set()
    for uid, regs in by_unit.items():
        for r in sorted(regs, key=lambda x: -x.handle[1]):
            unit, s, e = r.handle
            ok = unit.replace_span(s, e, r.replacement or "")
            if not ok:
                r.verification_status, r.reason = "failed", "no editable text element covers this span"
        touched.add(uid)
    for name, (part, root) in plan["generic"].items():
        commit_part(part, root)
    # 2. hyperlink targets -------------------------------------------------------------------------------------
    for rel, reg, f in plan["rels"]:
        if _acting(reg):
            from .detect import make_replacement
            new_ref = rel.target_ref[:f.start] + make_replacement(reg.action, reg.category, f.raw, rc.policy.mask_keep_last) + rel.target_ref[f.end:]
            rel._target = new_ref
            rel.__dict__["target_ref"] = new_ref          # python-pptx caches target_ref as a lazyproperty
    # 3. core properties ------------------------------------------------------------------------------------------
    from .detect import make_replacement
    cp = obj.core_properties
    for attr, value, fs, blank in plan["core"]:
        new, pos = [], 0
        for f in sorted(fs, key=lambda x: x.start):
            if f.action in ("REDACT", "MASK", "GENERALISE"):
                new.append(value[pos:f.start])
                new.append(make_replacement(f.action, f.rule.category, f.raw, rc.policy.mask_keep_last))
                pos = f.end
        new.append(value[pos:])
        cleaned = "".join(new)
        if blank:
            cleaned = ""
            rc.payloads.add("metadata", "sanitised", 1, f"{attr} blanked by policy")
        setattr(cp, attr, cleaned)
    if plan["core"]:
        rc.payloads.add("metadata", "sanitised", len(plan["core"]), "core properties")
    # 4. images (copy-on-write per occurrence) ----------------------------------------------------------------------
    done_media: dict[str, bytes] = {}
    for sha, regs in plan["img_regions"].items():
        acting = [r for r in regs if _acting(r)]
        if not acting:
            continue
        refs_for = {id(r.handle[1]): r.handle[1] for r in acting}
        first_ref = next(iter(refs_for.values()))
        try:
            im = open_image(io.BytesIO(first_ref.target.blob), rc.limits.max_image_pixels)
            try:
                scan = plan["img_cache"][sha]
                if sha not in done_media:
                    # paint the union of every acting region of this media (region geometry is identical per occurrence)
                    uniq = {}
                    for r in acting:
                        uniq.setdefault(tuple(round(v, 1) for v in r.bbox), r)
                    done_media[sha] = render_sanitised(rc, im, list(uniq.values()), scan.fmt)
            finally:
                im.close()
            for ref in refs_for.values():
                replace_image_occurrence(pkg, ref, done_media[sha], RT.IMAGE)
            rc.payloads.add("embedded_images", "sanitised", len(refs_for), "copy-on-write per occurrence")
        except Exception as exc:
            for r in acting:
                r.verification_status, r.reason = "failed", f"image could not be rewritten: {exc}"
            rc.payloads.add("embedded_images", "failed", 1, str(exc))
    # 5. thumbnail: a rendered preview of the original content ----------------------------------------------------
    if plan["thumb_rids"] and any(_acting(r) for r in rc.regions):
        for rid in plan["thumb_rids"]:
            try:
                plan["pkg_rels"].pop(rid)
                rc.payloads.add("thumbnails", "sanitised", 1, "removed (rendered preview of original content)")
            except Exception:
                rc.payloads.add("thumbnails", "failed", 1, "could not remove thumbnail")
    obj.save(out_path)


def simulate_expected(plan: dict, regions: list[Region]) -> None:
    """Fill plan['units'][*]['text_after_sha'] as if the acting regions had been applied (verify-only mode)."""
    acting: dict[str, list[Region]] = {}
    for r in regions:
        if r.handle and isinstance(r.handle[0], TextUnit) and _acting(r):
            acting.setdefault(r.unit_id, []).append(r)
    for uid, unit in plan["unit_objs"].items():
        text = unit.text
        for r in sorted(acting.get(uid, []), key=lambda x: -x.handle[1]):
            _u, s, e = r.handle
            text = text[:s] + (r.replacement or "") + text[e:]
        plan["units"][uid]["text_after_sha"] = hashlib.sha256(text.encode()).hexdigest()
        plan["units"][uid]["touched"] = uid in acting


def _orphan_members(path, parts) -> list[str]:
    """Zip members that no relationship reaches (python-docx / python-pptx drop them when saving)."""
    if not path:
        return []
    try:
        with zipfile.ZipFile(path) as z:
            names = {n for n in z.namelist() if not n.endswith("/")}
    except Exception:
        return []
    reachable = {str(p.partname).lstrip("/") for p in parts} | {"[Content_Types].xml"}
    return sorted(n for n in names if n not in reachable and not n.endswith(".rels"))
