"""Run-aware text units for WordprocessingML and DrawingML (A3).

A TextUnit is the text of one paragraph (or one attribute / field instruction / single value),
together with the exact XML elements that hold each character. A replacement is applied by
character offset, so:

  * a span that crosses several runs is rewritten in place (the first run keeps its formatting and
    receives the replacement; the covered parts of the other runs are removed),
  * only the exact occurrence found by the detector is touched - the same text elsewhere in the
    document is never replaced unless it was detected as its own occurrence,
  * run properties (``rPr``) are never rewritten, only text nodes.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Iterator

from lxml import etree

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
XML_NS = "http://www.w3.org/XML/1998/namespace"


def W(tag: str) -> str:
    return f"{{{W_NS}}}{tag}"


def A(tag: str) -> str:
    return f"{{{A_NS}}}{tag}"


SAFE_PARSER = etree.XMLParser(resolve_entities=False, no_network=True, huge_tree=False, remove_blank_text=False)


def parse_xml(blob: bytes):
    return etree.fromstring(blob, SAFE_PARSER)


def serialise_xml(root) -> bytes:
    return etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)


@dataclass
class Seg:
    elem: object | None             # lxml element holding the text, or None for synthetic chars (tab / line break)
    text: str
    attr: str | None = None         # when text lives in an attribute
    hidden: bool = False
    tracked_deletion: bool = False


@dataclass
class TextUnit:
    unit_id: str
    kind: str                       # visible | deleted | instr | value | attr
    part: str
    segs: list[Seg] = field(default_factory=list)
    offsets: list[tuple[int, int]] = field(default_factory=list)
    text: str = ""
    container: object | None = None      # the paragraph element (for formatting digests)

    def finalise(self) -> "TextUnit":
        pos, offs = 0, []
        for s in self.segs:
            offs.append((pos, pos + len(s.text)))
            pos += len(s.text)
        self.offsets, self.text = offs, "".join(s.text for s in self.segs)
        return self

    def segments_in(self, start: int, end: int) -> list[Seg]:
        return [s for s, (a, b) in zip(self.segs, self.offsets) if b > start and a < end]

    def covers_hidden(self, start: int, end: int) -> bool:
        return any(s.hidden for s in self.segments_in(start, end))

    def replace_span(self, start: int, end: int, repl: str) -> bool:
        """Replace text[start:end]. Returns False if no editable element covers the span."""
        first = True
        done = False
        to_clean = []
        for seg, (s0, s1) in zip(self.segs, self.offsets):
            if seg.elem is None or s1 <= start or s0 >= end:
                continue
            a, b = max(start, s0) - s0, min(end, s1) - s0
            old = seg.text
            seg.text = old[:a] + (repl if first else "") + old[b:]
            first = False
            done = True
            _write_seg(seg)
            if not seg.text:
                to_clean.append(seg)
        for seg in to_clean:
            _drop_empty_run(seg)
        return done


def _write_seg(seg: Seg) -> None:
    if seg.attr:
        seg.elem.set(seg.attr, seg.text)
        return
    seg.elem.text = seg.text
    if seg.text != seg.text.strip() or "  " in seg.text:
        seg.elem.set(f"{{{XML_NS}}}space", "preserve")


def _drop_empty_run(seg: Seg) -> None:
    el = seg.elem
    if seg.attr or el is None:
        return
    parent = el.getparent()
    if parent is None or etree.QName(parent).localname != "r":
        return
    others = [c for c in parent if c is not el and etree.QName(c).localname not in ("rPr",)]
    if not others:
        gp = parent.getparent()
        if gp is not None:
            gp.remove(parent)


# --------------------------------------------------------------------------- builders
def _owner_paragraph(el, ptag: str):
    p = el.getparent()
    while p is not None and p.tag != ptag:
        p = p.getparent()
    return p


def _is_hidden_run(run) -> bool:
    if run is None:
        return False
    rpr = run.find(W("rPr"))
    if rpr is None:
        return False
    v = rpr.find(W("vanish"))
    return v is not None and v.get(W("val"), "true") not in ("0", "false", "off")


def word_units(root, part: str) -> Iterator[TextUnit]:
    """Visible text, tracked-deletion text and field instructions for every w:p (in document order)."""
    ptag = W("p")
    for n, p in enumerate(root.iter(ptag)):
        vis = TextUnit(f"{part}:p{n}", "visible", part, container=p)
        dele = TextUnit(f"{part}:p{n}:del", "deleted", part, container=p)
        instr_idx = 0
        for el in p.iter(W("t"), W("delText"), W("tab"), W("br"), W("cr"), W("noBreakHyphen"), W("instrText")):
            if _owner_paragraph(el, ptag) is not p:
                continue
            tag = etree.QName(el).localname
            run = el.getparent()
            if tag == "t":
                vis.segs.append(Seg(el, el.text or "", hidden=_is_hidden_run(run)))
            elif tag == "delText":
                dele.segs.append(Seg(el, el.text or "", tracked_deletion=True))
            elif tag == "instrText":
                if el.text and el.text.strip():
                    yield TextUnit(f"{part}:p{n}:i{instr_idx}", "instr", part, [Seg(el, el.text)]).finalise()
                instr_idx += 1
            elif tag in ("tab",):
                vis.segs.append(Seg(None, "\t"))
            elif tag in ("br", "cr"):
                vis.segs.append(Seg(None, "\n"))
            elif tag == "noBreakHyphen":
                vis.segs.append(Seg(None, "-"))
        if vis.segs:
            yield vis.finalise()
        if dele.segs:
            yield dele.finalise()


def drawing_units(root, part: str) -> Iterator[TextUnit]:
    """DrawingML paragraphs (PowerPoint shapes, charts, diagrams): a:p -> a:r/a:t and a:fld/a:t."""
    ptag = A("p")
    for n, p in enumerate(root.iter(ptag)):
        u = TextUnit(f"{part}:p{n}", "visible", part, container=p)
        for el in p.iter(A("t"), A("br")):
            if _owner_paragraph(el, ptag) is not p:
                continue
            if etree.QName(el).localname == "br":
                u.segs.append(Seg(None, "\n"))
            else:
                u.segs.append(Seg(el, el.text or ""))
        if any(s.elem is not None for s in u.segs):
            yield u.finalise()


def value_units(root, part: str, xpath_tags: list[str], prefix: str = "v") -> Iterator[TextUnit]:
    """One unit per element whose text is a single value (chart caches, comment text, property values)."""
    k = 0
    for tag in xpath_tags:
        for el in root.iter(tag):
            if el.text and el.text.strip() and len(el) == 0:
                yield TextUnit(f"{part}:{prefix}{k}", "value", part, [Seg(el, el.text)]).finalise()
                k += 1


def attr_units(root, part: str, tag_attrs: list[tuple[str, str]], prefix: str = "a") -> Iterator[TextUnit]:
    """One unit per attribute value, e.g. shape alt-text (descr/title) or author names."""
    k = 0
    for tag, attr in tag_attrs:
        for el in root.iter(tag):
            v = el.get(attr)
            if v and v.strip():
                yield TextUnit(f"{part}:{prefix}{k}:{attr}", "attr", part, [Seg(el, v, attr=attr)]).finalise()
                k += 1


def all_text_nodes(root, part: str, prefix: str = "n") -> Iterator[TextUnit]:
    """Fallback for generic XML parts (custom properties, customXml): every non-blank text node is a unit."""
    k = 0
    for el in root.iter():
        if isinstance(el.tag, str) and el.text and el.text.strip() and len(el) == 0:
            yield TextUnit(f"{part}:{prefix}{k}", "value", part, [Seg(el, el.text)]).finalise()
            k += 1


def run_format_digest(unit_root_el) -> str:
    """Digest of the run-property XML of every run under an element (formatting-preservation check)."""
    h = hashlib.sha256()
    for r in unit_root_el.iter(W("r"), A("r")):
        for tag in (W("rPr"), A("rPr")):
            pr = r.find(tag)
            if pr is not None:
                h.update(etree.tostring(pr, method="c14n"))
        h.update(b"|")
    return h.hexdigest()[:16]
