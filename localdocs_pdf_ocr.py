"""Add local OCR as invisible, searchable Unicode text to existing PDF pages.

Only pypdf, pypdfium2, Pillow and the sibling localdocs_ocr are required.
No fonts, models or other resources are downloaded by this module.
"""

import argparse
import ctypes
import json
import math
import os
from pathlib import Path
import tempfile
import time

from pypdf import PdfReader, PdfWriter
from pypdf.generic import (
    ArrayObject, DecodedStreamObject, DictionaryObject, NameObject,
    NumberObject, TextStringObject,
)
import pypdfium2 as pdfium


def _stream(writer, data):
    obj = DecodedStreamObject()
    obj.set_data(data)
    return writer._add_object(obj)


def _font(writer, characters):
    """One two-byte CID per Unicode scalar; widths are all 1000 units."""
    descriptor = DictionaryObject({
        NameObject("/Type"): NameObject("/FontDescriptor"),
        NameObject("/FontName"): NameObject("/LocalDocsOCR"),
        NameObject("/Flags"): NumberObject(4),
        NameObject("/FontBBox"): ArrayObject([NumberObject(v) for v in (0, -200, 1000, 1000)]),
        NameObject("/ItalicAngle"): NumberObject(0),
        NameObject("/Ascent"): NumberObject(1000),
        NameObject("/Descent"): NumberObject(-200),
        NameObject("/CapHeight"): NumberObject(1000),
        NameObject("/StemV"): NumberObject(80),
    })
    descendant = DictionaryObject({
        NameObject("/Type"): NameObject("/Font"),
        NameObject("/Subtype"): NameObject("/CIDFontType2"),
        NameObject("/BaseFont"): NameObject("/LocalDocsOCR"),
        NameObject("/CIDSystemInfo"): DictionaryObject({
            NameObject("/Registry"): TextStringObject("Adobe"),
            NameObject("/Ordering"): TextStringObject("Identity"),
            NameObject("/Supplement"): NumberObject(0),
        }),
        NameObject("/FontDescriptor"): writer._add_object(descriptor),
        NameObject("/DW"): NumberObject(1000),
        NameObject("/CIDToGIDMap"): NameObject("/Identity"),
    })
    cmap = [
        "/CIDInit /ProcSet findresource begin", "12 dict begin", "begincmap",
        "/CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def",
        "/CMapName /LocalDocsOCRUnicode def", "/CMapType 2 def",
        "1 begincodespacerange", "<0000> <FFFF>", "endcodespacerange",
    ]
    entries = list(characters.items())
    for start in range(0, len(entries), 100):
        chunk = entries[start:start + 100]
        cmap.append(f"{len(chunk)} beginbfchar")
        for char, cid in chunk:
            cmap.append(f"<{cid:04X}> <{char.encode('utf-16-be').hex().upper()}>")
        cmap.append("endbfchar")
    cmap.extend(["endcmap", "CMapName currentdict /CMap defineresource pop", "end", "end"])
    return writer._add_object(DictionaryObject({
        NameObject("/Type"): NameObject("/Font"),
        NameObject("/Subtype"): NameObject("/Type0"),
        NameObject("/BaseFont"): NameObject("/LocalDocsOCR"),
        NameObject("/Encoding"): NameObject("/Identity-H"),
        NameObject("/DescendantFonts"): ArrayObject([writer._add_object(descendant)]),
        NameObject("/ToUnicode"): _stream(writer, ("\n".join(cmap) + "\n").encode("ascii")),
    }))


def _pixel_transform(page, width, height):
    """Invert PDFium's render viewport, including intrinsic Rotate and boxes.

    The viewport uses the actual bitmap dimensions (rounded at render time),
    so its affine inverse also accounts for the subpixel rounding at 72/dpi.
    DeviceToPage returns original PDF user coordinates, not rotated coordinates.
    """
    def corner(x, y):
        px, py = ctypes.c_double(), ctypes.c_double()
        ok = pdfium.raw.FPDF_DeviceToPage(
            page.raw, 0, 0, width, height, 0, x, y,
            ctypes.byref(px), ctypes.byref(py),
        )
        if not ok:
            raise ValueError("Cannot map rendered pixels to PDF coordinates")
        return px.value, py.value

    origin = corner(0, 0)
    right = corner(width, 0)
    bottom = corner(0, height)

    def transform(point):
        x, y = map(float, point)
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("Non-finite OCR box coordinate")
        return (
            origin[0] + x * (right[0] - origin[0]) / width + y * (bottom[0] - origin[0]) / height,
            origin[1] + x * (right[1] - origin[1]) / width + y * (bottom[1] - origin[1]) / height,
        )
    return transform


def map_lines_to_pdf(lines, transform, page=None):
    """Copy OCR lines, adding PDF-space ``pdf_box`` and page identity.

    The original pixel ``box`` is retained so callers can still map back to the
    rendered bitmap. The transform already includes the page's intrinsic
    /Rotate and crop/media offsets (see _pixel_transform).
    """
    out = []
    for line in lines or []:
        if not isinstance(line, dict):
            continue
        box = line.get("box")
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        try:
            mapped = [list(transform(point)) for point in box]
        except Exception:
            continue
        item = dict(line)
        item["pdf_box"] = mapped
        item["pdf_page"] = page
        out.append(item)
    return out


def _number(value):
    return f"{value:.10f}".rstrip("0").rstrip(".") or "0"


def _text_stream(lines, transform, font_name, characters):
    commands = ["q", "BT", "3 Tr", "0 Tc", "0 Tw", "0 Ts"]
    count = 0
    for line in lines:
        text = str(line.get("text", ""))
        if not text.strip():
            continue
        box = line["box"]
        if len(box) != 4 or any(len(point) != 2 for point in box):
            raise ValueError("OCR boxes must contain four [x, y] points")
        # OCR order: top-left, top-right, bottom-right, bottom-left.
        # Baseline follows the lower edge; the vertical axis points upwards.
        # map_lines_to_pdf() may have mapped the boxes already; reuse them then.
        mapped = line.get("pdf_box")
        if not isinstance(mapped, (list, tuple)) or len(mapped) != 4:
            mapped = [transform(p) for p in box]
        top_left, _, bottom_right, bottom_left = mapped
        dx = bottom_right[0] - bottom_left[0]
        dy = bottom_right[1] - bottom_left[1]
        ux = top_left[0] - bottom_left[0]
        uy = top_left[1] - bottom_left[1]
        width, height = math.hypot(dx, dy), math.hypot(ux, uy)
        if width <= 0 or height <= 0:
            continue
        cids = []
        for char in text:
            if char not in characters:
                if len(characters) >= 65535:
                    raise ValueError("OCR text exceeds the two-byte CID capacity")
                # Reject lone surrogates before producing an invalid CMap.
                char.encode("utf-16-be")
                characters[char] = len(characters) + 1
            cids.append(f"{characters[char]:04X}")
        matrix = (dx / width, dy / width, ux / height, uy / height, *bottom_left)
        commands.extend([
            f"{font_name} {_number(height)} Tf",
            f"{_number(100 * width / (len(text) * height))} Tz",
            " ".join(_number(v) for v in matrix) + " Tm",
            "<" + "".join(cids) + "> Tj",
        ])
        count += len(text)
    commands.extend(["ET", "Q"])
    return ("\n".join(commands) + "\n").encode("ascii"), count


def _append_layer(writer, page, overlay):
    # Preserve all original content bytes and isolate their graphics state.
    contents = page.get("/Contents")
    original = []
    if contents is not None:
        resolved = contents.get_object()
        original = list(resolved) if isinstance(resolved, ArrayObject) else [contents]
    page[NameObject("/Contents")] = ArrayObject([
        _stream(writer, b"q\n"), *original, _stream(writer, b"\nQ\n"),
        _stream(writer, overlay),
    ])


def make_searchable_pdf(src_pdf, dst_pdf, lang="rus+eng+pol", device="cpu",
                        engine="auto", dpi=200, max_pages=None, progress=None,
                        *, force=False, rotation=None, spreads=False):
    """Copy a PDF and add an invisible OCR layer to pages lacking text.

    ``force=True`` also OCRs pages with existing text (the old text is retained).
    ``max_pages`` limits inspection/OCR to the first N pages; every source page
    is retained. ``progress``, if provided, receives a dict after each inspected
    page: page (1-based), total (inspection limit), pages, pages_ocr, chars, engine.
    Return pages (all output pages), pages_ocr (pages with added text), chars
    (inserted Unicode characters), seconds and engine (actual engines joined
    with '+', or 'none'). Exceptions leave the destination unchanged.
    """
    # Probe entitlement: make_searchable_pdf
    import probe_licensing as _pl
    _pl.require('ocr')
    started = time.perf_counter()
    src, dst = Path(src_pdf).resolve(), Path(dst_pdf).resolve()
    if src == dst or (dst.exists() and os.path.samefile(src, dst)):
        raise ValueError("Source and destination PDFs must differ")
    if device not in ("cpu", "gpu"):
        raise ValueError("device must be cpu or gpu")
    if engine not in ("auto", "own", "rapid", "tesseract", "hybrid"):
        raise ValueError("Invalid OCR engine")
    if isinstance(dpi, bool) or not isinstance(dpi, (int, float)) or not math.isfinite(dpi) or dpi <= 0:
        raise ValueError("dpi must be a finite positive number")
    if max_pages is not None and (type(max_pages) is not int or max_pages < 0):
        raise ValueError("max_pages must be a nonnegative integer or None")
    if progress is not None and not callable(progress):
        raise TypeError("progress must be callable or None")
    if rotation not in (None, 0, 90, 180, 270):
        raise ValueError("rotation must be 0/90/180/270")
    if not isinstance(spreads, bool):
        raise ValueError("spreads must be bool")
    import localdocs_ocr

    characters, engines, layered = {}, set(), []
    pages_ocr = chars = 0
    writer = PdfWriter()
    try:
        with src.open("rb") as source:
            reader = PdfReader(source)
            writer.clone_document_from_reader(reader)
            pages = len(writer.pages)
            limit = pages if max_pages is None else min(pages, max_pages)
            with pdfium.PdfDocument(str(src)) as document:
                for index in range(limit):
                    page = writer.pages[index]
                    if force or not (page.extract_text() or "").strip():
                        rendered_page = document[index]
                        try:
                            bitmap = rendered_page.render(scale=dpi / 72, rotation=0)
                            try:
                                image = bitmap.to_pil()
                                try:
                                    transform = _pixel_transform(rendered_page, *image.size)
                                    result = localdocs_ocr.ocr_image(
                                        image, lang, device, engine,
                                        rotation=rotation or None, spreads=spreads)
                                finally:
                                    image.close()
                            finally:
                                bitmap.close()
                        finally:
                            rendered_page.close()
                        engines.add(str(result.get("engine", engine)))
                        resources = DictionaryObject()
                        existing = page.get("/Resources")
                        if existing is not None:
                            resources.update(existing.get_object())
                        fonts = DictionaryObject()
                        existing_fonts = resources.get("/Font")
                        if existing_fonts is not None:
                            fonts.update(existing_fonts.get_object())
                        name = "/LocalDocsOCR"
                        suffix = 0
                        while name in fonts:
                            suffix += 1
                            name = f"/LocalDocsOCR{suffix}"
                        mapped_lines = map_lines_to_pdf(result["lines"], transform, page=index + 1)
                        overlay, added = _text_stream(mapped_lines, transform, name, characters)
                        if added:
                            resources[NameObject("/Font")] = fonts
                            page[NameObject("/Resources")] = resources
                            _append_layer(writer, page, overlay)
                            layered.append((fonts, name))
                            chars += added
                            pages_ocr += 1
                    if progress is not None:
                        progress(dict(page=index + 1, total=limit, pages=pages,
                                      pages_ocr=pages_ocr, chars=chars,
                                      engine="+".join(sorted(engines)) or "none"))
            if layered:
                font = _font(writer, characters)
                for fonts, name in layered:
                    fonts[NameObject(name)] = font
            # Keep the source open until all lazy PDF objects have been written.
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(mode="wb", prefix=".localdocs-ocr-",
                                                 suffix=".pdf", dir=dst.parent,
                                                 delete=False) as output:
                    temporary = output.name
                    writer.write(output)
                    output.flush()
                    os.fsync(output.fileno())
                os.replace(temporary, dst)
                temporary = None
            finally:
                if temporary is not None:
                    os.unlink(temporary)
    finally:
        writer.close()
    return dict(pages=pages, pages_ocr=pages_ocr, chars=chars,
                seconds=time.perf_counter() - started,
                engine="+".join(sorted(engines)) or "none")


def main(argv=None):
    """CLI: python localdocs_pdf_ocr.py src.pdf dst.pdf [options]."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("src_pdf")
    parser.add_argument("dst_pdf")
    parser.add_argument("--lang", default="rus+eng+pol")
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--engine", choices=("auto", "own", "rapid", "tesseract"), default="auto")
    parser.add_argument("--dpi", type=float, default=200)
    parser.add_argument("--max-pages", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    result = make_searchable_pdf(**vars(args))
    print(json.dumps(result, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
