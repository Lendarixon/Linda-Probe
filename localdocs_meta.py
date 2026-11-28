"""LocalDocs file metadata: read-only, never raises.

file_meta(path, detailed=True, max_pages=40) -> dict (JSON-compatible,
strings only str). Errors are collected in field "errors": [str].
Read-only access to the file. Files > 2 GiB are not opened.
For PDF, at most 64 MiB is scanned for %%EOF/Linearized markers.
"""

import datetime
import os
import re
import zipfile

ROOT_2GB = 2 * 1024 * 1024 * 1024
PDF_SCAN_MAX = 64 * 1024 * 1024
FONTS_HARD_LIMIT = 40

_IMAGE_EXTS = {
    ".png", ".jpg", ".jpeg", ".tif", ".tiff",
    ".bmp", ".gif", ".webp",
}

_SUBSET_RE = re.compile(r"^[A-Z]{6}\+")


def _s(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8", "replace")
        except Exception:
            return repr(value)
    try:
        return str(value)
    except Exception:
        return ""


def _iso_mtime(path):
    try:
        ts = os.path.getmtime(path)
        dt = datetime.datetime.fromtimestamp(
            ts, tz=datetime.timezone.utc
        ).astimezone()
        return dt.isoformat()
    except Exception as e:
        return ""


def _parse_pdf_date(raw):
    """Parse PDF date D:YYYYMMDDHHmmSSOHH'mm into ISO; else original string."""
    text = _s(raw)
    if not text:
        return ""
    m = re.match(
        r"^D:(\d{4})(\d{2})?(\d{2})?(\d{2})?(\d{2})?(\d{2})?"
        r"(?:([Zz]|[+\-])(\d{2})?'?(\d{2})?'?)?$",
        text.strip(),
    )
    if not m:
        return text
    try:
        year = int(m.group(1))
        mon = int(m.group(2) or 1)
        day = int(m.group(3) or 1)
        hh = int(m.group(4) or 0)
        mm = int(m.group(5) or 0)
        ss = int(m.group(6) or 0)
        tz_sign = m.group(7)
        if tz_sign is None or tz_sign in ("Z", "z"):
            tz = datetime.timezone.utc
        else:
            tzh = int(m.group(8) or 0)
            tzm = int(m.group(9) or 0)
            off = datetime.timedelta(hours=tzh, minutes=tzm)
            if tz_sign == "-":
                off = -off
            tz = datetime.timezone(off)
        dt = datetime.datetime(year, mon, day, hh, mm, ss, tzinfo=tz)
        return dt.isoformat()
    except Exception:
        return text


def _dt_for_compare(iso_text):
    try:
        return datetime.datetime.fromisoformat(iso_text)
    except Exception:
        return None


def _count_eof_and_linearized(path, errors):
    incremental_updates = 0
    linearized = False
    try:
        count = 0
        leftover = b""
        total = 0
        first_chunk = None
        with open(path, "rb") as f:
            while True:
                if total >= PDF_SCAN_MAX:
                    break
                want = min(1024 * 1024, PDF_SCAN_MAX - total)
                chunk = f.read(want)
                if not chunk:
                    break
                if first_chunk is None:
                    first_chunk = chunk[:1024]
                total += len(chunk)
                data = leftover + chunk
                count += data.count(b"%%EOF")
                # count() on overlap double-counts only if marker spans
                # boundary; keep tail to recount safely: subtract possible
                # boundary double count by recounting tail below.
                leftover = chunk[-5:] if len(chunk) >= 5 else chunk
                # fix boundary double-count: recount using streaming below
            # Recompute boundary-safe: simplest robust recount on overlap
            # is already handled since %%EOF is 5 bytes and leftover is 5;
            # overlapping occurrence counted in next data; the previous
            # data tail occurrence may be double counted. Correct by
            # scanning with explicit overlap handling:
        # Redo precisely with overlap-safe streaming (cheap, <=64MB):
        count2 = 0
        prev_tail = b""
        total2 = 0
        with open(path, "rb") as f:
            while True:
                if total2 >= PDF_SCAN_MAX:
                    break
                chunk = f.read(min(1024 * 1024, PDF_SCAN_MAX - total2))
                if not chunk:
                    break
                total2 += len(chunk)
                buf = prev_tail + chunk
                # count occurrences fully inside buf, then keep tail
                # to avoid double counting: count, then tail carries over
                start = 0
                # count all, but occurrences starting in prev_tail region
                # were already counted except those spanning boundary;
                # with tail=4 bytes, spanning ones were NOT fully present
                # before, so full recount minus tail-region recount works:
                # simpler: slide manually
                idx = buf.find(b"%%EOF")
                n = 0
                while idx != -1:
                    # only count if end position is within new chunk part
                    if idx + 5 > len(prev_tail):
                        n += 1
                    idx = buf.find(b"%%EOF", idx + 1)
                count2 += n
                prev_tail = chunk[-4:] if len(chunk) >= 4 else chunk
        count = count2
        incremental_updates = max(0, count - 1)
        if first_chunk is None:
            with open(path, "rb") as f:
                first_chunk = f.read(1024)
        linearized = b"/Linearized" in (first_chunk or b"")[:1024]
        # pdf version from header
        pdf_version = ""
        head = (first_chunk or b"")[:1024]
        mv = re.search(rb"%PDF-(\d+\.\d+)", head)
        if mv:
            try:
                pdf_version = mv.group(1).decode("ascii")
            except Exception:
                pdf_version = ""
    except Exception as e:
        errors.append("eof_scan: %s" % _s(e))
        incremental_updates = 0
        linearized = False
        pdf_version = pdf_version if "pdf_version" in dir() else ""
    return incremental_updates, linearized, pdf_version


def _name_str(obj):
    try:
        s = str(obj)
        if s.startswith("/"):
            return s[1:]
        return s
    except Exception:
        return ""


def _font_info(fntdict, get_obj):
    try:
        obj = get_obj(fntdict)
    except Exception:
        obj = fntdict
    try:
        d = dict(obj.items()) if hasattr(obj, "items") else {}
    except Exception:
        d = {}
    base_raw = _s(d.get("/BaseFont", ""))
    if base_raw.startswith("/"):
        base_raw = base_raw[1:]
    subset = bool(_SUBSET_RE.match(base_raw))
    base_font = _SUBSET_RE.sub("", base_raw)
    subtype = _name_str(d.get("/Subtype", ""))
    # encoding
    enc = d.get("/Encoding", None)
    try:
        enc_resolved = get_obj(enc) if enc is not None else None
    except Exception:
        enc_resolved = enc
    encoding = "builtin"
    try:
        if enc_resolved is None:
            encoding = "builtin"
        elif hasattr(enc_resolved, "items"):
            ed = dict(enc_resolved.items())
            if "/Differences" in ed:
                encoding = "custom Differences"
            elif "/BaseEncoding" in ed:
                encoding = _name_str(ed.get("/BaseEncoding", "")) or "builtin"
            else:
                encoding = "builtin"
        else:
            encoding = _name_str(enc_resolved) or "builtin"
    except Exception:
        encoding = "builtin"
    # ToUnicode
    has_tounicode = "/ToUnicode" in d
    # embedded / descendant
    embedded = False
    descendant = ""
    try:
        desc = d.get("/DescendantFonts", None)
        fdesc = d.get("/FontDescriptor", None)
        targets = []
        if desc is not None:
            try:
                arr = get_obj(desc)
                items = list(arr) if hasattr(arr, "__iter__") else [arr]
                for it in items:
                    try:
                        r = get_obj(it)
                    except Exception:
                        r = it
                    try:
                        rd = dict(r.items()) if hasattr(r, "items") else {}
                    except Exception:
                        rd = {}
                    st = _name_str(rd.get("/Subtype", ""))
                    if st in ("CIDFontType0", "CIDFontType2") and not descendant:
                        descendant = st
                    fd = rd.get("/FontDescriptor", None)
                    if fd is not None:
                        try:
                            targets.append(get_obj(fd))
                        except Exception:
                            targets.append(fd)
            except Exception:
                pass
        if fdesc is not None:
            try:
                targets.append(get_obj(fdesc))
            except Exception:
                targets.append(fdesc)
        for t in targets:
            try:
                td = dict(t.items()) if hasattr(t, "items") else {}
            except Exception:
                continue
            if ("/FontFile" in td or "/FontFile2" in td
                    or "/FontFile3" in td):
                embedded = True
                break
    except Exception:
        pass
    return {
        "base_font": _s(base_font),
        "subset": bool(subset),
        "subtype": _s(subtype),
        "encoding": _s(encoding),
        "embedded": bool(embedded),
        "has_tounicode": bool(has_tounicode),
        "descendant": _s(descendant),
    }


def _collect_fonts_and_images(reader, get_obj, limit_pages):
    fonts = []
    seen = set()
    has_image_first = False
    try:
        n = len(reader.pages)
    except Exception:
        n = 0
    pages_to_scan = min(n, max(0, limit_pages), FONTS_HARD_LIMIT)

    def walk_resources(res, depth, collect_fonts, find_images):
        nonlocal has_image_first
        found_fonts = []
        found_image = False
        try:
            r = get_obj(res)
        except Exception:
            return found_fonts, False
        try:
            rd = dict(r.items()) if hasattr(r, "items") else {}
        except Exception:
            return found_fonts, False
        # fonts
        if collect_fonts:
            try:
                fdict = rd.get("/Font", None)
                if fdict is not None:
                    try:
                        fobj = get_obj(fdict)
                        fitems = dict(fobj.items())
                    except Exception:
                        fitems = {}
                    for _k, v in fitems.items():
                        try:
                            info = _font_info(v, get_obj)
                        except Exception:
                            continue
                        key = (info["base_font"], info["subtype"],
                               info["encoding"])
                        if key not in seen:
                            seen.add(key)
                            found_fonts.append(info)
            except Exception:
                pass
        # xobjects
        try:
            xo = rd.get("/XObject", None)
            if xo is not None:
                try:
                    xobj = get_obj(xo)
                    xitems = dict(xobj.items())
                except Exception:
                    xitems = {}
                for _k, v in xitems.items():
                    try:
                        o = get_obj(v)
                    except Exception:
                        continue
                    try:
                        od = dict(o.items()) if hasattr(o, "items") else {}
                    except Exception:
                        continue
                    st = _name_str(od.get("/Subtype", ""))
                    if st == "Image":
                        found_image = True
                    elif st == "Form" and depth < 3:
                        sub = od.get("/Resources", None)
                        if sub is not None:
                            ff, fi = walk_resources(
                                sub, depth + 1, collect_fonts, True)
                            found_fonts.extend(ff)
                            if fi:
                                found_image = True
        except Exception:
            pass
        return found_fonts, found_image

    for i in range(pages_to_scan):
        try:
            page = reader.pages[i]
            res = page.get("/Resources", None)
            if res is None:
                continue
            ff, fi = walk_resources(res, 0, True, True)
            fonts.extend(ff)
            if i == 0 and fi:
                has_image_first = True
        except Exception:
            continue
    # images-only check needs first page even if pages > limit? first page
    # is within scanned range when limit >= 1; if limit == 0, check first
    # page separately for images without collecting fonts.
    if pages_to_scan == 0 and n >= 1:
        try:
            page = reader.pages[0]
            res = page.get("/Resources", None)
            if res is not None:
                _ff, fi = walk_resources(res, 0, False, True)
                if fi:
                    has_image_first = True
        except Exception:
            pass
    return fonts, has_image_first


def _pdf_detail(path, errors, max_pages):
    out = {}
    try:
        from pypdf import PdfReader
    except Exception as e:
        errors.append("pypdf import: %s" % _s(e))
        return out
    inc, lin, ver = _count_eof_and_linearized(path, errors)
    out["incremental_updates"] = int(inc)
    out["linearized"] = bool(lin)
    out["pdf_version"] = _s(ver)
    try:
        reader = PdfReader(path, strict=False)
    except Exception as e:
        errors.append("pdf read: %s" % _s(e))
        return out
    try:
        out["pages"] = int(len(reader.pages))
    except Exception:
        out["pages"] = 0
    try:
        out["encrypted"] = bool(getattr(reader, "is_encrypted", False))
    except Exception:
        out["encrypted"] = False
    # metadata
    try:
        md = reader.metadata
    except Exception as e:
        errors.append("pdf metadata: %s" % _s(e))
        md = None
    def mget(*names):
        if md is None:
            return ""
        for nm in names:
            try:
                v = getattr(md, nm, None)
                if v is not None and _s(v) != "":
                    return _s(v)
            except Exception:
                continue
        # raw dict fallback
        try:
            d = dict(md.items()) if hasattr(md, "items") else {}
            for k, v in d.items():
                ks = _s(k).lstrip("/")
                if ks in names or ("/" + ks) in names:
                    return _s(v)
        except Exception:
            pass
        return ""
    out["producer"] = mget("producer", "/Producer")
    out["creator"] = mget("creator", "/Creator")
    out["title"] = mget("title", "/Title")
    out["author"] = mget("author", "/Author")
    out["subject"] = mget("subject", "/Subject")
    raw_created = mget("creation_date", "creationDate", "/CreationDate")
    raw_modified = mget("modification_date", "modificationDate", "/ModDate")
    # raw fallback may be datetime -> str it; parse if looks like PDF date
    out["created"] = _parse_pdf_date(raw_created) if raw_created else ""
    out["modified"] = _parse_pdf_date(raw_modified) if raw_modified else ""
    # modified_after_creation
    try:
        dc = _dt_for_compare(out["created"])
        dm = _dt_for_compare(out["modified"])
        out["modified_after_creation"] = bool(
            dc is not None and dm is not None and dm > dc)
    except Exception:
        out["modified_after_creation"] = False
    # xmp
    try:
        xmp = getattr(reader, "xmp_metadata", None)
        if xmp is not None:
            xt, xp = "", ""
            for cand in ("creator_tool", "xmp_creator_tool",
                         "creatortool", "creatorTool"):
                try:
                    v = getattr(xmp, cand, None)
                    if v:
                        xt = _s(v() if callable(v) else v)
                        if xt:
                            break
                except Exception:
                    continue
            for cand in ("producer", "pdf_producer", "xmp_producer"):
                try:
                    v = getattr(xmp, cand, None)
                    if v:
                        xp = _s(v() if callable(v) else v)
                        if xp:
                            break
                except Exception:
                    continue
            xmp_d = {}
            if xt:
                xmp_d["creator_tool"] = xt
            if xp:
                xmp_d["producer"] = xp
            if xmp_d:
                out["xmp"] = xmp_d
    except Exception:
        pass
    # fonts + images
    try:
        def get_obj(o):
            try:
                if hasattr(o, "get_object"):
                    return o.get_object()
            except Exception:
                pass
            return o
        limit = max_pages
        try:
            limit = int(limit)
        except Exception:
            limit = FONTS_HARD_LIMIT
        if limit < 0:
            limit = 0
        limit = min(limit, FONTS_HARD_LIMIT)
        fonts, has_img = _collect_fonts_and_images(reader, get_obj, limit)
        out["fonts"] = fonts
        not_emb = [f["base_font"] for f in fonts if not f.get("embedded")]
        emb_n = sum(1 for f in fonts if f.get("embedded"))
        out["summary"] = {
            "fonts_total": int(len(fonts)),
            "fonts_embedded": int(emb_n),
            "fonts_not_embedded": [_s(x) for x in not_emb],
            "has_scanned_images_only": bool(len(fonts) == 0 and has_img),
        }
    except Exception as e:
        errors.append("pdf fonts: %s" % _s(e))
        if "fonts" not in out:
            out["fonts"] = []
        if "summary" not in out:
            out["summary"] = {
                "fonts_total": 0,
                "fonts_embedded": 0,
                "fonts_not_embedded": [],
                "has_scanned_images_only": False,
            }
    return out


def _ooxml_detail(path, errors):
    out = {}
    try:
        with zipfile.ZipFile(path, "r") as z:
            names = set(z.namelist())
            # raw DTD/entity guard
            for target in ("docProps/core.xml", "docProps/app.xml"):
                if target not in names:
                    continue
                try:
                    raw = z.read(target)
                except Exception as e:
                    errors.append("%s read: %s" % (target, _s(e)))
                    continue
                head = raw[:4096]
                if b"<!DOCTYPE" in head.upper() or b"<!ENTITY" in head.upper():
                    errors.append("%s: DTD/entities forbidden" % target)
                    continue
                try:
                    import xml.etree.ElementTree as ET
                    root = ET.fromstring(raw)
                except Exception as e:
                    errors.append("%s xml: %s" % (target, _s(e)))
                    continue
                def tag_local(t):
                    t = _s(t)
                    if "}" in t:
                        return t.split("}", 1)[1]
                    if ":" in t:
                        return t.split(":")[-1]
                    return t
                if target == "docProps/core.xml":
                    want = {"creator": "", "lastModifiedBy": "",
                            "created": "", "modified": "",
                            "title": "", "revision": ""}
                    for child in root.iter():
                        ln = tag_local(child.tag)
                        if ln in want and child.text:
                            want[ln] = _s(child.text).strip()
                    out.update(want)
                else:
                    want = {"Application": "", "AppVersion": "",
                            "Company": "", "Pages": "", "Slides": "",
                            "Words": ""}
                    for child in root.iter():
                        ln = tag_local(child.tag)
                        if ln in want and child.text:
                            want[ln] = _s(child.text).strip()
                    for k in list(want.keys()):
                        out[k] = want[k]
    except zipfile.BadZipFile as e:
        errors.append("zip: %s" % _s(e))
    except Exception as e:
        errors.append("ooxml: %s" % _s(e))
    return out


def _image_detail(path, errors):
    out = {}
    try:
        from PIL import Image
        from PIL.ExifTags import TAGS
    except Exception as e:
        errors.append("pillow import: %s" % _s(e))
        return out
    try:
        with Image.open(path) as im:
            try:
                im.load()
            except Exception:
                pass
            out["format"] = _s(im.format or "")
            out["mode"] = _s(getattr(im, "mode", "") or "")
            try:
                w, h = im.size
                out["width"] = int(w)
                out["height"] = int(h)
            except Exception:
                pass
            try:
                dpi = im.info.get("dpi")
                if dpi is not None:
                    if isinstance(dpi, (tuple, list)) and len(dpi) == 2:
                        out["dpi"] = "%s x %s" % (_s(dpi[0]), _s(dpi[1]))
                    else:
                        out["dpi"] = _s(dpi)
                else:
                    out["dpi"] = ""
            except Exception:
                out["dpi"] = ""
            try:
                ex = im.getexif() if hasattr(im, "getexif") else {}
            except Exception:
                ex = {}
            inv = {v: k for k, v in TAGS.items()}
            for name in ("Make", "Model", "Software", "DateTime"):
                try:
                    tag = inv.get(name)
                    v = ex.get(tag) if tag is not None else None
                    out["exif_" + name] = _s(v) if v is not None else ""
                except Exception:
                    out["exif_" + name] = ""
    except Exception as e:
        errors.append("image: %s" % _s(e))
    return out


# ---------------------------------------------------------------------------
# Handler registry. A handler is fn(path, errors, opts) -> dict (JSON-compatible)
# and must not raise (file_meta isolates it anyway). Add formats with
# @register("kind", ".ext", ...), from localdocs_meta_formats.py or from a
# drop-in plugin folder (see _load_extras).
# ---------------------------------------------------------------------------
_HANDLERS = {}


def register(kind, *exts):
    def deco(fn):
        for e in exts:
            _HANDLERS[str(e).lower() if str(e).startswith(".") else "." + str(e).lower()] = (kind, fn)
        return fn
    return deco


register("pdf", ".pdf")(lambda p, errors, o: _pdf_detail(p, errors, o["max_pages"]))
for _e in (".docx", ".xlsx", ".pptx"):
    register(_e[1:], _e)(lambda p, errors, o: _ooxml_detail(p, errors))
register("image", *sorted(_IMAGE_EXTS))(lambda p, errors, o: _image_detail(p, errors))

_EXTRAS_LOADED = False


def _load_extras():
    """Load built-in extra formats and user plugins once. Never raises."""
    global _EXTRAS_LOADED
    if _EXTRAS_LOADED:
        return
    _EXTRAS_LOADED = True
    try:
        import localdocs_meta_formats  # noqa: F401  (registers its handlers on import)
    except ImportError:
        pass
    except Exception:
        pass
    folder = os.environ.get("LOCALDOCS_META_PLUGINS")
    if folder is None:
        folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "meta_plugins")
    if folder in ("", "0"):
        return  # plugins disabled
    try:
        names = sorted(n for n in os.listdir(folder) if n.endswith(".py") and not n.startswith("_"))
    except OSError:
        return
    import importlib.util
    for n in names:
        try:
            spec = importlib.util.spec_from_file_location("ld_meta_plugin_" + n[:-3], os.path.join(folder, n))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            kind = str(getattr(mod, "KIND", n[:-3]))
            fn = getattr(mod, "extract")
            for e in getattr(mod, "EXTENSIONS"):
                register(kind, e)(lambda p, errors, o, fn=fn: fn(p, o))
        except Exception:
            continue


def supported_extensions():
    _load_extras()
    return sorted(_HANDLERS)




def file_meta(path, detailed=True, max_pages=40):
    """Return JSON-compatible dict with file metadata. Never raises."""
    errors = []
    result = {}
    try:
        p = _s(path)
        name = os.path.basename(p) or p
        result["name"] = _s(name)
        ext = os.path.splitext(name)[1].lower()
        result["ext"] = _s(ext)
        _load_extras()
        entry = _HANDLERS.get(ext)
        kind = entry[0] if entry else "other"
        result["kind"] = kind
        try:
            result["size"] = int(os.path.getsize(p))
        except Exception as e:
            result["size"] = 0
            errors.append("stat: %s" % _s(e))
        result["mtime"] = _s(_iso_mtime(p))
        result["errors"] = []
        if result["size"] > ROOT_2GB:
            errors.append("file larger than 2 GiB, not opened")
            result["errors"] = [_s(e) for e in errors]
            return result
        if not detailed:
            result["errors"] = [_s(e) for e in errors]
            return result
        try:
            limit = int(max_pages)
        except Exception:
            limit = FONTS_HARD_LIMIT
        if entry:
            try:
                extra = entry[1](p, errors, {"max_pages": limit})
                if isinstance(extra, dict):
                    result.update(extra)
            except Exception as e:
                errors.append("%s: %s" % (kind, _s(e)))
        result["errors"] = [_s(e) for e in errors]
        # ensure JSON-compatible: no bytes anywhere
        def clean(o):
            if isinstance(o, bytes):
                return _s(o)
            if isinstance(o, dict):
                return {str(k): clean(v) for k, v in o.items()}
            if isinstance(o, (list, tuple)):
                return [clean(v) for v in o]
            if isinstance(o, bool) or isinstance(o, int) \
                    or isinstance(o, float) or o is None:
                return o
            if isinstance(o, str):
                return o
            try:
                return str(o)
            except Exception:
                return ""
        result = clean(result)
        return result
    except Exception as e:
        try:
            result["errors"] = [_s(x) for x in errors] + [_s(e)]
            return result
        except Exception:
            return {"name": "", "size": 0, "mtime": "",
                    "ext": "", "kind": "other",
                    "errors": [_s(e)]}
