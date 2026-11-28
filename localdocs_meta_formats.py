from localdocs_meta import register, _s
"""Extra LocalDocs metadata handlers (stdlib only). Never raises; errors via errors.append(str)."""

import csv
import datetime
import gzip
import json
import os
import re
import struct
import tarfile
import zipfile

HEAD_MAX = 4 * 1024 * 1024


def _read_head(path, errors, limit=HEAD_MAX):
    try:
        with open(path, "rb") as f:
            return f.read(limit)
    except Exception as e:
        errors.append("read: %s" % _s(e))
        return b""


# ---------------------------------------------------------------- text/code
_TEXT_EXTS = (
    ".txt", ".md", ".csv", ".tsv", ".json", ".xml", ".yaml", ".yml",
    ".ini", ".cfg", ".toml", ".log", ".py", ".js", ".ts", ".java",
    ".c", ".cpp", ".h", ".cs", ".go", ".rs", ".sql", ".sh", ".bat",
    ".ps1", ".html", ".htm", ".css", ".svg",
)

_POLISH_CHARS = set("ąćęłńóśźżĄĆĘŁŃÓŚŹŻ")


def _detect_encoding(data):
    if data.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig", "bom", True
    if data.startswith(b"\xff\xfe\x00\x00") or data.startswith(b"\x00\x00\xfe\xff"):
        return "utf-32", "bom", True
    if data.startswith(b"\xff\xfe"):
        return "utf-16le", "bom", True
    if data.startswith(b"\xfe\xff"):
        return "utf-16be", "bom", True
    if data.startswith(b"\x00\x00\xfe\xff"):
        return "utf-32be", "bom", True
    try:
        data.decode("utf-8 strict" if False else "utf-8")
        return "utf-8", "valid-utf8", False
    except Exception:
        pass
    # heuristic: cp1251 if cyrillic-like
    try:
        t1251 = data.decode("cp1251")
    except Exception:
        t1251 = ""
    cyr = sum(1 for ch in t1251 if "\u0400" <= ch <= "\u045f")
    if cyr >= 2:
        return "cp1251", "heuristic", False
    try:
        t1250 = data.decode("cp1250")
    except Exception:
        t1250 = ""
    pol = sum(1 for ch in t1250 if ch in _POLISH_CHARS)
    if pol >= 1:
        return "cp1250", "heuristic", False
    return "latin-1", "heuristic", False


def _decode_for_text(data, encoding):
    try:
        if encoding == "utf-32":
            # auto-detect LE/BE by BOM
            if data.startswith(b"\x00\x00\xfe\xff"):
                return data.decode("utf-32be", errors="replace")
            return data.decode("utf-32", errors="replace")
        return data.decode(encoding, errors="replace")
    except Exception:
        try:
            return data.decode("latin-1", errors="replace")
        except Exception:
            return ""


def _text_detail(path, errors, ext):
    out = {}
    try:
        data = _read_head(path, errors)
        enc, conf, bom = _detect_encoding(data)
        out["encoding"] = _s(enc)
        out["encoding_confidence"] = _s(conf)
        out["has_bom"] = bool(bom)
        text = _decode_for_text(data, enc)
        # strip BOM char for analysis
        if text.startswith("\ufeff"):
            text = text[1:]
        n_crlf = text.count("\r\n")
        tmp = text.replace("\r\n", "")
        n_lf = tmp.count("\n")
        tmp2 = tmp.replace("\n", "")
        n_cr = tmp2.count("\r")
        kinds = set()
        if n_crlf:
            kinds.add("CRLF")
        if n_lf:
            kinds.add("LF")
        if n_cr:
            kinds.add("CR")
        if len(kinds) == 0:
            out["line_endings"] = ""
        elif len(kinds) == 1:
            out["line_endings"] = next(iter(kinds))
        else:
            out["line_endings"] = "mixed"
        try:
            out["lines"] = int(len(text.splitlines()))
        except Exception:
            out["lines"] = 0
        try:
            first = text.splitlines()[0] if text.splitlines() else ""
            out["shebang"] = _s(first) if first.startswith("#!") else ""
        except Exception:
            out["shebang"] = ""
        # csv/tsv
        if ext in (".csv", ".tsv"):
            try:
                first_line = text.splitlines()[0] if text.splitlines() else ""
                if ext == ".tsv":
                    delim = "\t"
                else:
                    cands = [",", ";", "\t", "|"]
                    delim = max(cands, key=lambda d: first_line.count(d))
                    if first_line.count(delim) == 0:
                        delim = ","
                out["delimiter"] = _s(delim)
                try:
                    reader = csv.reader([first_line], delimiter=delim)
                    cols = next(reader, [])
                    out["columns"] = int(len(cols))
                except Exception:
                    out["columns"] = int(len(first_line.split(delim)) if first_line else 0)
            except Exception as e:
                errors.append("csv: %s" % _s(e))
        # html
        if ext in (".html", ".htm"):
            try:
                m = re.search(r'<meta[^>]*charset\s*=\s*["\']?([^"\'\s;/>]+)', text, re.I)
                out["meta_charset"] = _s(m.group(1).strip().strip("\"'")) if m else ""
                m2 = re.search(r"<title[^>]*>(.*?)</title>", text, re.I | re.S)
                out["title"] = _s(re.sub(r"\s+", " ", m2.group(1)).strip()) if m2 else ""
                m3 = re.search(r'<meta[^>]*name\s*=\s*["\']generator["\'][^>]*>', text, re.I)
                gen = ""
                if m3:
                    tag = m3.group(0)
                    mg = re.search(r'content\s*=\s*["\']([^"\']*)["\']', tag, re.I)
                    if mg:
                        gen = mg.group(1)
                    else:
                        mg2 = re.search(r"content\s*=\s*([^\s>]+)", tag, re.I)
                        if mg2:
                            gen = mg2.group(1)
                out["generator"] = _s(gen)
            except Exception as e:
                errors.append("html: %s" % _s(e))
        # xml/svg
        if ext in (".xml", ".svg"):
            try:
                m = re.search(r'<\?xml[^>]*encoding\s*=\s*["\']([^"\']+)["\']', text, re.I)
                out["declared_encoding"] = _s(m.group(1)) if m else ""
                body = re.sub(r"<\?.*?\?>", "", text, count=1, flags=re.S)
                body = re.sub(r"<!--.*?-->", "", body, flags=re.S)
                m2 = re.search(r"<\s*([A-Za-z_][\w\-.:]*)", body)
                out["root_tag"] = _s(m2.group(1)) if m2 else ""
            except Exception as e:
                errors.append("xml: %s" % _s(e))
        # json
        if ext == ".json":
            try:
                json.loads(text if text.strip() else "null" if False else text)
                out["json_valid"] = True
            except Exception:
                # empty file is invalid
                try:
                    json.loads(text)
                    out["json_valid"] = True
                except Exception:
                    out["json_valid"] = False
    except Exception as e:
        errors.append("text: %s" % _s(e))
    return out


def _make_text_handler(ext):
    def _h(path, errors, opts):
        return _text_detail(path, errors, ext)
    return _h


for _e in _TEXT_EXTS:
    register("text", _e)(_make_text_handler(_e))


# ---------------------------------------------------------------- ODF
@register("odf", ".odt", ".ods", ".odp", ".odg")
def _odf_detail(path, errors, opts):
    out = {}
    try:
        try:
            z = zipfile.ZipFile(path, "r")
        except Exception as e:
            errors.append("odf zip: %s" % _s(e))
            return out
        with z:
            try:
                names = set(z.namelist())
            except Exception as e:
                errors.append("odf list: %s" % _s(e))
                return out
            if "meta.xml" not in names:
                errors.append("odf: meta.xml missing")
                return out
            try:
                raw = z.read("meta.xml")
            except Exception as e:
                errors.append("odf meta.xml read: %s" % _s(e))
                return out
            if len(raw) > HEAD_MAX:
                raw = raw[:HEAD_MAX]
            try:
                import xml.etree.ElementTree as ET
                root = ET.fromstring(raw)
            except Exception as e:
                errors.append("odf meta.xml xml: %s" % _s(e))
                return out

            def local(t):
                t = _s(t)
                if "}" in t:
                    return t.split("}", 1)[1]
                if ":" in t:
                    return t.split(":")[-1]
                return t

            vals = {}
            stat = {}
            for el in root.iter():
                ln = local(el.tag)
                tx = _s(el.text).strip() if el.text else ""
                if ln == "generator" and "generator" not in vals:
                    vals["generator"] = tx
                elif ln == "initial-creator" and "initial-creator" not in vals:
                    vals["initial-creator"] = tx
                elif ln == "creator" and "creator" not in vals:
                    vals["creator"] = tx
                elif ln == "creation-date" and "creation-date" not in vals:
                    vals["creation-date"] = tx
                elif ln == "date" and "date" not in vals:
                    vals["date"] = tx
                elif ln == "title" and "title" not in vals:
                    vals["title"] = tx
                elif ln == "editing-cycles" and "editing-cycles" not in vals:
                    vals["editing-cycles"] = tx
                elif ln == "document-statistic":
                    try:
                        for k, v in el.attrib.items():
                            stat[local(k)] = _s(v)
                    except Exception:
                        pass
            for k in ("generator", "initial-creator", "creator", "creation-date",
                      "date", "title", "editing-cycles"):
                out[k.replace("-", "_")] = _s(vals.get(k, ""))
            out["document_statistic"] = {str(k): _s(v) for k, v in stat.items()}
    except Exception as e:
        errors.append("odf: %s" % _s(e))
    return out


# ---------------------------------------------------------------- EPUB
@register("epub", ".epub")
def _epub_detail(path, errors, opts):
    out = {}
    try:
        try:
            z = zipfile.ZipFile(path, "r")
        except Exception as e:
            errors.append("epub zip: %s" % _s(e))
            return out
        with z:
            try:
                names = set(z.namelist())
            except Exception as e:
                errors.append("epub list: %s" % _s(e))
                return out
            opf_name = ""
            opf_raw = b""
            # locate OPF via container.xml
            if "META-INF/container.xml" in names:
                try:
                    craw = z.read("META-INF/container.xml")
                    import xml.etree.ElementTree as ET
                    croot = ET.fromstring(craw[:HEAD_MAX])
                    for el in croot.iter():
                        t = _s(el.tag)
                        ln = t.split("}", 1)[1] if "}" in t else t
                        if ln == "rootfile":
                            for ak, av in el.attrib.items():
                                al = ak.split("}", 1)[1] if "}" in ak else ak
                                if al == "full-path":
                                    opf_name = _s(av)
                                    break
                            if opf_name:
                                break
                except Exception as e:
                    errors.append("epub container: %s" % _s(e))
            if not opf_name:
                for n in z.namelist():
                    if n.lower().endswith(".opf"):
                        opf_name = n
                        break
            if not opf_name:
                errors.append("epub: OPF missing")
                return out
            try:
                opf_raw = z.read(opf_name)
            except Exception as e:
                errors.append("epub opf read: %s" % _s(e))
                return out
            try:
                import xml.etree.ElementTree as ET
                root = ET.fromstring(opf_raw[:HEAD_MAX])
            except Exception as e:
                errors.append("epub opf xml: %s" % _s(e))
                return out
            out["epub_version"] = _s(root.attrib.get("version", ""))
            if not out["epub_version"]:
                for ak, av in root.attrib.items():
                    if ak.lower().endswith("version"):
                        out["epub_version"] = _s(av)
                        break

            def local(t):
                t = _s(t)
                return t.split("}", 1)[1] if "}" in t else (t.split(":")[-1])

            fields = {"title": "", "creator": "", "publisher": "",
                      "language": "", "date": "", "identifier": ""}
            for el in root.iter():
                ln = local(el.tag).lower()
                if ln in fields and not fields[ln]:
                    fields[ln] = _s(el.text).strip() if el.text else ""
            out["title"] = _s(fields["title"])
            out["creator"] = _s(fields["creator"])
            out["publisher"] = _s(fields["publisher"])
            out["language"] = _s(fields["language"])
            out["date"] = _s(fields["date"])
            out["identifier"] = _s(fields["identifier"])
    except Exception as e:
        errors.append("epub: %s" % _s(e))
    return out


# ---------------------------------------------------------------- RTF
@register("rtf", ".rtf")
def _rtf_detail(path, errors, opts):
    out = {"title": "", "author": "", "company": "", "creatim": "",
           "revtim": "", "codepage": "", "generator": ""}
    try:
        data = _read_head(path, errors)
        try:
            text = data.decode("latin-1", errors="replace")
        except Exception:
            text = ""
        head = text[:HEAD_MAX]
        m = re.search(r"\\ansicpg(\d+)", head)
        if m:
            out["codepage"] = _s("ansicpg" + m.group(1))
        # info group: search whole head (info fields live inside {\info ...})
        scope = head[:65536]
        for key in ("title", "author", "company"):
            mm = re.search(r"\\" + key + r"\s+([^;\\}]+)", scope)
            if mm:
                out[key] = _s(mm.group(1).strip())
        for key in ("creatim", "revtim"):
            mm = re.search(r"\\" + key + r"\s*\\yr(\d+)?", scope)
            if mm:
                # collect full tim group: \yrN\moN\dyN\hrN\minN
                mg = re.search(r"\\" + key + r"((?:\\[a-z]+\d*)+)", scope)
                out[key] = _s(mg.group(1) if mg else mm.group(0))
            else:
                mm2 = re.search(r"\\" + key + r"([^}]*?)(?=\\[a-z]+|\})", scope)
                if mm2 and mm2.group(1).strip():
                    out[key] = _s(mm2.group(1).strip())
        mg = re.search(r"\\generator\s+([^;\\}]+)", head)
        if mg:
            out["generator"] = _s(mg.group(1).strip())
        else:
            mg2 = re.search(r"\{\\\*\s*\\generator\s*([^}]*)\}", head)
            if mg2:
                out["generator"] = _s(mg2.group(1).strip())
    except Exception as e:
        errors.append("rtf: %s" % _s(e))
    return out


# ---------------------------------------------------------------- EML
@register("email", ".eml")
def _eml_detail(path, errors, opts):
    out = {}
    try:
        data = _read_head(path, errors)
        try:
            from email import message_from_bytes
            from email.utils import getaddresses
            msg = message_from_bytes(data)
        except Exception as e:
            errors.append("email parse: %s" % _s(e))
            return out
        try:
            from email.utils import getaddresses
            raw_from = _s(msg.get("From", ""))
            addrs = getaddresses([raw_from]) if raw_from else []
            dom = ""
            if addrs:
                em = _s(addrs[0][1])
                dom = em.split("@")[-1] if "@" in em else ""
            out["from_domain"] = _s(dom)
        except Exception:
            out["from_domain"] = ""
        try:
            from email.utils import getaddresses
            raw_to = msg.get_all("To", [])
            all_addr = []
            for r in raw_to:
                all_addr.extend(getaddresses([_s(r)]))
            emails = [_s(a[1]) for a in all_addr if _s(a[1])]
            doms = sorted({e.split("@")[-1] for e in emails if "@" in e})
            out["to_count"] = int(len(emails))
            out["to_domains"] = [_s(d) for d in doms]
        except Exception:
            out["to_count"] = 0
            out["to_domains"] = []
        for hdr, key in (("Subject", "subject"), ("Date", "date"),
                         ("Message-ID", "message_id"), ("X-Mailer", "x_mailer")):
            try:
                out[key] = _s(msg.get(hdr, ""))
            except Exception:
                out[key] = ""
        try:
            cs = _s(msg.get_content_charset() or "")
            if not cs:
                ct = _s(msg.get("Content-Type", ""))
                m = re.search(r'charset\s*=\s*["\']?([^"\'\s;]+)', ct, re.I)
                if m:
                    cs = m.group(1)
            out["charset"] = _s(cs)
            out["content_type"] = _s(msg.get_content_type() or "")
        except Exception:
            out["charset"] = ""
            out["content_type"] = ""
        try:
            n_att = 0
            if msg.is_multipart():
                for part in msg.walk():
                    try:
                        if part.is_multipart():
                            continue
                        disp = _s(part.get("Content-Disposition", ""))
                        if "attachment" in disp.lower():
                            n_att += 1
                        elif part.get_filename():
                            n_att += 1
                    except Exception:
                        continue
            out["attachments"] = int(n_att)
        except Exception:
            out["attachments"] = 0
    except Exception as e:
        errors.append("email: %s" % _s(e))
    return out


# ---------------------------------------------------------------- archives: zip family
@register("archive", ".zip", ".jar", ".apk", ".whl", ".xpi")
def _zip_detail(path, errors, opts):
    out = {}
    try:
        try:
            z = zipfile.ZipFile(path, "r")
        except Exception as e:
            errors.append("zip open: %s" % _s(e))
            return out
        with z:
            try:
                infos = z.infolist()
            except Exception as e:
                errors.append("zip list: %s" % _s(e))
                return out
            try:
                out["entries"] = int(len(infos))
                out["uncompressed_total"] = int(sum(int(i.file_size) for i in infos))
                out["compressed_total"] = int(sum(int(i.compress_size) for i in infos))
                try:
                    out["comment"] = _s(z.comment.decode("utf-8", "replace") if z.comment else "")
                except Exception:
                    out["comment"] = ""
                enc = sum(1 for i in infos if (int(i.flag_bits) & 0x1))
                out["encrypted_entries"] = int(enc)
                meths = sorted({("deflated" if int(i.compress_type) == 8 else
                                 "stored" if int(i.compress_type) == 0 else
                                 "method-%d" % int(i.compress_type)) for i in infos})
                out["methods"] = [_s(m) for m in meths]
            except Exception as e:
                errors.append("zip stat: %s" % _s(e))
    except Exception as e:
        errors.append("archive: %s" % _s(e))
    return out


def _parse_gzip_header(data, errors):
    info = {"original_name": "", "mtime": "", "mtime_unix": 0}
    try:
        if len(data) < 10 or data[0] != 0x1F or data[1] != 0x8B:
            return info
        flg = data[3]
        mtime = struct.unpack("<I", data[4:8])[0]
        info["mtime_unix"] = int(mtime)
        try:
            if mtime:
                dt = datetime.datetime.fromtimestamp(mtime, tz=datetime.timezone.utc)
                info["mtime"] = _s(dt.isoformat())
        except Exception:
            pass
        pos = 10
        if flg & 0x04:  # FEXTRA
            if pos + 2 <= len(data):
                xlen = struct.unpack("<H", data[pos:pos + 2])[0]
                pos += 2 + xlen
        if flg & 0x08:  # FNAME
            end = data.find(b"\x00", pos)
            if end != -1:
                try:
                    info["original_name"] = _s(data[pos:end].decode("utf-8", "replace"))
                except Exception:
                    info["original_name"] = ""
                pos = end + 1
        return info
    except Exception as e:
        errors.append("gzip header: %s" % _s(e))
        return info


def _count_tar_entries(path, mode, errors, limit=100000):
    try:
        n = 0
        try:
            with tarfile.open(path, mode) as t:
                for _m in t:
                    n += 1
                    if n >= limit:
                        break
        except Exception as e:
            errors.append("tar list: %s" % _s(e))
            return 0
        return int(n)
    except Exception as e:
        errors.append("tar: %s" % _s(e))
        return 0


@register("archive", ".tar")
def _tar_detail(path, errors, opts):
    out = {"format": "tar"}
    try:
        out["entries"] = int(_count_tar_entries(path, "r:", errors))
    except Exception as e:
        errors.append("tar: %s" % _s(e))
    return out


@register("archive", ".tgz")
def _tgz_detail(path, errors, opts):
    out = {"format": "tgz"}
    try:
        data = _read_head(path, errors, limit=1024 * 64)
        g = _parse_gzip_header(data, errors)
        out["original_name"] = _s(g.get("original_name", ""))
        out["mtime"] = _s(g.get("mtime", ""))
        out["mtime_unix"] = int(g.get("mtime_unix", 0))
        out["entries"] = int(_count_tar_entries(path, "r:gz", errors))
    except Exception as e:
        errors.append("tgz: %s" % _s(e))
    return out


@register("archive", ".gz")
def _gz_detail(path, errors, opts):
    out = {"format": "gz"}
    try:
        data = _read_head(path, errors, limit=1024 * 64)
        g = _parse_gzip_header(data, errors)
        out["original_name"] = _s(g.get("original_name", ""))
        out["mtime"] = _s(g.get("mtime", ""))
        out["mtime_unix"] = int(g.get("mtime_unix", 0))
    except Exception as e:
        errors.append("gz: %s" % _s(e))
    return out


@register("archive", ".bz2")
def _bz2_detail(path, errors, opts):
    out = {"format": "bz2"}
    try:
        data = _read_head(path, errors, limit=16)
        out["signature"] = _s("BZh" if data.startswith(b"BZh") else "")
        # maybe tar inside
        try:
            n = _count_tar_entries(path, "r:bz2", errors, limit=100000)
            # only report entries if it really is a tar (no error appended for valid tar)
            # heuristic: if n>0 keep, else 0
            out["entries"] = int(n)
        except Exception:
            out["entries"] = 0
        # drop the tar-list error when it's just a single stream
        try:
            errors[:] = [e for e in errors if not _s(e).startswith("tar list")]
        except Exception:
            pass
    except Exception as e:
        errors.append("bz2: %s" % _s(e))
    return out


@register("archive", ".xz")
def _xz_detail(path, errors, opts):
    out = {"format": "xz"}
    try:
        data = _read_head(path, errors, limit=16)
        out["signature"] = _s("XZ" if data.startswith(b"\xfd7zXZ\x00") else "")
        try:
            n = _count_tar_entries(path, "r:xz", errors, limit=100000)
            out["entries"] = int(n)
        except Exception:
            out["entries"] = 0
        try:
            errors[:] = [e for e in errors if not _s(e).startswith("tar list")]
        except Exception:
            pass
    except Exception as e:
        errors.append("xz: %s" % _s(e))
    return out


@register("archive", ".7z", ".rar")
def _seven_rar_detail(path, errors, opts):
    out = {}
    try:
        data = _read_head(path, errors, limit=32)
        if data.startswith(b"7z\xbc\xaf'\x1c"):
            out["format"] = "7z"
            out["signature"] = "7z"
            try:
                major, minor = data[6], data[7]
                out["version"] = _s("%d.%02d" % (major, minor))
            except Exception:
                out["version"] = ""
        elif data.startswith(b"Rar!\x1a\x07\x01\x00"):
            out["format"] = "rar"
            out["signature"] = "RAR5"
            out["version"] = "5"
        elif data.startswith(b"Rar!\x1a\x07\x00"):
            out["format"] = "rar"
            out["signature"] = "RAR4"
            out["version"] = "4"
        elif data.startswith(b"Rar!"):
            out["format"] = "rar"
            out["signature"] = "RAR"
            out["version"] = ""
        else:
            out["format"] = ""
            out["signature"] = ""
            out["version"] = ""
    except Exception as e:
        errors.append("7z/rar: %s" % _s(e))
    return out


# ---------------------------------------------------------------- sqlite
@register("sqlite", ".sqlite", ".db", ".sqlite3")
def _sqlite_detail(path, errors, opts):
    out = {}
    try:
        data = _read_head(path, errors, limit=128)
        if len(data) < 100:
            errors.append("sqlite: header too short")
            return out
        if not data.startswith(b"SQLite format 3\x00"):
            errors.append("sqlite: bad magic")
            return out
        try:
            ps = struct.unpack(">H", data[16:18])[0]
            if ps == 1:
                ps = 65536
            out["page_size"] = int(ps)
            te = struct.unpack(">I", data[56:60])[0]
            out["text_encoding"] = int(te)
            out["text_encoding_name"] = _s({1: "utf-8", 2: "utf-16le", 3: "utf-16be"}.get(te, "unknown-%d" % te))
            out["user_version"] = int(struct.unpack(">I", data[60:64])[0])
            out["application_id"] = int(struct.unpack(">I", data[68:72])[0])
            out["sqlite_version_number"] = int(struct.unpack(">I", data[96:100])[0])
        except Exception as e:
            errors.append("sqlite header: %s" % _s(e))
    except Exception as e:
        errors.append("sqlite: %s" % _s(e))
    return out


# ---------------------------------------------------------------- audio/video
@register("audio", ".wav")
def _wav_detail(path, errors, opts):
    out = {}
    try:
        data = _read_head(path, errors, limit=1024 * 1024)
        if len(data) < 44 or not data.startswith(b"RIFF") or data[8:12] != b"WAVE":
            errors.append("wav: bad header")
            return out
        try:
            pos = 12
            channels = rate = bits = 0
            nframes = 0
            found_fmt = False
            data_len = 0
            while pos + 8 <= len(data):
                tag = data[pos:pos + 4]
                size = struct.unpack("<I", data[pos + 4:pos + 8])[0]
                body = pos + 8
                if tag == b"fmt " and size >= 16 and body + 16 <= len(data):
                    _fmt, channels, rate = struct.unpack("<HHI", data[body:body + 8])
                    _br, _ba, bits = struct.unpack("<IHH", data[body + 8:body + 16])
                    found_fmt = True
                elif tag == b"data":
                    data_len = size
                pos = body + size + (size % 2)
                if found_fmt and data_len:
                    break
            if not found_fmt:
                errors.append("wav: fmt chunk missing")
                return out
            out["channels"] = int(channels)
            out["rate"] = int(rate)
            out["bits"] = int(bits)
            try:
                dur = (data_len / (channels * rate * (bits // 8))) if channels and rate and bits else 0
                out["duration"] = "%.3f" % dur
            except Exception:
                out["duration"] = ""
        except Exception as e:
            errors.append("wav parse: %s" % _s(e))
    except Exception as e:
        errors.append("wav: %s" % _s(e))
    return out


@register("audio", ".mp3")
def _mp3_detail(path, errors, opts):
    out = {"title": "", "artist": "", "album": "", "year": ""}
    try:
        data = _read_head(path, errors, limit=1024 * 512)
        if len(data) < 10 or not data.startswith(b"ID3"):
            return out
        try:
            size = ((data[6] & 0x7F) << 21) | ((data[7] & 0x7F) << 14) | \
                ((data[8] & 0x7F) << 7) | (data[9] & 0x7F)
            pos = 10
            end = min(10 + size, len(data))
            # skip extended header
            if len(data) >= 10 and (data[5] & 0x40) and pos + 4 <= end:
                # v2.3 ext size is plain int; v2.4 syncsafe — approximate
                try:
                    xsize = struct.unpack(">I", data[pos:pos + 4])[0]
                    pos += 4 + max(0, min(xsize, end - pos - 4))
                except Exception:
                    pass

            def _frame_text(body):
                if not body:
                    return ""
                enc = body[0]
                raw = body[1:]
                try:
                    if enc == 0:
                        return raw.decode("latin-1", "replace").rstrip("\x00").strip()
                    if enc == 1:
                        return raw.decode("utf-16", "replace").rstrip("\x00").strip()
                    if enc == 2:
                        return raw.decode("utf-16be", "replace").rstrip("\x00").strip()
                    return raw.decode("utf-8", "replace").rstrip("\x00").strip()
                except Exception:
                    return ""

            while pos + 10 <= end:
                fid = data[pos:pos + 4]
                try:
                    fsize = struct.unpack(">I", data[pos + 4:pos + 8])[0]
                except Exception:
                    break
                if fid == b"\x00\x00\x00\x00":
                    break
                fbody = data[pos + 10:pos + 10 + fsize]
                try:
                    name = fid.decode("ascii", "replace")
                except Exception:
                    name = ""
                if name == "TIT2" and not out["title"]:
                    out["title"] = _s(_frame_text(fbody))
                elif name == "TPE1" and not out["artist"]:
                    out["artist"] = _s(_frame_text(fbody))
                elif name == "TALB" and not out["album"]:
                    out["album"] = _s(_frame_text(fbody))
                elif name in ("TYER", "TDRC") and not out["year"]:
                    out["year"] = _s(_frame_text(fbody)[:4])
                pos += 10 + fsize
        except Exception as e:
            errors.append("mp3 id3: %s" % _s(e))
    except Exception as e:
        errors.append("mp3: %s" % _s(e))
    return out


@register("audio", ".flac")
def _flac_detail(path, errors, opts):
    out = {}
    try:
        data = _read_head(path, errors, limit=1024 * 64)
        if len(data) < 42 or not data.startswith(b"fLaC"):
            errors.append("flac: bad header")
            return out
        try:
            pos = 4
            info = None
            while pos + 4 <= len(data):
                b0 = data[pos]
                last = bool(b0 & 0x80)
                typ = b0 & 0x7F
                blen = int.from_bytes(data[pos + 1:pos + 4], "big")
                body = data[pos + 4:pos + 4 + blen]
                if typ == 0 and len(body) >= 34:
                    info = body[:34]
                    break
                pos += 4 + blen
                if last or pos > 1024 * 64 - 34:
                    break
            if info is None:
                errors.append("flac: STREAMINFO missing")
                return out
            tail = info[10:18]
            val = int.from_bytes(tail, "big")
            rate = (val >> 44) & 0xFFFFF
            ch = ((val >> 41) & 0x7) + 1
            bps = ((val >> 36) & 0x1F) + 1
            total = val & 0xFFFFFFFFF
            out["rate"] = int(rate)
            out["channels"] = int(ch)
            out["bits"] = int(bps)
            out["total_samples"] = int(total)
            try:
                out["duration"] = "%.3f" % (total / rate) if rate else ""
            except Exception:
                out["duration"] = ""
        except Exception as e:
            errors.append("flac parse: %s" % _s(e))
    except Exception as e:
        errors.append("flac: %s" % _s(e))
    return out


@register("audio", ".ogg")
def _ogg_detail(path, errors, opts):
    out = {}
    try:
        data = _read_head(path, errors, limit=64)
        if len(data) < 4 or not data.startswith(b"OggS"):
            errors.append("ogg: bad signature")
            out["signature"] = ""
            return out
        out["signature"] = "OggS"
        try:
            out["version"] = int(data[4]) if len(data) > 4 else 0
        except Exception:
            out["version"] = 0
    except Exception as e:
        errors.append("ogg: %s" % _s(e))
    return out


def _mp4_atoms(data, errors):
    atoms = []
    ftyp = {"major_brand": "", "minor": "", "compatible": []}
    mvhd = {"duration": "", "timescale": 0}
    try:
        pos = 0
        n = len(data)
        while pos + 8 <= n and len(atoms) < 64:
            size = struct.unpack(">I", data[pos:pos + 4])[0]
            typ = data[pos + 4:pos + 8]
            try:
                tname = typ.decode("ascii")
            except Exception:
                break
            hdr = 8
            if size == 1 and pos + 16 <= n:
                size = struct.unpack(">Q", data[pos + 8:pos + 16])[0]
                hdr = 16
            if size == 0:
                size = n - pos
            if size < hdr or pos + size > n + 1:
                # last atom may extend beyond 4MB head; record type and stop
                atoms.append(_s(tname))
                break
            atoms.append(_s(tname))
            body = data[pos + hdr:pos + size]
            if tname == "ftyp" and not ftyp["major_brand"] and len(body) >= 8:
                try:
                    ftyp["major_brand"] = _s(body[0:4].decode("ascii", "replace"))
                    ftyp["minor"] = str(int.from_bytes(body[4:8], "big"))
                    comp = []
                    for i in range(8, len(body), 4):
                        if i + 4 <= len(body):
                            comp.append(_s(body[i:i + 4].decode("ascii", "replace")))
                    ftyp["compatible"] = comp
                except Exception:
                    pass
            if tname == "moov":
                # search mvhd inside moov body
                try:
                    q = 0
                    while q + 8 <= len(body):
                        ss = struct.unpack(">I", body[q:q + 4])[0]
                        st = body[q + 4:q + 8]
                        hh = 8
                        if ss == 1 and q + 16 <= len(body):
                            ss = struct.unpack(">Q", body[q + 8:q + 16])[0]
                            hh = 16
                        if ss < hh or q + ss > len(body):
                            break
                        try:
                            sname = st.decode("ascii")
                        except Exception:
                            break
                        if sname == "mvhd" and mvhd["duration"] == "":
                            pay = body[q + hh:q + ss]
                            if len(pay) >= 20:
                                ver = pay[0]
                                if ver == 1 and len(pay) >= 28:
                                    ts = struct.unpack(">I", pay[20:24])[0]
                                    du = struct.unpack(">Q", pay[24:32])[0]
                                    mvhd["timescale"] = int(ts)
                                    mvhd["duration"] = "%.3f" % (du / ts) if ts else ""
                                else:
                                    ts = struct.unpack(">I", pay[12:16])[0]
                                    du = struct.unpack(">I", pay[16:20])[0]
                                    mvhd["timescale"] = int(ts)
                                    mvhd["duration"] = "%.3f" % (du / ts) if ts else ""
                        q += ss
                except Exception:
                    pass
            pos += size
    except Exception as e:
        errors.append("mp4 atoms: %s" % _s(e))
    return atoms, ftyp, mvhd


@register("video", ".mp4", ".m4a", ".mov")
def _mp4_detail(path, errors, opts):
    out = {}
    try:
        data = _read_head(path, errors)
        atoms, ftyp, mvhd = _mp4_atoms(data, errors)
        out["atoms"] = [_s(a) for a in atoms]
        out["major_brand"] = _s(ftyp.get("major_brand", ""))
        if ftyp.get("compatible"):
            out["compatible_brands"] = [_s(c) for c in ftyp["compatible"]]
        if mvhd.get("duration"):
            out["duration"] = _s(mvhd["duration"])
            out["timescale"] = int(mvhd.get("timescale", 0))
    except Exception as e:
        errors.append("mp4: %s" % _s(e))
    return out


# ---------------------------------------------------------------- PE
_PE_MACHINE_NAMES = {
    0x014c: "i386", 0x014d: "i486", 0x014e: "i586",
    0x0160: "R3000BE", 0x0162: "R3000LE", 0x0166: "R4000LE",
    0x0168: "R10000", 0x0184: "Alpha_AXP", 0x01a2: "SH3",
    0x01a4: "SH3DSP", 0x01a6: "SH4", 0x01a8: "SH5",
    0x01c0: "ARM", 0x01c2: "Thumb", 0x01c4: "ARMNT",
    0x01f0: "PowerPC", 0x01f1: "PowerPCFP", 0x01f2: "PowerPC64",
    0x0200: "IA64", 0x0266: "MIPS16", 0x0284: "Alpha_AXP64",
    0x0284 + 0: "AXP64", 0x0366: "MIPSFPU", 0x0466: "MIPSFPU16",
    0x0520: "Tricore", 0x0cef: "CEF", 0x0ebc: "EFIByteCode",
    0x8664: "x86-64", 0xaa64: "ARM64", 0xebc: "EBC",
}


@register("pe", ".exe", ".dll", ".sys")
def _pe_detail(path, errors, opts):
    out = {}
    try:
        data = _read_head(path, errors, limit=1024 * 256)
        if len(data) < 64 or not data.startswith(b"MZ"):
            errors.append("pe: bad MZ")
            return out
        try:
            e_lfanew = struct.unpack("<I", data[0x3C:0x40])[0]
        except Exception:
            errors.append("pe: bad e_lfanew")
            return out
        if e_lfanew + 6 > len(data) or data[e_lfanew:e_lfanew + 4] != b"PE\x00\x00":
            errors.append("pe: bad PE signature")
            return out
        try:
            machine, nsec, ts = struct.unpack("<HHI", data[e_lfanew + 4:e_lfanew + 12])
            out["machine"] = int(machine)
            out["machine_name"] = _s(_PE_MACHINE_NAMES.get(machine, "unknown-0x%04x" % machine))
            out["sections"] = int(nsec)
            try:
                dt = datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc)
                out["timestamp"] = _s(dt.isoformat())
            except Exception:
                out["timestamp"] = ""
            out["timestamp_unix"] = int(ts)
            # optional header
            opt_off = e_lfanew + 24
            if opt_off + 70 <= len(data):
                magic = struct.unpack("<H", data[opt_off:opt_off + 2])[0]
                out["is_64"] = bool(magic == 0x20B)
                # subsystem at +68 (both 32/64)
                try:
                    subsys = struct.unpack("<H", data[opt_off + 68:opt_off + 70])[0]
                    out["subsystem"] = int(subsys)
                except Exception:
                    out["subsystem"] = 0
            else:
                out["is_64"] = False
                out["subsystem"] = 0
            # characteristics at COFF+18
            try:
                chars = struct.unpack("<H", data[e_lfanew + 22:e_lfanew + 24])[0]
                out["is_dll"] = bool(chars & 0x2000)
            except Exception:
                out["is_dll"] = False
        except Exception as e:
            errors.append("pe parse: %s" % _s(e))
    except Exception as e:
        errors.append("pe: %s" % _s(e))
    return out
