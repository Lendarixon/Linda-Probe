"""Filesystem browsing helpers for Linda Probe (localdocs).

list_dir(root, rel='') -> {"path", "rel", "parent", "entries"}
breadcrumbs(root, rel) -> [{"name", "rel"}]
"""
import os
import re

try:
    import ctypes
    import ctypes.wintypes
    _HAS_CTYPES = True
except Exception:
    _HAS_CTYPES = False

_LIMIT_DEFAULT = 2000


def _norm_rel(rel):
    if rel is None:
        rel = ""
    rel = str(rel).strip()
    if rel in ("", ".", "./"):
        return ""
    # accept both separators, normalize
    rel = rel.replace("\\", "/")
    # absolute path -> forbidden (POSIX or drive-letter)
    if rel.startswith("/") or os.path.isabs(rel):
        raise PermissionError("absolute path not allowed: %r" % (rel,))
    if re.match(r"^[A-Za-z]:", rel):
        raise PermissionError("absolute path not allowed: %r" % (rel,))
    norm = os.path.normpath(rel.replace("/", os.sep))
    if norm in (".", ""):
        return ""
    if os.path.isabs(norm):
        raise PermissionError("absolute path not allowed: %r" % (rel,))
    return norm


def _to_rel_str(norm):
    if norm in ("", "."):
        return ""
    return norm.replace(os.sep, "/")


def _resolve(root_real, rel_norm):
    if not rel_norm:
        return root_real
    full = os.path.realpath(os.path.join(root_real, rel_norm))
    return full


def _check_inside(root_real, full):
    try:
        common = os.path.commonpath([root_real, full])
    except ValueError:
        raise PermissionError("path escapes project root")
    if common != root_real:
        raise PermissionError("path escapes project root")


def _is_hidden(full_path, name):
    if name.startswith("."):
        return True
    if os.name == "nt" and _HAS_CTYPES:
        try:
            attrs = ctypes.windll.kernel32.GetFileAttributesW(full_path)
            if attrs != -1:
                # FILE_ATTRIBUTE_HIDDEN = 0x2, FILE_ATTRIBUTE_SYSTEM = 0x4
                if attrs & 0x2 or attrs & 0x4:
                    return True
        except Exception:
            pass
    return False


def _nat_key(name):
    parts = re.split(r"(\d+)", name.lower())
    key = []
    for p in parts:
        if p.isdigit():
            key.append((1, int(p)))
        else:
            key.append((0, p))
    return key


def _sort_key(entry):
    # entry: (is_dir, name)
    is_dir, name = entry[0], entry[1]
    return (0 if is_dir else 1, _nat_key(name))


def _check_root(root):
    if not os.path.isabs(root):
        raise ValueError("root must be an absolute path: %r" % (root,))
    root_real = os.path.realpath(root)
    if not os.path.isdir(root_real):
        raise FileNotFoundError("project root not found: %r" % (root,))
    return root_real


def list_dir(root, rel="", limit=2000, show_hidden=False):
    root_real = _check_root(root)
    rel_norm = _norm_rel(rel)
    full = _resolve(root_real, rel_norm)
    _check_inside(root_real, full)
    if not os.path.isdir(full):
        raise FileNotFoundError("directory not found: %r" % (rel,))
    rel_out = _to_rel_str(rel_norm)
    if not rel_out:
        parent = None
    else:
        d = os.path.dirname(rel_norm)
        parent = None if d in ("", ".") and "/" not in rel_out else _to_rel_str(d)
        if parent is None and "/" not in rel_out:
            # single-level child of root -> parent is root ("")
            parent = ""
    entries = []
    try:
        with os.scandir(full) as it:
            for de in it:
                try:
                    name = de.name
                    ep = os.path.join(full, name)
                    if not show_hidden and _is_hidden(ep, name):
                        continue
                    try:
                        is_dir = de.is_dir(follow_symlinks=False)
                    except OSError:
                        continue
                    try:
                        st = de.stat(follow_symlinks=False)
                        size = 0 if is_dir else int(st.st_size)
                        mtime = float(st.st_mtime)
                    except OSError:
                        continue
                    if is_dir:
                        ext = ""
                    else:
                        ext = os.path.splitext(name)[1].lstrip(".").lower()
                    erel = (rel_out + "/" + name) if rel_out else name
                    entries.append({
                        "name": name,
                        "path": erel,
                        "is_dir": bool(is_dir),
                        "size": size,
                        "mtime": mtime,
                        "ext": ext,
                    })
                except OSError:
                    continue
    except FileNotFoundError:
        raise
    # guard against symlink/junction entries pointing outside root:
    # keep them listed (name visible) but that is fine; do NOT resolve them.
    # sort: dirs first, then files, natural case-insensitive
    from functools import cmp_to_key

    def cmp(a, b):
        if a["is_dir"] != b["is_dir"]:
            return -1 if a["is_dir"] else 1
        ka, kb = _nat_key(a["name"]), _nat_key(b["name"])
        if ka == kb:
            return 0
        return -1 if ka < kb else 1

    entries.sort(key=cmp_to_key(cmp))
    try:
        n = int(limit)
    except Exception:
        n = _LIMIT_DEFAULT
    entries = entries[:n]
    return {
        "path": full,
        "rel": rel_out,
        "parent": parent,
        "entries": entries,
    }


def breadcrumbs(root, rel):
    root_real = _check_root(root)
    rel_norm = _norm_rel(rel)
    full = _resolve(root_real, rel_norm)
    _check_inside(root_real, full)
    rel_out = _to_rel_str(rel_norm)
    if not rel_out:
        return []
    parts = rel_out.split("/")
    crumbs = []
    acc = []
    for p in parts:
        acc.append(p)
        crumbs.append({"name": p, "rel": "/".join(acc)})
    return crumbs
