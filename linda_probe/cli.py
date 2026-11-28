"""Standardized CLI for Linda Probe (`linda-probe` / `python -m linda_probe`)."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
from pathlib import Path
from typing import Any

from linda_probe import __version__


def _configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def _parse_scalar(raw: str) -> Any:
    low = raw.strip().lower()
    if low in ("true", "yes", "1", "on"):
        return True
    if low in ("false", "no", "0", "off"):
        return False
    try:
        return int(raw)
    except ValueError:
        return raw


def _print_json(data: Any) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2, default=str))


def _print_rows(rows: list[dict[str, Any]], columns: list[str], fmt: str) -> None:
    if fmt == "json":
        _print_json(rows)
        return
    if fmt == "csv":
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in columns})
        print(buf.getvalue(), end="")
        return
    if fmt == "md":
        print("| " + " | ".join(columns) + " |")
        print("| " + " | ".join("---" for _ in columns) + " |")
        for row in rows:
            vals = [str(row.get(k, "")).replace("|", "\\|").replace("\n", " ") for k in columns]
            print("| " + " | ".join(vals) + " |")
        return

    # Plain aligned table
    if not rows:
        print("(no results)")
        return
    widths = [len(c) for c in columns]
    str_rows: list[list[str]] = []
    for row in rows:
        srow = [str(row.get(k, "")).replace("\n", " ") for k in columns]
        for idx, val in enumerate(srow):
            widths[idx] = min(64, max(widths[idx], len(val)))
        str_rows.append(srow)
    header = "  ".join(c.ljust(widths[i]) for i, c in enumerate(columns))
    print(header)
    print("  ".join("-" * widths[i] for i in range(len(columns))))
    for srow in str_rows:
        cells = [
            (val[: widths[i] - 1] + "…") if len(val) > widths[i] else val.ljust(widths[i])
            for i, val in enumerate(srow)
        ]
        print("  ".join(cells))


def _cmd_find(args: argparse.Namespace) -> int:
    import localdocs_quick

    roots = [str(Path(r).resolve()) for r in args.root] if args.root else localdocs_quick.default_roots()
    qi = localdocs_quick.QuickIndex(roots)
    qi.rescan_now()
    res = qi.search(args.query, limit=args.limit, only_roots=roots if args.root else None)
    if args.format == "json":
        _print_json(res)
        return 0
    hits = res.get("results", []) if isinstance(res, dict) else res
    rows = [
        {"name": h.get("name", ""), "size": h.get("size", ""), "path": h.get("path", "")}
        for h in hits
    ]
    _print_rows(rows, ["name", "size", "path"], args.format)
    return 0


def _cmd_index(args: argparse.Namespace) -> int:
    from project_search.backend import index_project

    folder = str(Path(args.folder).resolve())
    if not os.path.isdir(folder):
        print(f"error: folder not found: {folder}", file=sys.stderr)
        return 1
    res = index_project(folder, max_files=args.max_files, max_chunks=args.max_chunks)
    if args.format == "json":
        _print_json(res)
    else:
        print(f"indexed: {folder}")
        if isinstance(res, dict):
            for k in ("files", "chunks", "elapsed_s", "status"):
                if k in res:
                    print(f"  {k}: {res[k]}")
    return 0


def _cmd_search(args: argparse.Namespace) -> int:
    from project_search.backend import search_project

    folder = str(Path(args.folder).resolve())
    if not os.path.isdir(folder):
        print(f"error: folder not found: {folder}", file=sys.stderr)
        return 1
    res = search_project(folder, args.query, k=args.limit)
    if args.format == "json":
        _print_json(res)
        return 0
    hits = res.get("results", []) if isinstance(res, dict) else res
    rows = [
        {
            "score": round(float(h.get("score", 0.0)), 4) if h.get("score") is not None else "",
            "path": h.get("path", h.get("file", "")),
            "snippet": str(h.get("text", h.get("snippet", "")))[:120],
        }
        for h in hits
    ]
    _print_rows(rows, ["score", "path", "snippet"], args.format)
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    from project_search.backend import list_projects

    res = list_projects()
    if args.format == "json":
        _print_json(res)
        return 0
    items = res.get("projects", []) if isinstance(res, dict) else res
    rows = [
        it if isinstance(it, dict) else {"folder": str(it)}
        for it in items
    ]
    cols = list(rows[0].keys()) if rows else ["folder"]
    _print_rows(rows, cols, args.format)
    return 0


def _cmd_dupes(args: argparse.Namespace) -> int:
    import localdocs_dupes

    roots = [str(Path(r).resolve()) for r in args.folders]
    for r in roots:
        if not os.path.isdir(r):
            print(f"error: folder not found: {r}", file=sys.stderr)
            return 1
    res = localdocs_dupes.find_duplicates(roots, min_size=args.min_size, max_files=args.max_files)
    if args.format == "json":
        _print_json(res)
        return 0
    groups = res.get("groups", []) if isinstance(res, dict) else []
    rows = []
    for idx, g in enumerate(groups, 1):
        paths = g.get("paths", g.get("files", []))
        rows.append(
            {
                "group": idx,
                "size": g.get("size", ""),
                "count": len(paths),
                "paths": " ; ".join(paths),
            }
        )
    _print_rows(rows, ["group", "size", "count", "paths"], args.format)
    return 0


def _cmd_settings(args: argparse.Namespace) -> int:
    import localdocs_settings

    if args.settings_action == "show":
        cfg = localdocs_settings.get()
        if args.format == "json":
            _print_json(cfg)
        else:
            rows = [{"key": k, "value": v} for k, v in sorted(cfg.items())]
            _print_rows(rows, ["key", "value"], args.format)
        return 0

    if args.settings_action == "get":
        cfg = localdocs_settings.get()
        if args.key not in cfg:
            print(f"error: unknown setting '{args.key}'", file=sys.stderr)
            return 1
        val = cfg[args.key]
        if args.format == "json":
            _print_json({args.key: val})
        else:
            print(val)
        return 0

    if args.settings_action == "set":
        val = _parse_scalar(args.value)
        updated = localdocs_settings.update({args.key: val})
        if args.format == "json":
            _print_json({args.key: updated.get(args.key)})
        else:
            print(f"{args.key} = {updated.get(args.key)}")
        return 0

    if args.settings_action == "reset":
        cfg = localdocs_settings.reset()
        if args.format == "json":
            _print_json(cfg)
        else:
            print("settings reset to defaults")
        return 0

    return 2


def _cmd_models(args: argparse.Namespace) -> int:
    import localdocs_models

    items = localdocs_models.list_models(kind=args.kind)
    if args.format == "json":
        _print_json(items)
        return 0
    rows = []
    for m in items:
        rows.append(
            {
                "id": m.get("id", ""),
                "kind": m.get("kind", ""),
                "license": m.get("license", ""),
                "size_mb": m.get("size_mb", ""),
                "active": "yes" if m.get("active") else "",
                "repo": m.get("repo", ""),
            }
        )
    _print_rows(rows, ["id", "kind", "license", "size_mb", "active", "repo"], args.format)
    return 0


def _cmd_ocr(args: argparse.Namespace) -> int:
    import localdocs_ocr

    target = str(Path(args.file).resolve())
    if not os.path.isfile(target):
        print(f"error: file not found: {target}", file=sys.stderr)
        return 1
    res = localdocs_ocr.ocr_image(target, lang=args.lang, device=args.device, engine=args.engine)
    if args.format == "json":
        _print_json(res)
    else:
        text = res.get("text", "") if isinstance(res, dict) else str(res)
        print(text)
    return 0


def _cmd_doctor(args: argparse.Namespace) -> int:
    from project_search.doctor import run_checks

    checks = run_checks()
    if args.json or args.format == "json":
        _print_json(checks)
    else:
        for c in checks:
            st = "OK" if c["ok"] is True else ("WARN" if c.get("ok") == "warn" else "FAIL")
            print(f"{st:4} {c['name']}: {c['detail']}")
            if c.get("hint") and st != "OK":
                print(f"      hint: {c['hint']}")
        nf = sum(1 for c in checks if c["ok"] is False)
        nw = sum(1 for c in checks if c.get("ok") == "warn")
        print(f"result: {len(checks)} checks, {nf} FAIL, {nw} WARN")
    return 1 if any(c["ok"] is False for c in checks) else 0


def _cmd_serve(args: argparse.Namespace) -> int:
    import runpy

    server_path = str(Path(__file__).resolve().parents[1] / "localdocs_server.py")
    if args.host:
        os.environ.setdefault("LOCALDOCS_HOST", args.host)
    sys.argv = [server_path, str(args.port)]
    runpy.run_path(server_path, run_name="__main__")
    return 0


def _cmd_gui(args: argparse.Namespace) -> int:
    import linda_probe_app

    sys.argv = ["linda_probe_app.py", "--port", str(args.port)]
    linda_probe_app.main()
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="linda-probe",
        description="Linda Probe — local Windows file search, offline OCR, and semantic/photo search.",
    )
    p.add_argument("--version", action="version", version=f"linda-probe {__version__}")

    fmt_parent = argparse.ArgumentParser(add_help=False)
    fmt_parent.add_argument(
        "--format",
        choices=("table", "json", "csv", "md"),
        default="table",
        help="Output format (default: table)",
    )

    sub = p.add_subparsers(dest="command", required=True)

    # find
    p_find = sub.add_parser("find", parents=[fmt_parent], help="Instant filename search")
    p_find.add_argument("query", help="Filename query")
    p_find.add_argument("--root", action="append", default=[], help="Folder(s) to scan (repeatable)")
    p_find.add_argument("--limit", type=int, default=30, help="Maximum results (default: 30)")
    p_find.set_defaults(func=_cmd_find)

    # index
    p_idx = sub.add_parser("index", parents=[fmt_parent], help="Index a folder for content/semantic search")
    p_idx.add_argument("folder", help="Folder path")
    p_idx.add_argument("--max-files", type=int, default=10000)
    p_idx.add_argument("--max-chunks", type=int, default=30000)
    p_idx.set_defaults(func=_cmd_index)

    # search
    p_srch = sub.add_parser("search", parents=[fmt_parent], help="Search an indexed folder")
    p_srch.add_argument("folder", help="Indexed folder path")
    p_srch.add_argument("query", help="Search query")
    p_srch.add_argument("--limit", type=int, default=10, help="Maximum hits (default: 10)")
    p_srch.set_defaults(func=_cmd_search)

    # list
    p_list = sub.add_parser("list", parents=[fmt_parent], help="List indexed folders")
    p_list.set_defaults(func=_cmd_list)

    # dupes
    p_dupes = sub.add_parser("dupes", parents=[fmt_parent], help="Find duplicate files by size and BLAKE2b hash")
    p_dupes.add_argument("folders", nargs="+", help="Folder(s) to scan")
    p_dupes.add_argument("--min-size", type=int, default=1024, help="Minimum file size in bytes (default: 1024)")
    p_dupes.add_argument("--max-files", type=int, default=200000, help="Maximum files to scan")
    p_dupes.set_defaults(func=_cmd_dupes)

    # settings
    p_set = sub.add_parser("settings", help="Inspect or update Probe settings")
    set_sub = p_set.add_subparsers(dest="settings_action", required=True)
    set_show = set_sub.add_parser("show", parents=[fmt_parent], help="Show all settings")
    set_show.set_defaults(func=_cmd_settings)
    set_get = set_sub.add_parser("get", parents=[fmt_parent], help="Get one setting")
    set_get.add_argument("key")
    set_get.set_defaults(func=_cmd_settings)
    set_set = set_sub.add_parser("set", parents=[fmt_parent], help="Set one setting")
    set_set.add_argument("key")
    set_set.add_argument("value")
    set_set.set_defaults(func=_cmd_settings)
    set_reset = set_sub.add_parser("reset", parents=[fmt_parent], help="Reset settings to defaults")
    set_reset.set_defaults(func=_cmd_settings)

    # models
    p_mod = sub.add_parser("models", parents=[fmt_parent], help="List available OCR and embedding models")
    p_mod.add_argument("--kind", default=None, help="Filter by model kind (e.g. ocr_rec, image_embed)")
    p_mod.set_defaults(func=_cmd_models)

    # ocr
    p_ocr = sub.add_parser("ocr", parents=[fmt_parent], help="Run local OCR on an image")
    p_ocr.add_argument("file", help="Image file path")
    p_ocr.add_argument("--lang", default="rus+eng+pol", help="OCR languages (default: rus+eng+pol)")
    p_ocr.add_argument("--device", default="cpu", choices=("cpu", "gpu"))
    p_ocr.add_argument("--engine", default="auto")
    p_ocr.set_defaults(func=_cmd_ocr)

    # doctor
    p_doc = sub.add_parser("doctor", parents=[fmt_parent], help="Run system and runtime diagnostics")
    p_doc.add_argument("--json", action="store_true", help="Shortcut for --format json")
    p_doc.set_defaults(func=_cmd_doctor)

    # serve
    p_srv = sub.add_parser("serve", help="Start the local HTTP UI/API server")
    p_srv.add_argument("--host", default="127.0.0.1")
    p_srv.add_argument("--port", type=int, default=8798)
    p_srv.set_defaults(func=_cmd_serve)

    # gui
    p_gui = sub.add_parser("gui", help="Launch the Linda Probe desktop window")
    p_gui.add_argument("--port", type=int, default=8798)
    p_gui.set_defaults(func=_cmd_gui)

    return p


def main(argv: list[str] | None = None) -> int:
    _configure_stdio()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except BrokenPipeError:
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
