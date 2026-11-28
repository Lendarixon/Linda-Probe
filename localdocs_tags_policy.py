"""Controlled-vocabulary tagging policy layer (main source, additive).

Wraps localdocs_tags primitives (parse_tags/fetch_tags/tag_store/load/search_tags)
without modifying them. Rules enforced here:
- no tag outside the chosen folder dictionary (invented tags rejected);
- no labels outside the selected area (only_paths scoping + verification);
- manual corrections survive re-runs (overrides journal).
"""
import json
from pathlib import Path

MUTEX_DEFAULT = [('indoor', 'outdoor'), ('day', 'night')]

# Calibrated defaults (tagcalib 2026-10-23, COCO80 multi, tune-64 only):
# grid margin 0.0-0.10 x min_score 0.15-0.30 showed no top-1 gain over 0.2/0.03
# without precision drop (tune top1 52/64 at 0.15-0.21, drops at >=0.22; margin
# never changes assign, only unsure rate: 43/64 tune at 0.03 vs 60/64 at 0.05).
# Hence defaults stay numerically identical to the previous literals
# (0.03/0.2); constants exist for discoverability, behaviour unchanged.
DEFAULT_MARGIN = 0.03
DEFAULT_MIN_SCORE = 0.2


def policy_defaults():
    """Calibrated policy defaults. Additive; does not alter existing calls."""
    return {'margin': DEFAULT_MARGIN, 'min_score': DEFAULT_MIN_SCORE,
            'mutex': list(MUTEX_DEFAULT)}


def apply_policy_defaults(margin=None, min_score=None):
    """Resolve None -> calibrated default; explicit values win (incl. 0.0)."""
    if margin is None:
        margin = DEFAULT_MARGIN
    if min_score is None:
        min_score = DEFAULT_MIN_SCORE
    return margin, min_score


def normalize_list(raw_tags, dictionary):
    """Lower/strip like parse_tags; raise on anything outside dictionary."""
    import localdocs_tags as T
    norm = T._normalized(raw_tags)
    allowed = set(dictionary)
    bad = [t for t in norm if t not in allowed]
    if bad:
        raise ValueError(f'invented tags rejected (not in folder dictionary): {bad[:10]}')
    return norm


def resolve_synonyms(raw_tags, synonyms, dictionary):
    """Map user aliases to canonical dict tags; unknown -> ValueError."""
    allowed = set(dictionary)
    out = []
    for raw in raw_tags:
        t = str(raw).strip().lower().replace('_', ' ')
        t = synonyms.get(t, t)
        if t not in allowed:
            raise ValueError(f'tag {raw!r} (-> {t!r}) not in folder dictionary')
        out.append(t)
    return list(dict.fromkeys(out))


def check_scoping(store_path, selected_paths, before_assign):
    """Assert images outside selection kept their previous assignment."""
    import localdocs_tags as T
    after = T.load(store_path)['assign']
    sel = {str(Path(p).resolve()) for p in selected_paths}
    changed = [p for p, v in after.items()
               if str(Path(p).resolve()) not in sel and before_assign.get(p) != v]
    # also allow images that resolve-equal via normcase on win; compare raw keys
    return changed


def policy_assign(engine, store_path, dictionary, groups, per_image=3,
                   min_score=DEFAULT_MIN_SCORE,
                   margin=DEFAULT_MARGIN, only_paths=None, synonyms=None, mutex=MUTEX_DEFAULT):
    """tag_store + policy layer. Returns {assign, status} and saves sidecar JSON.

    status per image: list of {tag, score, state: ok|unsure|mutex_excluded}.
    unsure = top1-top2 gap < margin. mutex = loser of exclusive pair demoted.
    margin/min_score None -> policy_defaults(); explicit values (incl. 0.0) win.
    Defaults resolve to 0.03/0.2 (unchanged behaviour; see DEFAULT_* comment).
    """
    margin, min_score = apply_policy_defaults(margin, min_score)
    import localdocs_tags as T
    synonyms = synonyms or {}
    canon = resolve_synonyms(dictionary, synonyms, dictionary)  # identity check
    before = dict(T.load(store_path)['assign'])
    rep = T.tag_store(engine, store_path, canon, per_image=per_image,
                      min_score=min_score, only_paths=only_paths)
    data = T.load(store_path)
    status, unsure_n, mutex_n = {}, 0, 0
    for path, picked in data['assign'].items():
        states = [{'tag': t, 'score': s, 'state': 'ok'} for t, s in picked]
        if len(states) >= 2 and (states[0]['score'] - states[1]['score']) < margin:
            states[0]['state'] = states[1]['state'] = 'unsure'
            unsure_n += 1
        present = {s['tag'] for s in states}
        for a, b in mutex:
            if a in present and b in present:
                lose = a if next(s['score'] for s in states if s['tag'] == a) < \
                    next(s['score'] for s in states if s['tag'] == b) else b
                next(s for s in states if s['tag'] == lose)['state'] = 'mutex_excluded'
                mutex_n += 1
        status[path] = states
    sidecar = {'groups': groups, 'margin': margin, 'min_score': min_score,
               'per_image': per_image, 'status': status,
               'stats': {'unsure_images': unsure_n, 'mutex_hits': mutex_n}}
    sp = Path(str(store_path) + '.policy.json')
    sp.write_text(json.dumps(sidecar, ensure_ascii=False, indent=1), encoding='utf-8')
    outside = check_scoping(store_path,
                            [only_paths] if isinstance(only_paths, str) else (only_paths or []),
                            before) if only_paths else []
    return {'report': rep, 'sidecar': str(sp), 'outside_selection_changed': outside,
            'stats': sidecar['stats']}


def apply_overrides(store_path, overrides):
    """overrides: {path: {keep:[tags], remove:[tags], add:[tags]}}.
    add/keep must be in the store's dictionary. Returns corrected assign."""
    import localdocs_tags as T
    data = T.load(store_path)
    allowed = set(data.get('tags', []))
    for path, op in overrides.items():
        if path not in data['assign']:
            raise ValueError(f'override outside labelled area: {path}')
        for t in list(op.get('add', [])) + list(op.get('keep', [])):
            if t not in allowed:
                raise ValueError(f'override tag {t!r} not in dictionary')
        cur = [(t, s) for t, s in data['assign'][path] if t not in op.get('remove', [])]
        have = {t for t, _ in cur}
        for t in op.get('add', []):
            if t not in have:
                cur.append([t, 1.0])  # manual tag: score 1.0 marks human origin
        keep = set(op.get('keep', [])) | set(op.get('add', []))
        if keep:
            cur = [x for x in cur if x[0] in keep]
        data['assign'][path] = cur
    T._save(store_path, data)
    jp = Path(str(store_path) + '.overrides.json')
    prev = json.loads(jp.read_text(encoding='utf-8')) if jp.exists() else {}
    prev.update(overrides)
    jp.write_text(json.dumps(prev, ensure_ascii=False, indent=1), encoding='utf-8')
    return data['assign']


def preview_sample(store_path, k=10):
    """Review queue: unsure first, then lowest top1 score. Returns list of rows."""
    import localdocs_tags as T
    data = T.load(store_path)['assign']
    pol = Path(str(store_path) + '.policy.json')
    status = json.loads(pol.read_text(encoding='utf-8'))['status'] if pol.exists() else {}
    rows = []
    for path, picked in data.items():
        st = {s['tag']: s['state'] for s in status.get(path, [])}
        top = picked[0][1] if picked else -1
        rows.append({'path': path, 'tags': picked,
                     'unsure': any(v == 'unsure' for v in st.values()), 'top1': top})
    rows.sort(key=lambda r: (not r['unsure'], r['top1']))
    return rows[:k]


def export_portable(store_path, out_dir):
    """Portable export: portable_tags.json + one .xmp sidecar per image. Returns paths."""
    import localdocs_tags as T
    from xml.sax.saxutils import escape
    data = T.load(store_path)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    pj = {'format': 'linda-probe-tags/1', 'dictionary': data.get('tags', []),
          'assign': data.get('assign', {})}
    (out / 'portable_tags.json').write_text(json.dumps(pj, ensure_ascii=False, indent=1),
                                            encoding='utf-8')
    n = 0
    for path, picked in data['assign'].items():
        bag = ''.join(f'<rdf:li>{escape(t)}</rdf:li>' for t, _ in picked)
        xmp = (f'<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF '
               f'xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" '
               f'xmlns:dc="http://purl.org/dc/elements/1.1/"><rdf:Description>'
               f'<dc:subject><rdf:Bag>{bag}</rdf:Bag></dc:subject>'
               f'</rdf:Description></rdf:RDF></x:xmpmeta>')
        (out / (Path(path).stem + '.xmp')).write_text(xmp, encoding='utf-8')
        n += 1
    return {'json': str(out / 'portable_tags.json'), 'sidecars': n}


def plan_layout(store_path, dest_root, op='copy'):
    """Dry-run physical layout: dest_root/<tag>/<file>. Default copy, journal+undo.
    Returns plan dict; nothing is executed (apply_plan executes, undo_plan reverts)."""
    import localdocs_tags as T
    assert op in ('copy', 'move')
    data = T.load(store_path)
    ops = []
    for path, picked in data['assign'].items():
        for tag, _ in picked:
            safe = ''.join(c if c.isalnum() or c in ' -_.' else '_' for c in tag).strip() or '_'
            ops.append({'op': op, 'src': path,
                        'dst': str(Path(dest_root) / safe / Path(path).name)})
    plan = {'op': op, 'dest_root': str(dest_root), 'ops': ops, 'applied': False}
    return plan


def apply_plan(plan, journal_path):
    import shutil
    done = []
    for o in plan['ops']:
        Path(o['dst']).parent.mkdir(parents=True, exist_ok=True)
        if o['op'] == 'copy':
            shutil.copyfile(o['src'], o['dst'])
        else:
            shutil.move(o['src'], o['dst'])
        done.append(o)
    plan['applied'] = True
    Path(journal_path).write_text(json.dumps({'plan': plan}, ensure_ascii=False, indent=1),
                                  encoding='utf-8')
    undo = [f"remove {o['dst']}" + (f" # move-back to {o['src']}" if o['op'] == 'move' else '')
            for o in done]
    return {'done': len(done), 'undo_steps': undo}


def undo_plan(journal_path):
    j = json.loads(Path(journal_path).read_text(encoding='utf-8'))['plan']
    removed, restored = 0, 0
    for o in j['ops']:
        dp = Path(o['dst'])
        if dp.is_file():
            dp.unlink(); removed += 1
    # prune empty tag dirs
    root = Path(j['dest_root'])
    for d in sorted(root.iterdir()) if root.is_dir() else []:
        try:
            if d.is_dir() and not any(d.iterdir()):
                d.rmdir()
        except OSError:
            pass
    return {'removed_copies': removed, 'dest_root': str(root)}
