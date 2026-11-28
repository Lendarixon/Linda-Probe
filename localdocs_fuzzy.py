"""Typo-tolerant file-name search (F1).

Math-lab verdict implemented (work/math_lab/exp_fuzzy_names.py, REPORT.md)::

    trigram (q=3) count filter + bounded Damerau-Levenshtein (OSA) rescoring.

Pipeline per query:

1. normalise query and names: basename, casefold, diacritics stripped
   (Polish ``l``/``o`` mappings, ``e`` -> ``e``, NFKD Mn strip).
2. length filter ``abs(len - len_q) <= k``.
3. numpy-vectorised trigram posting filter: shared-trigram overlap
   ``>= max(n_q(doc), n_q(query)) - (q+1)*k``.  The ``(q+1)*k`` term (not
   ``q*k``) keeps single adjacent transpositions safe: one transposition
   can destroy up to ``q+1`` q-grams (``q-1`` covering both chars plus two
   covering exactly one). Unit-Levenshtein labs use ``q*k``; Damerau needs
   the looser bound or transpositions are false negatives.
4. numpy-vectorised bounded Levenshtein (Ukkonen band, as in the lab
   ``BandedNames``) with bound ``2*k`` to keep every Damerau<=k candidate
   (Levenshtein <= 2*Damerau), then exact bounded OSA Damerau-Levenshtein
   (adjacent transposition costs 1) for the survivors.

Sorting is ``(distance, len(normalised_basename), row_index)`` and the
result is truncated to ``limit``. Reference recall is the bounded
Damerau universe; the filter changes execution, not the reference.

Memory note: retained Python/numpy objects only, not peak RSS; the
full normalised-name store is required for exact rescoring in addition
to the trigram postings. Synthetic measurements cannot establish
production performance.
"""

from __future__ import annotations

import unicodedata
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np

Q = 3
_Q_PENALTY = Q + 1  # max q-grams destroyed by one Damerau edit (transposition)

__all__ = ["fuzzy_search", "normalize_name"]

# Characters that do not decompose under NFKD and need explicit folding.
# Applied after casefold, before NFKD strip.
_EXTRA_FOLD = {
    "ł": "l",
    "ø": "o",
    "œ": "oe",
    "æ": "ae",
    "ß": "ss",
    "đ": "d",
    "ð": "d",
    "þ": "th",
    "ŋ": "n",
    "ħ": "h",
    "ı": "i",
    "ĳ": "ij",
    "ё": "е",
}


def _basename(path: str) -> str:
    return path.replace("\\", "/").rsplit("/", 1)[-1]


def normalize_name(name: str) -> str:
    """Basename, case-insensitive, diacritics-stripped form used for matching."""
    base = _basename(name).casefold()
    for src, dst in _EXTRA_FOLD.items():
        if src in base:
            base = base.replace(src, dst)
    decomposed = unicodedata.normalize("NFKD", base)
    return "".join(c for c in decomposed if unicodedata.category(c) != "Mn")


def _query_grams(query_norm: str) -> Counter:
    if len(query_norm) < Q:
        return Counter()
    return Counter(query_norm[i : i + Q] for i in range(len(query_norm) - Q + 1))


def _damerau_osa(a: str, b: str, k: int) -> int:
    """Optimal-string-alignment distance, capped at k+1.

    Adjacent transposition costs 1. Each substring edited at most once;
    for k<=2 this matches true Damerau-Levenshtein except pathological
    overlapping multi-edit cases.
    """
    la, lb = len(a), len(b)
    if abs(la - lb) > k:
        return k + 1
    if a == b:
        return 0
    # Full (la+1)x(lb+1) DP; strings are short (basenames), so this is cheap.
    prev_prev = None
    prev = list(range(lb + 1))
    # keep row i-2 for transposition
    rows = [prev]
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        ca = a[i - 1]
        for j in range(1, lb + 1):
            cost = 0 if ca == b[j - 1] else 1
            best = prev[j] + 1  # deletion
            ins = cur[j - 1] + 1  # insertion
            if ins < best:
                best = ins
            sub = prev[j - 1] + cost  # substitution
            if sub < best:
                best = sub
            if i > 1 and j > 1 and ca == b[j - 2] and a[i - 2] == b[j - 1]:
                tr = rows[i - 2][j - 2] + 1  # transposition
                if tr < best:
                    best = tr
            cur[j] = best
        rows.append(cur)
        prev = cur
    dist = rows[la][lb]
    return dist if dist <= k else k + 1


def _banded_levenshtein_subset(
    chars: np.ndarray,
    lengths: np.ndarray,
    query: str,
    k: int,
    ids: np.ndarray,
) -> np.ndarray:
    """Vectorised bounded Levenshtein (Ukkonen band) over a candidate subset.

    Mirrors the lab ``BandedNames.distances`` execution. Returns int array
    aligned with ``ids``, capped at k+1. ``chars`` is the full
    (N, maxlen+1) uint32 matrix, ``lengths`` the full int32 vector.
    """
    if len(ids) == 0:
        return np.empty(0, dtype=np.int16)
    # Length pre-filter (callers also filter, kept here for safety).
    keep = np.abs(lengths[ids] - len(query)) <= k
    valid_ids = ids[keep]
    out = np.full(len(ids), k + 1, dtype=np.int16)
    if len(valid_ids) == 0:
        return out
    sub_chars = chars[valid_ids]
    sub_lengths = lengths[valid_ids]
    width, inf = 2 * k + 1, k + 1
    prev = np.full((width, len(valid_ids)), inf, dtype=np.int16)
    for d in range(0, k + 1):
        prev[d + k] = np.where(sub_lengths >= d, d, inf)
    q_ords = [ord(c) for c in query]
    for i, ch in enumerate(q_ords, 1):
        row = np.empty_like(prev)
        row.fill(inf)
        for off, d in enumerate(range(-k, k + 1)):
            j = i + d
            if j < 0:
                continue
            if j == 0:
                row[off] = i
                continue
            if j > sub_chars.shape[1]:
                continue
            value = prev[off] + (sub_chars[:, j - 1] != ch)
            if off + 1 < width:
                value = np.minimum(value, prev[off + 1] + 1)
            if off:
                value = np.minimum(value, row[off - 1] + 1)
            row[off] = np.where(sub_lengths >= j, np.minimum(value, inf), inf)
        prev = row
    dist = prev[sub_lengths - len(query) + k, np.arange(len(valid_ids))]
    out[np.flatnonzero(keep)] = dist
    return out


_CACHE: Dict[str, object] = {"ref": None, "id": None, "n": -1, "data": None}


def _build_index(norm_names: List[str]) -> dict:
    n = len(norm_names)
    lengths = np.array([len(s) for s in norm_names], dtype=np.int32)
    ngrams = np.maximum(lengths - Q + 1, 0).astype(np.int32)
    maxlen = int(lengths.max()) if n else 0
    chars = np.zeros((n, maxlen + 1), dtype=np.uint32)
    for i, s in enumerate(norm_names):
        if s:
            chars[i, : len(s)] = [ord(c) for c in s]
    postings: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    tmp_idx: Dict[str, list] = {}
    tmp_cnt: Dict[str, list] = {}
    for idx, s in enumerate(norm_names):
        if len(s) < Q:
            continue
        counts = Counter(s[i : i + Q] for i in range(len(s) - Q + 1))
        for tri, c in counts.items():
            if tri in tmp_idx:
                tmp_idx[tri].append(idx)
                tmp_cnt[tri].append(c)
            else:
                tmp_idx[tri] = [idx]
                tmp_cnt[tri] = [c]
    for tri, idxs in tmp_idx.items():
        postings[tri] = (
            np.array(idxs, dtype=np.int64),
            np.array(tmp_cnt[tri], dtype=np.int32),
        )
    return {
        "norm": norm_names,
        "lengths": lengths,
        "ngrams": ngrams,
        "chars": chars,
        "postings": postings,
    }


def _get_index(names_lower: List[str]) -> dict:
    key_id = id(names_lower)
    key_n = len(names_lower)
    if (
        _CACHE["ref"] is names_lower
        and _CACHE["id"] == key_id
        and _CACHE["n"] == key_n
    ):
        return _CACHE["data"]  # type: ignore[return-value]
    norm = [normalize_name(s) for s in names_lower]
    data = _build_index(norm)
    _CACHE["ref"] = names_lower
    _CACHE["id"] = key_id
    _CACHE["n"] = key_n
    _CACHE["data"] = data
    return data


def fuzzy_search(
    names_lower: List[str],
    query: str,
    max_edits: Optional[int] = None,
    limit: int = 50,
) -> List[Tuple[int, int]]:
    """Typo-tolerant file-name search.

    Args:
        names_lower: candidate file names/paths (any case; basenames used).
        query: file-name query (any case; basename used).
        max_edits: Damerau bound. ``None`` selects 1 for normalised query
            length <= 5 else 2.
        limit: max rows returned.

    Returns:
        List of ``(row_index, distance)`` sorted by ``(distance,
        len(normalised_basename), row_index)``. Distances are bounded
        Damerau-Levenshtein (adjacent transposition = 1) on normalised
        basenames. Queries shorter than 3 chars (after normalisation)
        return [].
    """
    if not isinstance(query, str):
        query = str(query)
    qn = normalize_name(query)
    if len(qn) < 3:
        return []
    if max_edits is None:
        k = 1 if len(qn) <= 5 else 2
    else:
        k = int(max_edits)
    if k < 0 or limit <= 0:
        return []
    if len(names_lower) == 0:
        return []

    index = _get_index(names_lower)
    norm_names: List[str] = index["norm"]  # type: ignore[assignment]
    lengths: np.ndarray = index["lengths"]  # type: ignore[assignment]
    ngrams: np.ndarray = index["ngrams"]  # type: ignore[assignment]
    chars: np.ndarray = index["chars"]  # type: ignore[assignment]
    postings: Dict[str, Tuple[np.ndarray, np.ndarray]] = index["postings"]  # type: ignore[assignment]

    n = len(norm_names)
    q_len = len(qn)
    q_counter = _query_grams(qn)
    nq_q = max(q_len - Q + 1, 0)

    overlap = np.zeros(n, dtype=np.int32)
    for tri, cq in q_counter.items():
        posting = postings.get(tri)
        if posting is None:
            continue
        idxs, cnts = posting
        # min(count_doc, count_query) per posting, vectorised
        if cq >= int(cnts.max()):
            overlap[idxs] += cnts
        elif cq == 1:
            # common case: min(cnt,1) == 1 for every doc containing tri
            overlap[idxs] += 1
        else:
            overlap[idxs] += np.minimum(cnts, cq)

    threshold = np.maximum(ngrams, nq_q) - _Q_PENALTY * k
    cand_mask = (overlap >= threshold) & (np.abs(lengths - q_len) <= k)
    cand_ids = np.flatnonzero(cand_mask)
    if len(cand_ids) == 0:
        return []

    # Vectorised Levenshtein with bound 2k keeps every Damerau<=k row.
    kb = 2 * k
    lev = _banded_levenshtein_subset(chars, lengths, qn, kb, cand_ids)
    near_mask = lev <= kb
    near_ids = cand_ids[near_mask]
    if len(near_ids) == 0:
        return []

    scored: List[Tuple[int, int]] = []
    for idx in near_ids.tolist():
        d = _damerau_osa(qn, norm_names[int(idx)], k)
        if d <= k:
            scored.append((int(idx), int(d)))
    if not scored:
        return []
    scored.sort(key=lambda t: (t[1], len(norm_names[t[0]]), t[0]))
    return scored[: int(limit)]
