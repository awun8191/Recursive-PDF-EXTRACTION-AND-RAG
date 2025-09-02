from typing import List, Dict, Any, Tuple
from src.services.RAG.log_utils import get_logger, snapshot

log = get_logger("chunk")

def split_paragraphs(text: str) -> List[str]:
    import re
    text = re.sub(r"\r\n?", "\n", text)
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return paras if paras else ([text.strip()] if text.strip() else [])


def merge_paras(paras: List[str], min_chars: int, max_chars: int) -> List[str]:
    out, buf = [], ""
    for p in paras:
        if not buf:
            buf = p
            continue
        if len(buf) < min_chars or (len(buf) + 2 + len(p) <= max_chars):
            buf = f"{buf}\n\n{p}"
        else:
            out.append(buf)
            buf = p
    if buf:
        out.append(buf)
    return out


def chunk(text: str, min_chars=200, max_chars=1600, overlap=80) -> List[str]:
    base = merge_paras(split_paragraphs(text), min_chars, max_chars)
    if overlap <= 0 or len(base) <= 1:
        log.info(f"[CHUNK] produced={len(base)} first='{snapshot(base[0]) if base else ''}'")
        return base
    out = [base[0]]
    for i in range(1, len(base)):
        tail = base[i-1][-overlap:]
        sp = tail.find(" ")
        if sp > 0:
            tail = tail[sp+1:]
        out.append(f"{tail} {base[i]}")
    log.info(f"[CHUNK] produced={len(out)} first='{snapshot(out[0]) if out else ''}'")
    return out


def sha1_text(s: str) -> str:
    import hashlib
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def dedupe(chunks: List[str]) -> Tuple[List[str], Dict[int, Tuple[int, str]]]:
    seen: Dict[str, int] = {}
    keep, dup = [], {}
    for i, c in enumerate(chunks):
        h = sha1_text(c)
        if h in seen:
            dup[i] = (seen[h], h)
        else:
            seen[h] = len(keep)
            keep.append(c)
    log.info(f"[CHUNK] dedupe orig={len(chunks)} uniq={len(keep)} dups={len(dup)}")
    return keep, dup

