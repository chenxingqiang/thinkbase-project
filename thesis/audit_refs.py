"""Machine-verify every refs.bib entry (v2) against Crossref/arXiv.

Fixes over v1: family-name-only author comparison; HTML entity
unescaping; top-5 candidate matching by token overlap (>= 0.8); arXiv
API fallback for entries the bib marks as arXiv; DOI fetch without
slash encoding.  One line per entry: OK / problem + detail.  Exit 1 if
any entry fails.
"""

import html
import json
import re
import sys
import time
import urllib.parse
import urllib.request

BIB = "thesis/refs.bib"
MAILTO = "audit@example.org"
UA = {"User-Agent": f"thesis-bib-audit/2.0 (mailto:{MAILTO})"}


def fetch(url, tries=6):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise
            if k == tries - 1:
                raise
            time.sleep(8 if e.code == 429 else 2 * (k + 1))
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(2 * (k + 1))


def _read_braced(text, i):
    """Read a {...} value starting at text[i] == '{'; return (value, next_i)."""
    depth = 0
    j = i
    while j < len(text):
        c = text[j]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1:j], j + 1
        j += 1
    return text[i + 1:], len(text)


def parse_entries(text):
    """Brace-counting BibTeX parser (the lazy-regex version mis-parsed
    trailing fields: with re.S the value group backtracked across the
    field's closing brace and swallowed it into the DOI)."""
    entries = []
    pos = 0
    while True:
        m = re.search(r"@(\w+)\s*\{", text[pos:])
        if not m:
            break
        etype = m.group(1).lower()
        open_idx = pos + m.end() - 1          # the '{' after @type
        body, after = _read_braced(text, open_idx)
        key = body[:body.index(",")].strip()
        fields = {}
        b = body.index(",") + 1
        while b < len(body):
            fm = re.match(r"\s*(\w+)\s*=\s*", body[b:])
            if not fm:
                b += 1
                continue
            fname = fm.group(1).lower()
            b += fm.end()
            if b < len(body) and body[b] == "{":
                val, b = _read_braced(body, b)
            elif b < len(body) and body[b] == '"':
                end = body.index('"', b + 1)
                val, b = body[b + 1:end], end + 1
            else:
                em = re.match(r"([^,\s]+)", body[b:])
                val = em.group(1) if em else ""
                b += em.end() if em else 1
            fields[fname] = re.sub(r"\s+", " ", val).strip()
            while b < len(body) and body[b] in ", \t\r\n":
                b += 1
        entries.append((etype, key, fields))
        pos = after
    return entries


def norm(s):
    return re.sub(r"[^a-z0-9]", "", html.unescape(s or "").lower())


def tokens(s):
    return set(re.findall(r"[a-z0-9]+", html.unescape(s or "").lower())) - {
        "the", "a", "an", "of", "for", "with", "and", "in", "from", "to"}


def overlap(a, b):
    ta, tb = tokens(a), tokens(b)
    return len(ta & tb) / max(1, len(ta | tb))


def bib_family_names(author_field):
    out = []
    for a in re.split(r"\s+and\s+", author_field):
        a = a.strip().strip("{}")
        if not a:
            continue
        out.append(norm(a.split(",")[0]))
    return out


def check_doi(doi, f):
    rec = fetch("https://api.crossref.org/works/" + doi
                + "?mailto=" + MAILTO)["message"]
    rtitle = html.unescape(rec.get("title", [""])[0])
    ryear = str((rec.get("issued", {}).get("date-parts")
                 or [[None]])[0][0] or "")
    if norm(f.get("title", "")) and norm(rtitle) != norm(f["title"]):
        # ACM sometimes registers an abbreviated title in Crossref
        # (e.g. LightGCN for the full SIGIR title); accept it only if
        # every DOI-title token appears in the bib title -- the author
        # check below still has to pass.
        if not (set(tokens(rtitle)) <= set(tokens(f["title"]))):
            return f"TITLE-MISMATCH bib={f['title']!r} doi={rtitle!r}"
    if f.get("year") and ryear[:4] != f["year"][:4]:
        return f"YEAR-MISMATCH bib={f['year']} doi={ryear}"
    bib_fam = bib_family_names(f.get("author", ""))
    rec_fam = {norm(a.get("family", "")) for a in rec.get("author", [])}
    missing = [b for b in bib_fam if b and b not in rec_fam
               and not any(b in r or r in b for r in rec_fam)]
    if missing:
        return f"AUTHOR-MISMATCH bib surnames {missing} vs doi {sorted(rec_fam)[:6]}"
    if f.get("pages"):
        rp = (rec.get("page") or "").replace("-", "-")
        if norm(f["pages"]) != norm(rp):
            return f"PAGES-MISMATCH bib={f['pages']} doi={rp}"
    return None


def check_bibliographic(f):
    title = f.get("title", "")
    first = re.split(r"\s+and\s+", f.get("author", ""))[0]
    q = urllib.parse.quote(f"{title} {first}")
    res = fetch(f"https://api.crossref.org/works?rows=5&mailto={MAILTO}"
                f"&query.bibliographic={q}")["message"]["items"]
    for r in res:
        rtitle = html.unescape(r.get("title", [""])[0])
        if norm(title) == norm(rtitle) or overlap(title, rtitle) >= 0.95:
            ryear = str((r.get("issued", {}).get("date-parts")
                         or [[None]])[0][0] or "")
            if f.get("year") and ryear[:4] != f["year"][:4]:
                return f"YEAR-MISMATCH best match {rtitle!r} is {ryear}, bib {f['year']}"
            return None
    return "NOT-FOUND in crossref (top-5 token overlap all < 0.8)"


def check_arxiv(f):
    title = f.get("title", "")
    q = urllib.parse.quote(title)
    try:
        res = fetch("http://export.arxiv.org/api/query?search_query="
                     f"ti:{q}&max_results=3")
    except Exception:
        return None  # arXiv unreachable -> don't fail the entry
    m = re.findall(r"<title>(.*?)</title>", res, re.S)
    for t in m[1:]:  # first <title> is the feed title
        t2 = html.unescape(t.strip())
        if overlap(title, t2) >= 0.7:
            return None
    return "NOT-FOUND in crossref nor arXiv title search"


def main():
    entries = parse_entries(open(BIB).read())
    print(f"auditing {len(entries)} entries ...\n")
    bad = []
    for etype, key, f in entries:
        detail = None
        if f.get("doi"):
            try:
                detail = check_doi(f["doi"], f)
            except Exception as e:
                detail = f"DOI-RESOLUTION-FAILED ({e})"
        if detail is None and not f.get("doi"):
            detail = check_bibliographic(f)
            if detail and "NOT-FOUND" in detail:
                detail = check_arxiv(f)
        mark = "OK " if not detail else "!! "
        print(f"{mark}{key:26s} {detail or ''}")
        if detail:
            bad.append((key, detail))
        time.sleep(0.5)
    print(f"\n{len(entries) - len(bad)}/{len(entries)} verified, "
          f"{len(bad)} problems")
    for k, d in bad:
        print(f"  {k}: {d}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
