"""
Minimal pure-Python reader for R .RDS files (serialization format 2/3, XDR).

Written because pyreadr silently returns an empty dict for RDS files whose
top-level object is a bare list (VECSXP) rather than a data.frame, which is
exactly how the Mammalian Methylation Consortium ships several of its files.

Supports the subset of R's serialization grammar actually used by those files:
NULL, symbols, pairlists (used for attributes), character/integer/real/logical
vectors, generic lists, and the reference table for repeated symbols.

Returns plain Python objects; data.frames come back as dicts of columns with an
"_attr" entry preserving names/row.names/class so callers can rebuild a
pandas.DataFrame.
"""

import gzip
import struct

# --- R SEXP type codes -------------------------------------------------------
NILSXP, SYMSXP, LISTSXP, CLOSXP, ENVSXP, PROMSXP, LANGSXP = 0, 1, 2, 3, 4, 5, 6
SPECIALSXP, BUILTINSXP, CHARSXP, LGLSXP = 7, 8, 9, 10
INTSXP, REALSXP, CPLXSXP, STRSXP = 13, 14, 15, 16
DOTSXP, ANYSXP, VECSXP, EXPRSXP = 17, 18, 19, 20
BCODESXP, EXTPTRSXP, WEAKREFSXP, RAWSXP, S4SXP = 21, 22, 23, 24, 25
EMPTYENV_SXP, BASEENV_SXP, GLOBALENV_SXP = 242, 241, 253
NILVALUE_SXP, UNBOUNDVALUE_SXP, MISSINGARG_SXP = 254, 251, 252
REFSXP, PACKAGESXP, NAMESPACESXP, ALTREP_SXP = 255, 240, 239, 238

NA_INTEGER = -2147483648


class RDSReader:
    def __init__(self, data: bytes):
        self.buf = data
        self.pos = 0
        self.refs = []  # reference table (symbols, environments)

    # --- primitive readers ---------------------------------------------------
    def _int(self) -> int:
        v = struct.unpack_from(">i", self.buf, self.pos)[0]
        self.pos += 4
        return v

    def _double(self) -> float:
        v = struct.unpack_from(">d", self.buf, self.pos)[0]
        self.pos += 8
        return v

    def _bytes(self, n: int) -> bytes:
        v = self.buf[self.pos:self.pos + n]
        self.pos += n
        return v

    # --- header --------------------------------------------------------------
    def read_header(self):
        fmt = self._bytes(2)
        if fmt[:1] != b"X":
            raise ValueError(f"only XDR-format RDS supported, got {fmt!r}")
        self.pos += 0  # 'X\n' already consumed
        self.version = self._int()
        self._int()  # writer R version
        self._int()  # min reader R version
        if self.version >= 3:
            n = self._int()
            self._bytes(n)  # native encoding string, e.g. 'UTF-8'

    # --- flag decoding -------------------------------------------------------
    @staticmethod
    def _unpack_flags(flags):
        return {
            "type": flags & 0xFF,
            "levels": flags >> 12,
            "is_object": bool((flags >> 8) & 1),
            "has_attr": bool((flags >> 9) & 1),
            "has_tag": bool((flags >> 10) & 1),
        }

    # --- main dispatch -------------------------------------------------------
    def read_item(self):
        flags = self._int()
        f = self._unpack_flags(flags)
        t = f["type"]

        if t == NILVALUE_SXP or t == NILSXP:
            return None
        if t == GLOBALENV_SXP:
            return "<globalenv>"
        if t == EMPTYENV_SXP:
            return "<emptyenv>"
        if t == BASEENV_SXP:
            return "<baseenv>"
        if t == MISSINGARG_SXP:
            return "<missing>"
        if t == UNBOUNDVALUE_SXP:
            return "<unbound>"

        if t == REFSXP:
            idx = flags >> 8
            if idx == 0:
                idx = self._int()
            return self.refs[idx - 1]

        if t == SYMSXP:
            name = self.read_item()  # a CHARSXP
            self.refs.append(name)
            return name

        if t in (LISTSXP, LANGSXP, DOTSXP):
            return self._read_pairlist(f)

        if t == CHARSXP:
            n = self._int()
            if n == -1:
                return None  # NA_character_
            raw = self._bytes(n)
            try:
                return raw.decode("utf-8")
            except UnicodeDecodeError:
                return raw.decode("latin-1")

        if t in (LGLSXP, INTSXP):
            n = self._int()
            vals = list(struct.unpack_from(f">{n}i", self.buf, self.pos))
            self.pos += 4 * n
            if t == LGLSXP:
                vals = [None if v == NA_INTEGER else bool(v) for v in vals]
            else:
                vals = [None if v == NA_INTEGER else v for v in vals]
            return self._finish(vals, f)

        if t == REALSXP:
            n = self._int()
            vals = list(struct.unpack_from(f">{n}d", self.buf, self.pos))
            self.pos += 8 * n
            return self._finish(vals, f)

        if t == STRSXP:
            n = self._int()
            vals = [self.read_item() for _ in range(n)]
            return self._finish(vals, f)

        if t in (VECSXP, EXPRSXP):
            n = self._int()
            vals = [self.read_item() for _ in range(n)]
            return self._finish(vals, f)

        if t == RAWSXP:
            n = self._int()
            return self._bytes(n)

        if t == ALTREP_SXP:
            info = self.read_item()
            state = self.read_item()
            self.read_item()  # attributes
            return self._expand_altrep(info, state)

        raise NotImplementedError(f"SEXP type {t} (flags={flags}) at byte {self.pos}")

    # --- helpers -------------------------------------------------------------
    def _expand_altrep(self, info, state):
        """Handle compact integer sequences, the common ALTREP in row.names."""
        cls = info[0] if isinstance(info, (list, tuple)) and info else None
        if cls == "compact_intseq" and isinstance(state, list) and len(state) >= 3:
            n, start, step = int(state[0]), state[1], state[2]
            return [int(start + i * step) for i in range(n)]
        if cls == "deferred_string":
            return state
        return state

    def _read_pairlist(self, f):
        """Attributes and R pairlists: tag/value chain terminated by NULL."""
        out = {}
        order = []
        while True:
            tag = None
            if f["has_attr"]:
                self.read_item()  # attributes on the cell itself (rare)
            if f["has_tag"]:
                tag = self.read_item()
            value = self.read_item()
            key = tag if tag is not None else f"_{len(order)}"
            out[key] = value
            order.append(key)

            flags = self._int()
            f = self._unpack_flags(flags)
            if f["type"] in (NILVALUE_SXP, NILSXP):
                break
            if f["type"] != LISTSXP:
                # Non-list tail; step back so caller can parse it.
                self.pos -= 4
                break
        return out

    def _finish(self, vals, f):
        """Attach attributes if present, returning a dict wrapper when needed."""
        if not f["has_attr"]:
            return vals
        attrs = self.read_item()
        return {"_values": vals, "_attr": attrs}


def read_rds(path):
    with open(path, "rb") as fh:
        magic = fh.read(2)
    data = gzip.open(path, "rb").read() if magic == b"\x1f\x8b" else open(path, "rb").read()
    r = RDSReader(data)
    r.read_header()
    return r.read_item()


def to_dataframe(obj):
    """Rebuild a pandas.DataFrame from a parsed R data.frame, if possible."""
    import pandas as pd

    if not isinstance(obj, dict) or "_values" not in obj:
        raise TypeError("not an attributed R object")
    attrs = obj.get("_attr") or {}
    names = attrs.get("names")
    cols = obj["_values"]
    if names is None:
        raise TypeError("object has no names attribute")
    clean = []
    for c in cols:
        if isinstance(c, dict) and "_values" in c:
            cattr = c.get("_attr") or {}
            levels = cattr.get("levels")
            if levels is not None:  # factor -> labels
                lv = levels["_values"] if isinstance(levels, dict) else levels
                c = [None if v is None else lv[v - 1] for v in c["_values"]]
            else:
                c = c["_values"]
        clean.append(c)
    df = pd.DataFrame({n: pd.Series(c) for n, c in zip(names, clean)})
    rn = attrs.get("row.names")
    if isinstance(rn, list) and len(rn) == len(df) and all(isinstance(x, str) for x in rn):
        df.index = rn
    return df


if __name__ == "__main__":
    import sys

    obj = read_rds(sys.argv[1])

    def describe(o, indent=0, key=""):
        pad = "  " * indent
        if isinstance(o, dict) and "_values" in o:
            attrs = o.get("_attr") or {}
            cls = attrs.get("class")
            nm = attrs.get("names")
            print(f"{pad}{key} attributed<{cls}> n={len(o['_values'])} "
                  f"names={(nm[:8] if isinstance(nm, list) else nm)}")
            for i, v in enumerate(o["_values"][:6]):
                describe(v, indent + 1, key=f"[{nm[i] if isinstance(nm, list) and i < len(nm) else i}]")
        elif isinstance(o, list):
            print(f"{pad}{key} list n={len(o)} head={o[:4]}")
        elif isinstance(o, dict):
            print(f"{pad}{key} pairlist keys={list(o.keys())[:8]}")
        else:
            print(f"{pad}{key} {type(o).__name__} {str(o)[:60]}")

    describe(obj)
