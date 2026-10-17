"""Step 13 - Maintainability Index (MI) before/after a refactoring, written into the cases CSV.

Technical debt is NOT computed here: it needs SonarQube/SonarCloud (SQALE remediation time, `sqale_index`, minutes).
Copy those numbers into the debt_before / debt_after columns by hand (see README_new_cases.md).

MI definition (Visual Studio variant, 0-100, higher = more maintainable):
    MI = max(0, (171 - 5.2 ln(V) - 0.23 G - 16.2 ln(LOC)) * 100 / 171)
  V   = Halstead volume = (N1+N2) * log2(n1+n2) from the Java token stream
        (operators = keywords, operators, separators; operands = identifiers and literals),
  G   = cyclomatic complexity = 1 + #(if, for, while, case, catch, &&, ||, ?),
  LOC = number of source lines that contain code (blank and comment-only lines are not counted).
Columns written:
  mi_before / mi_after              MI of the target method (level = method) or LOC-weighted MI of the methods of the
                                    target class (level = class);
  mi_class_before / mi_class_after  LOC-weighted MI of all methods/constructors declared directly in the containing
                                    class (shows whether the extracted helper made the class better overall).
This is an approximation of tool-specific MI values; state the definition in the paper.

Usage:  python 13_mi_metrics.py --cases cases_todo.csv [--workdir work_ba]    (needs: pip install javalang)
"""
import argparse, importlib.util, math, os, re, sys
import pandas as pd
from javalang import tokenizer as T

DECISION_KW = {"if", "for", "while", "case", "catch"}
DECISION_OP = {"&&", "||", "?"}
LITERALS = (T.Integer, T.FloatingPoint, T.String, T.Character, T.Boolean, T.Null, T.DecimalInteger, T.OctalInteger,
            T.BinaryInteger, T.HexInteger, T.DecimalFloatingPoint, T.HexFloatingPoint)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m); return m


def mi_of(tokens):
    ops, opd = [], []
    for t in tokens:
        if isinstance(t, (T.Keyword, T.Operator, T.Separator, T.Modifier, T.BasicType)):
            ops.append(t.value)
        elif isinstance(t, (T.Identifier,) + LITERALS):
            opd.append(t.value)
    n = len(set(ops)) + len(set(opd)); length = len(ops) + len(opd)
    vol = max(1.0, length * math.log2(max(n, 2)))
    g = 1 + sum(1 for t in tokens if (isinstance(t, T.Keyword) and t.value in DECISION_KW)
                or (isinstance(t, T.Operator) and t.value in DECISION_OP))
    loc = max(1, len({t.position[0] for t in tokens}))
    mi = max(0.0, (171 - 5.2 * math.log(vol) - 0.23 * g - 16.2 * math.log(loc)) * 100 / 171)
    return mi, loc


def _match(tokens, i, open_v, close_v):
    """index of the token closing the bracket opened at tokens[i]."""
    depth = 0
    for j in range(i, len(tokens)):
        if isinstance(tokens[j], T.Separator) or isinstance(tokens[j], T.Operator):
            if tokens[j].value == open_v:
                depth += 1
            elif tokens[j].value == close_v:
                depth -= 1
                if depth == 0:
                    return j
    return None


def simple(t):
    return re.sub(r"<.*>", "", t).split(".")[-1].replace("...", "[]").strip()


def find_class_body(tokens, simple_cls):
    for i, t in enumerate(tokens[:-1]):
        if isinstance(t, T.Keyword) and t.value in ("class", "interface", "enum") and tokens[i + 1].value == simple_cls:
            j = next((k for k in range(i, len(tokens)) if tokens[k].value == "{" and isinstance(tokens[k], T.Separator)), None)
            if j is not None:
                return j, _match(tokens, j, "{", "}")
    return None


def param_types(tokens, lp, rp):
    """simple type names of the parameters between tokens[lp]='(' and tokens[rp]=')'."""
    out, cur, depth = [], [], 0
    for t in tokens[lp + 1:rp]:
        v = t.value
        if v in ("<", "(", "["):
            depth += 1 if v != "[" else 0
        if v in (">", ")"):
            depth -= 1
        if v == "," and depth <= 0:
            out.append(cur); cur = []
        else:
            cur.append(t)
    if cur:
        out.append(cur)
    res = []
    for p in out:
        ids = [t for t in p if isinstance(t, (T.Identifier, T.BasicType))]
        dims = sum(1 for t in p if t.value == "[") + (1 if any(t.value == "..." for t in p) else 0)
        gen, typ = 0, None
        for t in p[:-1]:                      # last token is the parameter name
            if t.value == "<":
                gen += 1
            elif t.value == ">":
                gen -= 1
            elif gen == 0 and isinstance(t, (T.Identifier, T.BasicType)):
                typ = t.value
        if typ is not None:
            res.append(typ + "[]" * dims)
    return res


def class_members(tokens, cls_name):
    """[(name, params, slice_tokens, start_line)] of the methods/constructors declared directly in the class."""
    cb = find_class_body(tokens, cls_name)
    if not cb or cb[1] is None:
        raise RuntimeError(f"class {cls_name} not found")
    lo, hi = cb
    members, i, seg = [], lo + 1, lo + 1
    while i < hi:
        t = tokens[i]
        if isinstance(t, T.Separator) and t.value == ";":
            seg = i + 1; i += 1; continue
        if isinstance(t, T.Separator) and t.value == "{":          # initializer block / enum constant body
            i = _match(tokens, i, "{", "}") + 1; seg = i; continue
        if isinstance(t, T.Operator) and t.value == "=":           # field initialiser: skip to its ';'
            d = 0
            while i < hi and not (tokens[i].value == ";" and d == 0):
                if tokens[i].value in "({[" and not isinstance(tokens[i], T.String):
                    d += 1
                elif tokens[i].value in ")}]" and not isinstance(tokens[i], T.String):
                    d -= 1
                i += 1
            seg = i + 1; i += 1; continue
        if isinstance(t, T.Annotation):                             # @Foo or @Foo(...)
            i += 2
            if i < hi and tokens[i].value == "(":
                i = _match(tokens, i, "(", ")") + 1
            continue
        if isinstance(t, T.Identifier) and i + 1 < hi and tokens[i + 1].value == "(":
            rp = _match(tokens, i + 1, "(", ")")
            j = rp + 1
            while j < hi and tokens[j].value not in ("{", ";"):
                j += 1                                              # throws clause / default value
            if j < hi and tokens[j].value == "{":
                end = _match(tokens, j, "{", "}")
                members.append((t.value, param_types(tokens, i + 1, rp), tokens[seg:end + 1], tokens[seg].position[0]))
                i = end + 1; seg = i; continue
            seg = j + 1; i = j + 1; continue                        # abstract / interface method
        i += 1
    return members


def weighted(ms):
    w = sum(l for _, _, l in ms)
    return sum(m * l for _, m, l in ms) / w if w else float("nan")


def measure(path, level, code_name, method_name, params, start_line):
    src = open(path, encoding="utf-8", errors="replace").read()
    tokens = list(T.tokenize(src))
    simple_cls = code_name.split("#")[0].split(".")[-1].split("$")[-1]
    members = class_members(tokens, simple_cls)
    scored = [(n, p, mi_of(sl), sl) for n, p, sl, _ in members]
    mi_class = weighted([(n, mi, loc) for n, p, (mi, loc), sl in scored])
    if level == "class":
        return mi_class, mi_class
    cands = [(n, p, mi_loc, sl) for n, p, mi_loc, sl in scored if n == method_name]
    if params:
        want = [simple(x) for x in params]
        f = [c for c in cands if [simple(x) for x in c[1]] == want]
        cands = f or cands
    if not cands:
        raise RuntimeError(f"method {method_name} not found in {simple_cls}")
    best = min(cands, key=lambda c: abs(c[3][0].position[0] - int(start_line or 0)))
    return best[2][0], mi_class


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="cases_todo.csv"); ap.add_argument("--workdir", default="work_ba")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    s02 = load_module("s02", os.path.join(os.path.dirname(os.path.abspath(__file__)), "02_fetch_and_run_ck.py"))
    df = pd.read_csv(a.cases, dtype=str, keep_default_na=False)
    for c in ("mi_before", "mi_after", "mi_class_before", "mi_class_after", "debt_before", "debt_after"):
        if c not in df.columns:
            df[c] = ""
    for i, r in df.iterrows():
        if not r.get("commit_after") or str(r.get("status", "")).lower().startswith("skip"):
            continue
        params = []
        if " " in r["code_name"]:
            params = [p for p in r["code_name"].split(" ", 1)[1].split("|") if p]
        try:
            res = {}
            for tag, url, sha in (("before", r["repo_url"], r["commit_before"]),
                                  ("after", r.get("repo_url_after") or r["repo_url"], r["commit_after"])):
                dest = os.path.join(a.workdir, s02.tag(url, sha))
                if not os.path.exists(dest):
                    ok, err = s02.fetch(url, sha, dest)
                    if not ok:
                        raise RuntimeError("fetch failed: " + str(err))
                res[tag] = measure(os.path.join(dest, r["path"].lstrip("/\\")), r["level"], r["code_name"].split(" ")[0],
                                   r.get("method_name", ""), params, r.get("start_line") or 0)
            for col, v in (("mi_before", res["before"][0]), ("mi_class_before", res["before"][1]),
                           ("mi_after", res["after"][0]), ("mi_class_after", res["after"][1])):
                df.at[i, col] = f"{v:.2f}"
            print(f'{r["case_id"]}: MI {res["before"][0]:.2f} -> {res["after"][0]:.2f} | class MI {res["before"][1]:.2f} -> {res["after"][1]:.2f}')
        except Exception as e:
            print(f'{r["case_id"]}: MI FAILED {e}')
    df.to_csv(a.cases, index=False, lineterminator="\n")


if __name__ == "__main__":
    main()
