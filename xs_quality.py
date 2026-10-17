"""Maintainability Index (MI) and a PMD-based technical-debt estimate for one class or method of a Java file.
Used by 11_before_after.py (Contribution 3). Needs:  pip install javalang   (MI)   and a PMD 7 install (debt, optional).

DEFINITIONS (state them in the paper):
  MI    = max(0, (171 - 5.2*ln(HV) - 0.23*CC - 16.2*ln(LOC)) * 100/171)        [Oman & Hagemeister; Visual Studio scaling 0-100]
          HV  = Halstead volume = N*log2(n) of the element's own tokens (javalang tokenizer);
                operands = identifiers + literals, operators = keywords, modifiers, basic types, operators, separators, annotations;
          CC  = CK WMC of the element;  LOC = CK LOC of the element.   Higher = more maintainable.
          For a class the same formula is applied to the whole class (not an average over its methods), so MI values are only
          comparable between the before and after version of the SAME element, not between a method and a class.
  debt  = sum of remediation minutes of the PMD (quickstart ruleset) violations whose line lies inside the element.
          minutes per PMD priority: DEBT_MINUTES below. This is a SQALE-style PROXY, NOT SonarQube's sqale_index: the minutes are
          an assumption and PMD runs without the project classpath (some type-based rules are less precise).
          If SonarQube numbers are required, fill the columns debt_before / debt_after of the cases file by hand; they are kept.
  issues = number of those PMD violations.  Lower = better for both.
"""
import csv
import math
import os
import re
import subprocess
import tempfile

import numpy as np

DEBT_MINUTES = {1: 60, 2: 30, 3: 20, 4: 10, 5: 5}      # assumption (priority 1 = most severe)


def _tokens(src):
    import javalang
    return list(javalang.tokenizer.tokenize(src))


def _is(tok, cls_name):
    return type(tok).__name__ == cls_name


def _val(tok):
    return tok.value


def _match_close(toks, i_open, open_v="{", close_v="}"):
    depth = 0
    for j in range(i_open, len(toks)):
        t = toks[j]
        if _is(t, "Separator"):
            if t.value == open_v:
                depth += 1
            elif t.value == close_v:
                depth -= 1
                if depth == 0:
                    return j
    return None


def locate_method(toks, name, start_line):
    """(i_start, i_end) token range of the method declaration called `name` whose declaration starts near `start_line`."""
    best = None
    for i, t in enumerate(toks[:-1]):
        if not (_is(t, "Identifier") and t.value == name and _is(toks[i + 1], "Separator") and toks[i + 1].value == "("):
            continue
        close_par = _match_close(toks, i + 1, "(", ")")
        if close_par is None:
            continue
        k = close_par + 1
        while k < len(toks) and not (_is(toks[k], "Separator") and toks[k].value in ("{", ";")):
            k += 1
        if k >= len(toks) or toks[k].value != "{":
            continue
        # a declaration is preceded by its return type / modifier (not by '.', 'new', '=', '(' ...)
        prev = toks[i - 1] if i else None
        ok_prev = prev is not None and (
            _is(prev, "Identifier") or _is(prev, "BasicType") or _is(prev, "Modifier")
            or (_is(prev, "Keyword") and prev.value == "void")
            or (_is(prev, "Operator") and prev.value in (">", ">>", ">>>"))
            or (_is(prev, "Separator") and prev.value == "]"))
        if not ok_prev:
            continue
        dist = abs(t.position[0] - start_line)
        if best is None or dist < best[0]:
            best = (dist, i, k)
    if best is None:
        return None
    _, i, k = best
    end = _match_close(toks, k)
    if end is None:
        return None
    s = i
    while s > 0 and toks[s - 1].position[0] >= start_line and not (_is(toks[s - 1], "Separator") and toks[s - 1].value in ";{}"):
        s -= 1
    return s, end


def locate_class(toks, simple_name):
    for i, t in enumerate(toks[:-1]):
        if _is(t, "Keyword") and t.value in ("class", "interface", "enum") and _is(toks[i + 1], "Identifier") \
                and toks[i + 1].value == simple_name:
            j = i + 1
            while j < len(toks) and not (_is(toks[j], "Separator") and toks[j].value == "{"):
                j += 1
            end = _match_close(toks, j) if j < len(toks) else None
            if end is None:
                return None
            s = i
            while s > 0 and (_is(toks[s - 1], "Modifier") or _is(toks[s - 1], "Annotation")
                             or (_is(toks[s - 1], "Identifier") and s > 1 and _is(toks[s - 2], "Annotation"))):
                s -= 1
            return s, end
    return None


def halstead_volume(tk):
    ops, opd = [], []
    for t in tk:
        n = type(t).__name__
        if n == "Identifier" or n in ("Integer", "DecimalInteger", "OctalInteger", "BinaryInteger", "HexInteger",
                                      "FloatingPoint", "DecimalFloatingPoint", "HexFloatingPoint", "String", "Character",
                                      "Boolean", "Null", "Literal"):
            opd.append(t.value)
        else:
            ops.append(t.value)
    n_distinct = len(set(ops)) + len(set(opd))
    big_n = len(ops) + len(opd)
    return big_n * math.log2(n_distinct) if n_distinct > 1 else float("nan")


def maintainability_index(hv, cc, loc):
    if any(v is None or (isinstance(v, float) and math.isnan(v)) or v <= 0 for v in (hv, loc)) or cc is None or (isinstance(cc, float) and math.isnan(cc)):
        return float("nan")
    return max(0.0, (171 - 5.2 * math.log(hv) - 0.23 * float(cc) - 16.2 * math.log(float(loc))) * 100 / 171)


def element_range(path, level, code_name, method_name, start_line):
    """-> (tokens_of_element, first_line, last_line) or raises."""
    src = open(path, encoding="utf-8", errors="replace").read()
    toks = _tokens(src)
    if level == "class":
        simple = re.split(r"[.$]", str(code_name))[-1]
        rng = locate_class(toks, simple)
    else:
        rng = locate_method(toks, method_name, int(start_line))
    if rng is None:
        raise RuntimeError("element not located in the token stream")
    s, e = rng
    return toks[s:e + 1], toks[s].position[0], toks[e].position[0]


def mi_for(path, level, code_name, method_name, start_line, cc, loc):
    tk, l0, l1 = element_range(path, level, code_name, method_name, start_line)
    return maintainability_index(halstead_volume(tk), cc, loc), l0, l1


def pmd_issues(pmd_cmd, path, first_line, last_line):
    """-> (n_issues, debt_minutes) of PMD violations inside [first_line, last_line] of the file."""
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "pmd.csv")
        cmd = [pmd_cmd, "check", "-d", path, "-R", "rulesets/java/quickstart.xml", "-f", "csv", "-r", out,
               "--no-cache", "--no-progress", "--no-fail-on-violation"]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        if not os.path.exists(out):
            raise RuntimeError("PMD produced no report: " + (r.stderr or r.stdout)[-300:])
        n, minutes = 0, 0
        with open(out, encoding="utf-8", errors="replace", newline="") as f:
            for row in csv.DictReader(f):
                try:
                    line, pr = int(row["Line"]), int(row["Priority"])
                except (KeyError, ValueError):
                    continue
                if first_line <= line <= last_line:
                    n += 1
                    minutes += DEBT_MINUTES.get(pr, DEBT_MINUTES[3])
        return n, minutes


if __name__ == "__main__":      # quick self-check of the maths
    assert abs(maintainability_index(1000, 10, 50) - (171 - 5.2 * math.log(1000) - 2.3 - 16.2 * math.log(50)) * 100 / 171) < 1e-9
    assert maintainability_index(1, 1, 1) == (171 - 0.23) * 100 / 171
    print("xs_quality self-check OK")
