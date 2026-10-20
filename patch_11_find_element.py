"""Sua loi chon nham overload trong 11_before_after.py (ca long_method_09).

Chay trong thu muc xsmell_mlcq:   python patch_11_find_element.py
- Sao luu file goc thanh 11_before_after.py.bak
- Thay ham find_element: khi co nhieu overload cung ten, chon dong co start line gan start_line da chon nhat
  (BEFORE) va chon dung chu ky (signature) do (AFTER).
- Sua cho goi find_element de truyen start_line va signature.
Neu khong tim thay doan can thay, script dung lai va khong sua gi.
"""
import re, shutil, sys

P = "11_before_after.py"
src = open(P, encoding="utf-8").read()
if "signature=None" in src:
    sys.exit("File da duoc va roi (co 'signature=None'). Khong lam gi them.")

old_def = re.search(r"def find_element\(ck, level, path, code_name, method_name, step3\):\n(?:    .*\n)+?    return None if c\.empty else c\.iloc\[0\]\n", src)
if not old_def:
    sys.exit("Khong tim thay ham find_element nhu mong doi. Gui noi dung ham nay cho minh.")

new_def = '''def find_element(ck, level, path, code_name, method_name, step3, start_line=0, signature=None):
    """Locate the refactored element in a CK table.
    class : by name.
    method: by file and name; if several overloads share the name, use the exact CK signature when it is given
            (the AFTER side uses the signature found BEFORE), otherwise the row whose start line is closest to
            the selected start line."""
    row = {"path": path, "code_name": code_name, "start_line": 0, "end_line": 10 ** 9}
    if level == "class":
        return step3.match_class(ck, row)
    c = ck[ck["_file"].map(lambda x: step3.same_file(x, path))]
    if method_name:
        c = c[c["method"].astype(str).str.startswith(str(method_name) + "/")]
    if c.empty:
        return None
    if signature is not None:
        exact = c[c["method"].astype(str) == str(signature)]
        if len(exact):
            c = exact
    if len(c) > 1 and "line" in c.columns:
        dist = (pd.to_numeric(c["line"], errors="coerce") - float(start_line or 0)).abs()
        c = c.iloc[dist.argsort(kind="stable")]
    return c.iloc[0]
'''
src = src.replace(old_def.group(0), new_def)

old_call = 'el = find_element(ck, c.level, c.path, c.code_name, getattr(c, "method_name", None), s03)'
new_call = ('sig = sides["before"]["method"] if (tag == "after" and kind == "method" and "before" in sides) else None\n'
            '                el = find_element(ck, c.level, c.path, c.code_name, getattr(c, "method_name", None), s03,\n'
            '                                  start_line=getattr(c, "start_line", 0), signature=sig)')
if old_call not in src:
    sys.exit("Khong tim thay dong goi find_element. Gui dong 'el = find_element(...)' cho minh.")
src = src.replace(old_call, new_call)

shutil.copy(P, P + ".bak")
open(P, "w", encoding="utf-8").write(src)
print("Da va xong. Ban sao luu: 11_before_after.py.bak")
