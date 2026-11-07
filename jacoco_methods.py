import sys, re, xml.etree.ElementTree as ET
xml, cls_re = sys.argv[1], sys.argv[2]
m_re = sys.argv[3] if len(sys.argv) > 3 else "."
root = ET.parse(xml).getroot()
for c in root.iter("class"):
    name = c.get("name").replace("/", ".")
    if not re.search(cls_re, name):
        continue
    for m in c.findall("method"):
        if not re.search(m_re, m.get("name")):
            continue
        d = {k.get("type"): (int(k.get("missed")), int(k.get("covered"))) for k in m.findall("counter")}
        ln = d.get("LINE", (0, 0)); br = d.get("BRANCH", (0, 0))
        print("%-70s lines %3d/%3d  branches %3d/%3d" % (name.split(".")[-1] + "." + m.get("name"), ln[1], ln[0] + ln[1], br[1], br[0] + br[1]))
