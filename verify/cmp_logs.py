import re, sys
def parse(path):
    out, m, h = {}, None, None
    for line in open(path, encoding="utf-8", errors="replace"):
        g = re.match(r"######## (m\d)$", line.strip())
        if g: m = g.group(1); h = None; continue
        if line.startswith("== "):
            h = line.strip(); continue
        g = re.match(r"\s*(\S+/\S+)\s+max (\S+) rms (\S+)", line)
        if g and h and m and "of vs of60" not in h:
            out[(m, h, g.group(1))] = (float(g.group(2)), float(g.group(3)))
    return out
a, c = parse(sys.argv[1]), parse(sys.argv[2])
print("%-4s %-55s %-28s %11s %11s" % ("M", "comparison", "field", "max before", "max after"))
worst = 0
for k in sorted(c):
    if k not in a: continue
    ma, mc = a[k][0], c[k][0]
    flag = " <--" if mc > 3*ma and mc > 1e-9 else ""
    if flag or k[2].endswith("/T") or k[2].endswith("/U"):
        print("%-4s %-55s %-28s %11.3e %11.3e%s" % (k[0], k[1][:55], k[2], ma, mc, flag))
