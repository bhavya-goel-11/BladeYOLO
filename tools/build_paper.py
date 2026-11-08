"""Build the paper from one source: paper/paper.md -> paper/paper.pdf (via HTML and headless Chrome) and paper/paper.tex.
Figures and tables are numbered here, in order of definition: "![@DEF fig:name. Caption](file)" and ": @DEF tab:name. Caption"
define one; @fig:name and @tab:name refer to it.
  .venv/bin/python tools/build_paper.py
"""

import os
import re
import shutil
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = os.path.join(ROOT, "paper")
WORK = os.path.join(ROOT, "runs", "work", "paper_build")


def number(text):
    nums, seen = {}, {"fig": 0, "tab": 0}
    sec = 0
    for m in re.finditer(r"^# (.+)$", text, re.M):
        if "{-}" not in m.group(1):
            sec += 1
            k = re.search(r"\{#(sec:[\w-]+)\}", m.group(1))
            if k:
                nums[k.group(1)] = sec
    text = re.sub(r"§\[-@(sec:[\w-]+)\]", lambda m: f"§\u00a0{nums[m.group(1)]}", text)
    def define(m):
        kind, name = m.group(1), m.group(2)
        assert f"{kind}:{name}" not in nums, name
        seen[kind] += 1
        nums[f"{kind}:{name}"] = seen[kind]
        return f"{'Figure' if kind == 'fig' else 'Table'} {seen[kind]}."
    text = re.sub(r"@DEF (fig|tab):([\w-]+)\.", define, text)
    def refer(m):
        key = f"{m.group(1)}:{m.group(2)}"
        assert key in nums, f"undefined reference {key}"
        return f"{'Fig.' if m.group(1) == 'fig' else 'Table'} {nums[key]}"
    text = re.sub(r"@(fig|tab):([\w-]+)", refer, text)
    print("figures", seen["fig"], "tables", seen["tab"])
    return text


def audit_table():
    rows = ["| dataset | class | raw images | raw boxes | unique images | scenes | learnable boxes | median box px | purity |", "|---|---|---|---|---|---|---|---|---|"]
    names = {"wtbd": "WTBD", "ws": "Wind Surface", "wt": "WTBlade", "bj": "Beijing", "wtbs": "WTBs2025"}
    for line in open(os.path.join(ROOT, "runs", "analysis", "datasets", "report.md"), encoding="utf-8"):
        p = [s.strip() for s in line.split("|")]
        if len(p) > 11 and p[1] in names and p[2] != "class":
            rows.append(f"| {names[p[1]]} | {p[2]} | {p[3]} | {p[4]} | {p[6]} | {p[7]} | {p[8]} | {p[10]} | {p[11]} |")
    return "\n".join(rows)


def registry_table():
    lines = open(os.path.join(ROOT, "research_log.md"), encoding="utf-8").read().split("\n")
    i = lines.index("## 27. Run registry")
    while not lines[i].startswith("| Run folder"):
        i += 1
    j = i
    while lines[j].startswith("|"):
        j += 1
    return "\n".join(lines[i:j])


def post_html(path):
    h = open(path, encoding="utf-8").read()
    h = re.sub(r"<colgroup>.*?</colgroup>", "", h, flags=re.S)
    h = re.sub(r"(<table)( style=\"[^\"]*\")?", r"\1", h)
    h = re.sub(r"<table>.*?</table>", lambda m: m.group(0).replace("<table>", '<table class="long">', 1) if m.group(0).count("<tr") > 14 else m.group(0), h, flags=re.S)
    open(path, "w", encoding="utf-8").write(h)


def main():
    os.makedirs(WORK, exist_ok=True)
    md = os.path.join(WORK, "paper.md")
    src = open(os.path.join(P, "paper.md"), encoding="utf-8").read()
    src = src.replace("<!--AUDIT-->", audit_table()).replace("<!--REGISTRY-->", registry_table())
    open(md, "w", encoding="utf-8").write(number(src))
    common = ["pandoc", md, "--from", "markdown+pipe_tables+implicit_figures", "--citeproc",
              "--bibliography", os.path.join(P, "references.bib"), "--csl", os.path.join(P, "numeric.csl"),
              "--resource-path", P, "--metadata", "link-citations=true", "--number-sections"]
    html = os.path.join(P, "paper.html")
    subprocess.run(common + ["--standalone", "--mathml", "--css", "paper.css", "--to", "html5", "-o", html], check=True)
    post_html(html)
    subprocess.run(common + ["--standalone", "--to", "latex", "-V", "geometry:margin=2cm", "-V", "fontsize=10pt", "-o",
                             os.path.join(P, "paper.tex")], check=True)
    chrome = shutil.which("google-chrome")
    subprocess.run([chrome, "--headless=new", "--no-sandbox", "--disable-gpu", "--no-pdf-header-footer",
                    f"--user-data-dir={os.path.join(WORK, 'chrome')}", f"--print-to-pdf={os.path.join(P, 'paper.pdf')}",
                    "file://" + html], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("built", os.path.join(P, "paper.pdf"))


if __name__ == "__main__":
    main()
