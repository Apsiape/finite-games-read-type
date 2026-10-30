"""Build the manuscript strictly in a separate directory."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import os

parser=argparse.ArgumentParser()
parser.add_argument('--output-dir',required=True)
parser.add_argument('--source-dir',default=str(Path(__file__).resolve().parent))
args=parser.parse_args()
source=Path(args.source_dir).resolve()
out=Path(args.output_dir).resolve()
if out==source or source in out.parents:raise ValueError('Choose a build directory outside the source tree')
out.mkdir(parents=True,exist_ok=True)
for path in sorted(source.glob('*.tex'))+[source/'references.bib']:
    shutil.copyfile(path,out/path.name)
os.environ['SOURCE_DATE_EPOCH']='1793232000'
os.environ['FORCE_SOURCE_DATE']='1'
latex=['pdflatex','-no-shell-escape','-interaction=nonstopmode','-halt-on-error','-file-line-error','main.tex']
commands=[latex,['bibtex','main'],latex,latex,latex]
for index,command in enumerate(commands,1):
    done=subprocess.run(command,cwd=out,capture_output=True,text=True,
                        encoding='utf-8',errors='replace',timeout=180)
    (out/('pass-%d.txt'%index)).write_text(done.stdout+done.stderr,encoding='utf-8',newline='\n')
    if done.returncode:raise RuntimeError('Build failed; inspect pass log')
log=(out/'main.log').read_text(encoding='utf-8',errors='replace')
blg=(out/'main.blg').read_text(encoding='utf-8',errors='replace')
issues=re.findall(r'^.*Warning[^\n]*$|^(?:Overfull|Underfull) \\[hv]box[^\n]*|'
                  r'^.*undefined references.*$',log,re.M)
bib_issues=re.findall(r'^Warning--.*$|^I couldn.t open.*$',blg,re.M)
if 'Missing character:' in log:issues.append('Missing character')
if issues or bib_issues:
    print(json.dumps(dict(status='FAIL',latex=issues,bibtex=bib_issues),indent=2))
    raise RuntimeError('Strict log audit failed')
summary=dict(status='PASS',latex_warnings=0,bibtex_warnings=0,
             undefined_references=0,overfull_boxes=0,underfull_boxes=0,
             pages=int(re.search(r'Output written on main.pdf \((\d+) pages',log).group(1)),
             pdf_sha256=hashlib.sha256((out/'main.pdf').read_bytes()).hexdigest())
(out/'build-summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8',newline='\n')
print(json.dumps(summary))
