"""Check source consistency, metadata, line endings and ancillary hashes."""
import json, re, subprocess, sys
from pathlib import Path
here=Path(__file__).resolve().parent
main=(here/'main.tex').read_text(encoding='utf8')
a=' '.join(re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}',main,re.S).group(1).split())
assert a==(here/'abstract.txt').read_text(encoding='utf8').strip()
assert len(a)<=1920
sections=re.findall(r'\\input\{([^}]+)\}',main)
assert all((here/(n+'.tex')).is_file() for n in sections)
t='\n'.join(p.read_text(encoding='utf8') for p in here.glob('*.tex'))
labels=re.findall(r'\\label\{([^}]+)\}',t)
assert len(labels)==len(set(labels))
assert set(re.findall(r'\\(?:eqref|ref)\{([^}]+)\}',t))<=set(labels)
bib=(here/'references.bib').read_text(encoding='utf8')
keys=set(re.findall(r'@\w+\{([^,]+),',bib))
cites=[]
for group in re.findall(r'\\cite(?:\[[^]]*\])?\{([^}]+)\}',t): cites+=group.split(',')
assert set(cites)==keys
for p in here.rglob('*'):
 if p.is_file() and p.suffix not in ('.pdf','.png','.zip'):
  assert b'\r' not in p.read_bytes(),p.name
subprocess.run([sys.executable,str(here/'anc/verify_hashes.py')],check=True)
root=here.parent
if (root/'.zenodo.json').exists():
 assert json.loads((root/'.zenodo.json').read_text(encoding='utf8'))['description']==a
 for name in ('README.md','README.txt','CITATION.cff'): assert a in (root/name).read_text(encoding='utf8')
assert a in (here/'README.txt').read_text(encoding='utf8')
print(json.dumps(dict(status='PASS',abstract_characters=len(a),sections=len(sections),labels=len(labels),bibliography_entries=len(keys),line_endings='LF')))
