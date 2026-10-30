"""Run finite checks sequentially, on one CPU, below one decimal gigabyte."""
if not __debug__: raise RuntimeError('Assertions must be enabled for these checks')
from pathlib import Path
import os, subprocess, sys, time, json
import psutil
HERE=Path(__file__).resolve().parent
TASKS=['critical/original.py','critical/independent.py','states/fixed_state_fms.py','states/independent.py','third/original.py','third/independent.py','prices/price_checks.py','prices/independent.py','robust/original.py','robust/airy.py','robust/independent.py','ratios/check_games.py','ratios/check_geometry.py','phase/check_phase.py']
BOOT="""import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS'): os.environ[k]='1'
import psutil
p=psutil.Process(); p.cpu_affinity([p.cpu_affinity()[0]])
import scipy.optimize, warnings, runpy, sys
old=scipy.optimize.linprog
def lp(*a,**kw):
 o=dict(kw.get('options') or {}); o['threads']=1; kw['options']=o; return old(*a,**kw)
scipy.optimize.linprog=lp
warnings.filterwarnings('ignore',message='Unrecognized options detected.*')
script=sys.argv.pop(1)
runpy.run_path(script,run_name='__main__')
"""
def main():
 records=[]
 env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUTF8='1')
 for rel in TASKS:
  script=HERE/rel; start=time.monotonic(); peak=0
  target=script.with_suffix('.log')
  with target.open('w',encoding='utf8',newline='\n') as log:
   p=subprocess.Popen([sys.executable,'-c',BOOT,str(script)],cwd=script.parent,env=env,stdout=log,stderr=subprocess.STDOUT)
   proc=psutil.Process(p.pid)
   while p.poll() is None:
    try:
     mem=proc.memory_info(); peak=max(peak,getattr(mem,'peak_wset',mem.rss))
     if peak>=1_000_000_000: p.kill(); raise RuntimeError('Memory limit exceeded: '+rel)
    except psutil.NoSuchProcess: pass
    time.sleep(.1)
  if p.returncode: raise RuntimeError('Check failed: '+rel+'; inspect '+target.name)
  if rel=='states/fixed_state_fms.py': (script.parent/'fixed_state_fms_output.txt').write_text(target.read_text(encoding='utf8'),encoding='utf8',newline='\n')
  records.append(dict(check=rel,status='PASS',seconds=round(time.monotonic()-start,3),peak_bytes=peak,cpu_count=1))
  print(rel+': PASS',flush=True)
 for file in HERE.rglob('*'):
  if file.is_file() and file.suffix in ('.py','.json','.txt','.log'):
   file.write_text(file.read_text(encoding='utf8'),encoding='utf8',newline='\n')
 (HERE/'summary.json').write_text(json.dumps(dict(status='PASS',checks=records),indent=2)+'\n',encoding='utf8',newline='\n')
if __name__=='__main__': main()
