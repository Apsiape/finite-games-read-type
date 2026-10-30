"""Verify the shipped ancillary manifest without changing any file."""
if not __debug__: raise RuntimeError('Assertions must be enabled for these checks')
import hashlib
from pathlib import Path

root=Path(__file__).resolve().parent
count=0
for line in (root/'SHA256SUMS').read_text(encoding='utf-8').splitlines():
    expected,name=line.split('  ',1)
    path=root/name
    assert path.resolve().is_relative_to(root.resolve()) and path.is_file()
    assert hashlib.sha256(path.read_bytes()).hexdigest()==expected,name+': checksum mismatch'
    count+=1
actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and p.name!='SHA256SUMS'}
listed={line.split('  ',1)[1] for line in (root/'SHA256SUMS').read_text().splitlines()}
assert actual==listed,'Manifest coverage mismatch'
print('ALL ANCILLARY CHECKSUMS PASS ('+str(count)+' files)')
