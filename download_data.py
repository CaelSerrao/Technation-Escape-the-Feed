"""Download the official MovieLens 32M archive and verify its exact identity."""
import argparse,hashlib,shutil,subprocess,urllib.request
from pathlib import Path
URL='https://files.grouplens.org/datasets/movielens/ml-32m.zip'
SHA256='e4a68655d7386b8f95f2f2424b2ff975dfdd15ffd59e0d864a14dca43e99d6ee'
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'data/ml-32m.zip');a=p.parse_args()
a.output.parent.mkdir(parents=True,exist_ok=True)
if not a.output.exists():
    partial=a.output.with_suffix('.zip.part')
    if shutil.which('curl'):
        subprocess.run(['curl','--fail','--location','--output',str(partial),URL],check=True)
    else:urllib.request.urlretrieve(URL,partial)
    actual=hashlib.sha256(partial.read_bytes()).hexdigest()
    if actual!=SHA256:raise RuntimeError(f'Download checksum mismatch: {actual}')
    partial.replace(a.output)
actual=hashlib.sha256(a.output.read_bytes()).hexdigest()
if actual!=SHA256:raise RuntimeError(f'Existing file checksum mismatch: {actual}')
print(f'Verified {a.output} ({a.output.stat().st_size:,} bytes)')
