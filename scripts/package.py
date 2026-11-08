#!/usr/bin/env python3
"""Empaqueta la revision confirmada: ZIP de entrega y bundle con historial."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import zipfile

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--output-dir', type=Path, default=root.parent)
options = parser.parse_args()
destination = options.output_dir.resolve()
destination.mkdir(parents=True, exist_ok=True)
status = subprocess.check_output(['git', 'status', '--porcelain'], cwd=root, text=True)
if status.strip():
    raise SystemExit('Confirme los cambios antes de empaquetar la revision')
archive = destination / 'PRY3_CUDA_entrega.zip'
bundle = destination / 'PRY3_CUDA_historial.bundle'
subprocess.run(['git', 'archive', '--format=zip', '--prefix=PRY3_CUDA/', '--output=' + str(archive), 'HEAD'], cwd=root, check=True)
subprocess.run(['git', 'bundle', 'create', str(bundle), 'main', 'HEAD'], cwd=root, check=True)
subprocess.run(['git', 'bundle', 'verify', str(bundle)], cwd=root, check=True)
with zipfile.ZipFile(archive) as zipped:
    assert zipped.testzip() is None
    for expected in ['src/hough.cu', 'docs/informe.pdf', 'results/runway_timings.csv', 'data/runway.pgm']:
        assert 'PRY3_CUDA/' + expected in zipped.namelist(), expected
    documents = [name for name in zipped.namelist() if name.startswith('PRY3_CUDA/docs/') and not name.endswith('/')]
    assert documents == ['PRY3_CUDA/docs/informe.pdf'], documents
manifest = destination / 'SHA256SUMS.txt'
manifest.write_text(''.join(hashlib.sha256(file.read_bytes()).hexdigest() + '  ' + file.name + '\n' for file in [archive, bundle]), encoding='utf-8')
print(archive)
print(bundle)
print(manifest)
