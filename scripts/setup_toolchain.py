#!/usr/bin/env python3
"""CUDA 13.1 aislado para Linux x86_64 y cabeceras glibc 2.39.

No modifica el driver ni las bibliotecas del sistema. Requiere g++-14, make,
curl/python3 y una GPU visible en WSL2. Ejecutar desde la raiz del repositorio.
Descargas oficiales verificadas por SHA256 de sus manifiestos.
"""
import concurrent.futures
import gzip
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import urllib.request


def fetch(url, destination, sha256=None):
    if not destination.exists():
        urllib.request.urlretrieve(url, destination)
    if sha256 and hashlib.sha256(destination.read_bytes()).hexdigest() != sha256:
        raise RuntimeError('SHA256 incorrecto: ' + destination.name)


def main():
    if platform.system() != 'Linux' or platform.machine() != 'x86_64':
        raise SystemExit('Ejecutar en Linux x86_64 o Ubuntu bajo WSL2')
    root = Path('.toolchain').resolve()
    cache = root / 'downloads'
    cache.mkdir(parents=True, exist_ok=True)
    cuda = root / 'cuda'
    cuda.mkdir(exist_ok=True)
    base = 'https://developer.download.nvidia.com/compute/cuda/redist/'
    manifest = json.load(urllib.request.urlopen(base + 'redistrib_13.1.1.json'))

    def component(name):
        info = manifest[name]['linux-x86_64']
        archive = cache / Path(info['relative_path']).name
        fetch(base + info['relative_path'], archive, info['sha256'])
        return archive

    names = ['cuda_nvcc', 'cuda_cudart', 'cuda_cccl', 'cuda_crt', 'libnvvm', 'cuda_sanitizer_api']
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        archives = list(pool.map(component, names))
    for archive in archives:
        # GNU tar elimina la carpeta del componente y conserva permisos/enlaces.
        subprocess.run(['tar', '-xJf', str(archive), '-C', str(cuda), '--strip-components=1'], check=True)

    sysroot = root / 'sysroot'
    sysroot.mkdir(exist_ok=True)
    index = gzip.decompress(urllib.request.urlopen('https://archive.ubuntu.com/ubuntu/dists/noble-updates/main/binary-amd64/Packages.gz').read()).decode()
    for block in index.split('\n\n'):
        info = dict(line.split(': ', 1) for line in block.splitlines() if ': ' in line and not line.startswith(' '))
        if info.get('Package') in ['libc6', 'libc6-dev', 'linux-libc-dev']:
            archive = cache / Path(info['Filename']).name
            fetch('https://archive.ubuntu.com/ubuntu/' + info['Filename'], archive, info['SHA256'])
            subprocess.run(['dpkg-deb', '-x', str(archive), str(sysroot)], check=True)
    # Rutas relativas al script generado, por lo que el repositorio se puede mover.
    env = '''#!/usr/bin/env bash
_pry3_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="$_pry3_root/cuda/bin:$PATH"
export LD_LIBRARY_PATH="$_pry3_root/cuda/lib:/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"
export HOST_CXX=g++-14
export EXTRA_NVCCFLAGS="-L$_pry3_root/cuda/lib -Xcompiler=--sysroot=$_pry3_root/sysroot"
unset _pry3_root
'''
    (root / 'env.sh').write_text(env)
    print('Listo. Ejecutar: source .toolchain/env.sh && make && make test')


if __name__ == '__main__':
    main()
