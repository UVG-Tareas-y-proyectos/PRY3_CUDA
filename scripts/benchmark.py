#!/usr/bin/env python3
"""Registra datos reales y resume CUDA events sin dependencias de terceros."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import statistics
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--binary', default='build/hough')
    parser.add_argument('--image', default='data/runway.pgm')
    parser.add_argument('--output', default='results/runway')
    parser.add_argument('--runs', type=int, default=30)
    options = parser.parse_args()
    if not 10 <= options.runs <= 10000:
        parser.error('La bitacora requiere al menos 10 repeticiones y admite hasta 10000')
    prefix = Path(options.output)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    command = [str(Path(options.binary).resolve()), options.image, '--runs', str(options.runs), '--output', options.output]
    started = datetime.now(timezone.utc).isoformat()
    run = subprocess.run(command, capture_output=True, text=True)
    Path(str(prefix) + '_execution.txt').write_text(run.stdout + run.stderr, encoding='utf-8')
    print(run.stdout, end='')
    if run.returncode:
        raise SystemExit(run.stderr or run.returncode)
    with Path(str(prefix) + '_timings.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    summary = {}
    for column in ['global_ms', 'constant_ms', 'shared_ms']:
        values = [float(row[column]) for row in rows]
        summary[column] = dict(n=len(values), mean=statistics.mean(values), median=statistics.median(values),
                               stdev=statistics.stdev(values), minimum=min(values), maximum=max(values))
    summary['global_over_constant'] = summary['global_ms']['mean'] / summary['constant_ms']['mean']
    summary['global_over_shared'] = summary['global_ms']['mean'] / summary['shared_ms']['mean']
    metadata = json.loads(Path(str(prefix) + '_metadata.json').read_text())
    metadata.update(measurement_started_utc=started, os=platform.platform(), cpu=platform.processor(),
                    input_sha256=hashlib.sha256(Path(options.image).read_bytes()).hexdigest(),
                    binary_sha256=hashlib.sha256(Path(options.binary).read_bytes()).hexdigest(),
                    variant_order=['global', 'constant', 'shared'],
                    scope='Solo kernel; excluye memset, copias, validacion CPU, PNG y E/S')
    for name, args in [('nvcc', ['--version']), ('nvidia-smi', ['--query-gpu=name,driver_version,power.limit', '--format=csv,noheader'])]:
        executable = shutil.which(name)
        if executable:
            query = subprocess.run([executable] + args, capture_output=True, text=True)
            metadata[name] = query.stdout.strip() if query.returncode == 0 else query.stderr.strip()
    Path(str(prefix) + '_metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    Path(str(prefix) + '_summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
