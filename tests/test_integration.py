#!/usr/bin/env python3
"""Pruebas con GPU real: bins, bloques parciales, entrada corrupta y PNG RGB.

Solo biblioteca estandar de Python. Las mediciones de estas pruebas no sustituyen
la bitacora de rendimiento, pues usan una sola repeticion por kernel.
"""
import argparse
import csv
import json
from pathlib import Path
import random
import struct
import subprocess
import tempfile
import zlib


def read_png(path):
    data = path.read_bytes()
    assert data[:8] == b'\x89PNG\r\n\x1a\n'
    offset, compressed, size = 8, bytearray(), None
    while offset < len(data):
        count = struct.unpack_from('>I', data, offset)[0]
        kind = data[offset + 4:offset + 8]
        payload = data[offset + 8:offset + 8 + count]
        crc = struct.unpack_from('>I', data, offset + 8 + count)[0]
        assert zlib.crc32(kind + payload) == crc, 'CRC PNG incorrecto'
        if kind == b'IHDR':
            w, h, depth, color, compression, filtering, interlace = struct.unpack('>IIBBBBB', payload)
            assert (depth, color, compression, filtering, interlace) == (8, 2, 0, 0, 0)
            size = w, h
        if kind == b'IDAT':
            compressed.extend(payload)
        offset += count + 12
    raw = zlib.decompress(compressed)
    w, h = size
    assert len(raw) == h * (w * 3 + 1)
    assert all(raw[y * (w * 3 + 1)] == 0 for y in range(h))
    return size, b''.join(raw[y * (w * 3 + 1) + 1:(y + 1) * (w * 3 + 1)] for y in range(h))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--binary', default='build/hough')
    options = parser.parse_args()
    binary = str(Path(options.binary).resolve())
    rng = random.Random(310)
    cases = [
        ('negra', 17, 19, bytes(17 * 19), 'P5'),
        ('un_pixel', 1, 1, b'\xff', 'P5'),
        ('bloque_257', 257, 1, bytes([255]) * 257, 'P5'),
        ('densa_impar', 31, 17, bytes([255]) * (31 * 17), 'P5'),
        ('aleatoria', 29, 23, bytes(255 if rng.random() < .23 else 0 for _ in range(29 * 23)), 'P5'),
        ('vertical', 9, 33, bytes(255 if x == 4 else 0 for y in range(33) for x in range(9)), 'P2'),
        ('horizontal', 33, 9, bytes(255 if y == 4 else 0 for y in range(9) for x in range(33)), 'P5'),
        ('diagonal', 35, 35, bytes(255 if x == y else 0 for y in range(35) for x in range(35)), 'P5'),
        ('byte_espacio', 3, 1, bytes([10, 32, 35]), 'P5'),
    ]
    passed = 0
    with tempfile.TemporaryDirectory(prefix='hough-tests-') as directory:
        root = Path(directory)
        for name, w, h, pixels, magic in cases:
            pgm = root / (name + '.pgm')
            header = f'{magic}\n# caso sintetico\n{w} {h}\n255\n'.encode()
            pgm.write_bytes(header + (pixels if magic == 'P5' else ' '.join(map(str, pixels)).encode()))
            prefix = root / name
            run = subprocess.run([binary, str(pgm), '--runs', '1', '--output', str(prefix)], capture_output=True, text=True)
            assert run.returncode == 0, (name, run.stdout, run.stderr)
            metadata = json.loads(Path(str(prefix) + '_metadata.json').read_text())
            assert metadata['all_measured_runs_match_cpu']
            assert metadata['active_pixels'] == sum(value > 0 for value in pixels)
            assert metadata['blocks'] == (w * h + 255) // 256
            size, rgb = read_png(Path(str(prefix) + '_lines.png'))
            assert size == (w, h)
            lines = list(csv.DictReader(Path(str(prefix) + '_lines.csv').open()))
            if name == 'negra':
                assert not lines and not any(rgb)
            elif name in ('vertical', 'horizontal'):
                expected_theta = 0 if name == 'vertical' else 90
                assert any(int(line['theta_degrees']) == expected_theta and int(line['votes']) == 33 for line in lines)
                assert any(rgb[i] != rgb[i + 1] for i in range(0, len(rgb), 3)), 'Falta color'
            elif name == 'diagonal':
                # 45 grados cae entre bins 44/46: la discretizacion reparte los
                # 35 votos entre radios contiguos, sin un pico de 35 exacto.
                assert any(abs(int(line['theta_degrees']) - 45) <= 1 and int(line['votes']) >= 17 for line in lines)
            assert len(list(csv.DictReader(Path(str(prefix) + '_timings.csv').open()))) == 1
            print('PASS', name)
            passed += 1

        malformed = {
            'vacia': b'',
            'formato': b'P6\n1 1\n255\n\x00',
            'cero': b'P5\n0 1\n255\n',
            'negativa': b'P2\n-1 1\n255\n0',
            'truncada': b'P5\n2 2\n255\n\xff',
            'max16': b'P5\n1 1\n65535\n\x00\x00',
            'rango_p2': b'P2\n1 1\n10\n11',
            'rango_p5': b'P5\n1 1\n10\n\x0b',
            'overflow': b'P5\n2147483647 2\n255\n',
        }
        for name, data in malformed.items():
            pgm = root / (name + '.pgm')
            pgm.write_bytes(data)
            run = subprocess.run([binary, str(pgm), '--output', str(root / 'invalid')], capture_output=True, text=True)
            assert run.returncode != 0 and 'Error:' in run.stderr, name
            print('PASS rechazo', name)
            passed += 1
        for args in [[], ['--help'], ['missing.pgm'], ['missing.pgm', '--runs', '0'], ['missing.pgm', '--desconocida', '1']]:
            run = subprocess.run([binary] + args, capture_output=True, text=True)
            assert (run.returncode == 0) == (args == ['--help']), args
            passed += 1
    print(f'{passed} pruebas aprobadas.')


if __name__ == '__main__':
    main()
