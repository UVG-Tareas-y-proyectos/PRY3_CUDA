# Transformada de Hough en CUDA

Proyecto de Computación Paralela y Distribuida, Universidad del Valle de
Guatemala. Entrega: **11 de noviembre de 2026**.

**Integrantes:** Ihan Marroquin (23108), Diego Patzan (23525) y Luis Gonzalez
(23353). El programa compara memoria global, global + constante y global +
constante + compartida. Cada ejecución medida exige coincidencia exacta de sus
9000 bins con una referencia CPU. La salida incluye un PNG con rectas a color.

## Entregables

- [Informe PDF](docs/informe.pdf), 13 páginas, con formato de los documentos de
  referencia, algoritmo, diagramas, bitácora, análisis, bibliografía y relación
  con las presentaciones de la clase.
- [Imagen con líneas](results/runway_lines.png) y [parámetros](results/runway_lines.csv).
- [90 mediciones individuales](results/runway_timings.csv),
  [estadísticos](results/runway_summary.json) y [entorno](results/runway_metadata.json).

El código y los documentos deben acompañar el enlace al repositorio.

## Ejecutar en esta computadora con Windows y Ubuntu WSL2

Desde PowerShell, ubicado en la raíz del repositorio:

```powershell
./ejecutar.ps1
./ejecutar.ps1 test
./ejecutar.ps1 benchmark
```

El lanzador usa la distribución `Ubuntu`. Si existe `.toolchain/env.sh`, activa
la instalación aislada del proyecto. Si falta CUDA, descarga componentes
oficiales CUDA 13.1 y cabeceras glibc 2.39, verificando SHA256. Requiere
`g++-14`, `make` y `python3`. No instala un driver Linux: usa el de Windows.
La preparación aislada resuelve el conflicto de cabeceras matemáticas entre
CUDA 13.1 y glibc 2.43 de Ubuntu 26.04 sin cambiar las bibliotecas del sistema.

Equivalente desde Ubuntu/WSL2, ubicado en este directorio:

```sh
python3 scripts/setup_toolchain.py  # una vez, si falta un toolkit compatible
source .toolchain/env.sh
make
make test
make run
```

## Compilar con CUDA ya instalado

Requiere Linux/WSL2, `nvcc`, compilador C++ compatible y una GPU NVIDIA con
al menos 36000 bytes de memoria compartida por bloque.

```sh
make
make test
make run OUTPUT=results/demo
make benchmark RUNS=30 OUTPUT=results/nueva_serie
```

`CUDA_ARCH=86` corresponde a la RTX 3060 usada. Para otra GPU, ajustar
`CUDA_ARCH` a una arquitectura soportada por el toolkit. `NVCC`, `HOST_CXX`,
`NVCCFLAGS` y `EXTRA_NVCCFLAGS` son configurables. Tras cambiar banderas, ejecutar
`make clean && make`. No usar `--use_fast_math`, pues puede cambiar bins por
redondeo. `--fmad=false` y `-ffp-contract=off` mantienen la comparación numérica.

## Opciones y archivos generados

```sh
./build/hough data/runway.pgm --runs 30 --output results/demo
./build/hough data/runway.pgm --threshold 4000 --max-lines 8 --output results/umbral
./build/hough --help
```

| Opción | Predeterminado | Significado |
|---|---|---|
| `--runs N` | 10 | Repeticiones medidas por kernel, de 1 a 10000. |
| `--threshold V` | 55% del pico | Solo dibuja votos estrictamente mayores a V. |
| `--max-lines N` | 12 | Máximo tras suprimir vecinos. |
| `--output PREFIJO` | `results/runway` | Directorio y prefijo de salida. |

Admite PGM P2/P5 de 8 bits. Cada píxel no nulo vota en 90 ángulos, de 0 a
178 grados, paso de 2 grados. Se usan 100 bins radiales. El ID de un hilo es
`blockIdx.x * blockDim.x + threadIdx.x`. El acumulador compartido
`localAcc[9000]` usa inicialización cooperativa, dos barreras y atómicos para
votar y fusionar. Los hilos sobrantes del último bloque alcanzan las barreras.

El programa escribe `_lines.png`, `_lines.csv`, `_timings.csv` y `_metadata.json`.
`scripts/benchmark.py` exige al menos diez repeticiones, agrega `_summary.json` y
`_execution.txt`, y registra huellas SHA256. Las rectas no representan clases de
objetos ni segmentos. Para runway, cada bin radial mide 10 píxeles.

**Para conservar los resultados de entrega**, ejecutar las demostraciones con
otro prefijo, como `OUTPUT=results/demo` o `--output results/demo`. La ejecución
predeterminada sobreescribe los archivos `results/runway_*`.

## Resultados registrados

RTX 3060 Laptop GPU, driver 592.82, CUDA 13.1, GCC 14.3 y WSL2. Imagen de
800 x 600, con 138406 píxeles activos. Cinco calentamientos y treinta mediciones
reales por variante, el 6 de noviembre de 2026 a las 18:21, Guatemala (UTC-6).

| Variante | Media (ms) | Desv. muestral (ms) |
|---|---:|---:|
| Global | 1.148245 | 0.135710 |
| Global + constante | 1.328060 | 0.038624 |
| Global + constante + compartida | 0.726596 | 0.030939 |

La compartida obtuvo **1.58x** frente a global, equivalente a 36.72% menos
tiempo del kernel. La constante sola tardó 15.66% más. Se excluyen reinicio,
copias, validación y PNG. El orden fue global, constante, compartida y no se
aislaron relojes ni tareas del escritorio. Los datos describen esta sesión;
véanse [condiciones y limitaciones](results/README.md).

## Validación y base de clase

`make test` aprobó 23 casos en GPU, incluidos bloques parciales, coordenadas
conocidas, imágenes negras/densas, PGM inválidos, CRC y descompresión del PNG.
Las 90 ejecuciones medidas coincidieron con CPU en todos los bins. Los buffers
y eventos se liberan por ámbito y después se reinicia el contexto CUDA.

`make sanitize` intenta memcheck, racecheck y synccheck. En este equipo,
Compute Sanitizer no pudo inicializar la interfaz de depuración WDDM. Los logs
conservan el fallo y no acreditan verificaciones instrumentales aprobadas.

La implementación aplica el flujo host/device, grids/bloques/hilos y jerarquía
de memoria de *Introducción a CUDA*. La acumulación por bloque y la coordinación
por fases siguen los temas de *Eficiencia en el acceso a memoria*, *Coherencia y
seguridad sobre threads* y *Modelo Producer Consumer*. El apéndice A del informe
identifica diapositivas específicas. CUDA events complementa esos contenidos
porque lo solicita el enunciado del proyecto.

## Estructura

```text
src/hough.cu                CPU, tres kernels, eventos y CLI
common/pgm.*               Lectura PGM y validaciones
common/output.*            Selección de rectas, PNG y CSV
tests/test_integration.py   Pruebas en GPU sin paquetes Python externos
scripts/benchmark.py        Bitácora y estadísticas
scripts/setup_toolchain.py  Preparación aislada para Linux/WSL2
scripts/package.py         ZIP de entrega y respaldo Git bundle
data/                      Imagen original
results/                   Mediciones, PNG, parámetros y logs
docs/informe.pdf           Informe final
```

El informe se entrega únicamente en PDF. El equipo elaborará su presentación
por separado.

Para publicar la revisión local, autenticar Git con escritura en el repositorio
y ejecutar `git push origin main`. Los commits anteriores se conservan.

Para generar el paquete desde la revisión confirmada: `python scripts/package.py`.
El ZIP contiene código, datos y documentos, sin `.git`, descargas ni binarios.
El `.bundle` conserva el historial completo y se recupera con
`git clone --branch main PRY3_CUDA_historial.bundle PRY3_CUDA`.

Código base de G. Barlas (2014), entregado con el curso. Véanse
[NOTICE.md](NOTICE.md), [LICENSE](LICENSE) y las referencias del informe.
