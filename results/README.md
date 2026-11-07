# Mediciones de la ejecución final

Fecha local: 6 de noviembre de 2026, 18:21 (Guatemala, UTC-6).
Entrada original: `data/runway.pgm`, 800 x 600, 138406 pixeles no nulos.
Equipo: NVIDIA GeForce RTX 3060 Laptop GPU, compute capability 8.6,
driver 592.82, CUDA 13.1, GCC 14.3, Ubuntu 26.04 bajo WSL2.

Cada versión usa 5 calentamientos no registrados y **30 mediciones reales con
CUDA events**. Cada repetición verifica los 9000 bins contra la CPU. El intervalo
incluye únicamente el kernel; quedan fuera reinicio, copias, validación y PNG.
Las variantes se midieron en el orden global, constante y compartida. La
comparación describe esta sesión; no controla frecuencias, consumo ni tareas
del escritorio y no constituye una garantía de aceleración en otros equipos.

| Versión | Media (ms) | Desv. muestral (ms) | Mediana (ms) |
|---|---:|---:|---:|
| Global | 1.148245 | 0.135710 | 1.119232 |
| Global + constante | 1.328060 | 0.038624 | 1.312768 |
| Global + constante + compartida | 0.726596 | 0.030939 | 0.726016 |

La versión compartida obtuvo 1.5803 veces el rendimiento de la global en el
tiempo del kernel (36.72% menos tiempo). La constante sola tardó 15.66% más.
Son observaciones, no diagnósticos de ocupación ni de caché. La CPU tardó
26.825217 ms en una sola ejecución; no se usa como benchmark estadístico ni
como comparación de tiempo completo de la aplicación.

`runway_timings.csv` conserva las 90 mediciones; `runway_summary.json` contiene
estadísticos calculados; `runway_metadata.json` identifica entorno y SHA256;
`runway_execution.txt` contiene la salida. El PNG y el CSV de líneas usan votos
estrictamente mayores a 2772, supresión de vecinos y un máximo de 12 rectas.

`tests.txt`: 23 casos aprobados en GPU, con comparación exacta CPU/GPU en todas
las repeticiones medidas. `sanitizer_*.txt`: **verificación instrumental no
disponible**. Compute Sanitizer falló al inicializar la interfaz de depuración
WDDM y reportó dispositivo no soportado para esa sesión. Estos archivos no
acreditan ausencia de carreras ni fugas. Puede repetirse con `make sanitize`
en un entorno que tenga habilitada la interfaz de depuración.

El compilador usa un sysroot aislado con cabeceras glibc 2.39 para evitar el
conflicto de declaraciones `rsqrt/rsqrtf` entre CUDA 13.1 y glibc 2.43. No se
modificaron las cabeceras del toolkit ni la biblioteca glibc del sistema.
