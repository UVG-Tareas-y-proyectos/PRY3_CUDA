#!/usr/bin/env python3
"""Genera el informe desde los datos reales. Requiere reportlab y pillow."""
import csv
from datetime import datetime, timedelta, timezone
from io import BytesIO
import json
from pathlib import Path
from xml.sax.saxutils import escape

from PIL import Image as PILImage
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs'
OUT.mkdir(exist_ok=True)
META = json.loads((ROOT / 'results/runway_metadata.json').read_text(encoding='utf-8'))
STATS = json.loads((ROOT / 'results/runway_summary.json').read_text())
ROWS = list(csv.DictReader((ROOT / 'results/runway_timings.csv').open()))
NAVY = colors.HexColor('#142E45')
TEAL = colors.HexColor('#007F73')
LIGHT = colors.HexColor('#EAF3F4')
GRAY = colors.HexColor('#526373')
font = Path('C:/Windows/Fonts/arial.ttf')
if font.exists():
    pdfmetrics.registerFont(TTFont('Body', str(font)))
    pdfmetrics.registerFont(TTFont('BodyBold', str(font.with_name('arialbd.ttf'))))
else:
    # Deja el generador utilizable tambien en Linux sin fuentes propietarias.
    from reportlab.pdfbase.pdfmetrics import Font
    pdfmetrics.registerFont(Font('Body', 'Helvetica', 'WinAnsiEncoding'))
    pdfmetrics.registerFont(Font('BodyBold', 'Helvetica-Bold', 'WinAnsiEncoding'))
pdfmetrics.registerFontFamily('Body', normal='Body', bold='BodyBold', italic='Body', boldItalic='BodyBold')
styles = getSampleStyleSheet()
styles.add(ParagraphStyle('P', fontName='Body', fontSize=10.7, leading=15.3, spaceAfter=10, textColor=NAVY))
styles.add(ParagraphStyle('SmallText', parent=styles['P'], fontSize=9, leading=12.5, textColor=GRAY))
styles.add(ParagraphStyle('TitleCustom', parent=styles['P'], fontName='BodyBold', fontSize=27, leading=33, spaceAfter=20))
styles.add(ParagraphStyle('H1Custom', parent=styles['P'], fontName='BodyBold', fontSize=20, leading=26, spaceAfter=17))
styles.add(ParagraphStyle('H2Custom', parent=styles['P'], fontName='BodyBold', fontSize=12.3, leading=17, textColor=TEAL, spaceBefore=8))
styles.add(ParagraphStyle('CodeCustom', fontName='Courier', fontSize=8.1, leading=12, textColor=NAVY, spaceAfter=10))
story = []


def p(text, style='P'):
    story.append(Paragraph(text, styles[style]))


def heading(number, title):
    p(f'{number}. {title}', 'H1Custom')


def page():
    story.append(PageBreak())


def table(rows, widths, size=9, row_height=None):
    data = [[Paragraph(escape(str(cell)), ParagraphStyle('cell', fontName='BodyBold' if r == 0 else 'Body', fontSize=size, leading=size+3, textColor=colors.white if r == 0 else NAVY)) for cell in row] for r, row in enumerate(rows)]
    obj = Table(data, colWidths=widths, rowHeights=row_height, repeatRows=1, hAlign='LEFT')
    obj.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), NAVY),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, LIGHT]),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LINEBELOW', (0, 0), (-1, 0), 1, TEAL),
    ]))
    story.append(obj)
    story.append(Spacer(1, 12))


def diagram(kind):
    d = Drawing(480, 195 if kind == 'constant' else 245)
    def box(x, y, w, h, title, detail):
        d.add(Rect(x, y, w, h, fillColor=LIGHT, strokeColor=TEAL, strokeWidth=1))
        d.add(String(x+w/2, y+h-19, title, fontName='BodyBold', fontSize=10, fillColor=NAVY, textAnchor='middle'))
        d.add(String(x+w/2, y+14, detail, fontName='Body', fontSize=8.5, fillColor=GRAY, textAnchor='middle'))
    def arrow(x1, y1, x2, y2):
        d.add(Line(x1, y1, x2, y2, strokeColor=TEAL, strokeWidth=1.5))
        if x1 == x2:
            d.add(Polygon([x2-4, y2+7, x2, y2, x2+4, y2+7], fillColor=TEAL, strokeColor=TEAL))
        else:
            delta = 7 if x2 < x1 else -7
            d.add(Polygon([x2+delta, y2-4, x2, y2, x2+delta, y2+4], fillColor=TEAL, strokeColor=TEAL))
    if kind == 'constant':
        box(0, 125, 140, 54, 'CPU', '90 senos + 90 cosenos')
        box(175, 125, 145, 54, 'CONSTANTE GPU', '720 bytes, solo lectura')
        box(355, 125, 125, 54, 'WARP', 'mismo angulo t')
        arrow(140, 152, 175, 152)
        arrow(320, 152, 355, 152)
        box(0, 25, 140, 54, 'GLOBAL', 'imagen PGM')
        box(175, 25, 145, 54, 'KERNEL', 'calcula rho y su bin')
        box(355, 25, 125, 54, 'GLOBAL', '9000 contadores')
        arrow(140, 52, 175, 52)
        arrow(320, 52, 355, 52)
        arrow(247, 125, 247, 79)
    else:
        box(0, 175, 140, 54, 'GLOBAL + CONSTANTE', 'imagen + tablas trig.')
        box(175, 175, 145, 54, 'BLOQUE DE 256 HILOS', 'inicializa localAcc = 0')
        box(355, 175, 125, 54, 'BARRERA 1', '__syncthreads()')
        arrow(140, 202, 175, 202)
        arrow(320, 202, 355, 202)
        box(355, 85, 125, 54, 'COMPARTIDA', 'atomicAdd local')
        box(175, 85, 145, 54, 'BARRERA 2', 'todos los votos listos')
        box(0, 85, 140, 54, 'GLOBAL', 'fusion con atomicAdd')
        arrow(417, 175, 417, 139)
        arrow(355, 112, 320, 112)
        arrow(175, 112, 140, 112)
        d.add(String(240, 28, 'Cada bloque posee su propio localAcc: 9000 x 4 = 36000 bytes.', fontName='Body', fontSize=10, fillColor=NAVY, textAnchor='middle'))
    return d


def frame(canvas, document):
    canvas.setTitle('Proyecto 3 - Transformada de Hough en CUDA')
    canvas.setAuthor('Ihan-Marroquin; dpatzan2; G1LB3T0')
    if document.page > 1:
        canvas.setFont('Body', 8)
        canvas.setFillColor(GRAY)
        canvas.drawString(54, A4[1]-32, 'UVG  /  COMPUTACIÓN PARALELA Y DISTRIBUIDA')
        canvas.setStrokeColor(TEAL)
        canvas.line(54, A4[1]-40, A4[0]-54, A4[1]-40)
    canvas.setFont('Body', 8)
    canvas.setFillColor(GRAY)
    canvas.drawString(54, 30, 'Proyecto 3  /  Entrega: 11 de noviembre de 2026')
    canvas.drawRightString(A4[0]-54, 30, str(document.page))


# 1. Carátula
story.append(Spacer(1, 48))
p('UNIVERSIDAD DEL VALLE DE GUATEMALA', 'H2Custom')
p('Computación Paralela y Distribuida', 'P')
story.append(Spacer(1, 65))
p('Transformada lineal<br/>de Hough en CUDA', 'TitleCustom')
p('Comparación de memoria global, constante y compartida', 'H2Custom')
story.append(Spacer(1, 54))
table([['Integrantes (autores del repositorio)'], ['Ihan-Marroquin'], ['dpatzan2'], ['G1LB3T0']], [480], 11)
p('Docente: Sebastián Galindo<br/>Proyecto 3<br/>Fecha de entrega: 11 de noviembre de 2026', 'P')
p('Mediciones: 6 de noviembre de 2026, 18:21, Guatemala (UTC-6).', 'SmallText')
p('<link href="https://github.com/UVG-Tareas-y-proyectos/PRY3_CUDA" color="#007F73">Repositorio: UVG-Tareas-y-proyectos / PRY3_CUDA</link>', 'SmallText')
page()

# 2. Índice e introducción
heading(1, 'Índice e introducción')
table([['Contenido', 'Página'], ['1. Índice e introducción', '2'], ['2. Algoritmo y coordenadas', '3'], ['3. Memoria global y constante', '4'], ['4. Memoria compartida y sincronización', '5'], ['5. Metodología y recursos', '6'], ['6. Resultados y análisis', '7'], ['7. Visualización de líneas', '8'], ['8. Bitácora de mediciones', '9'], ['9. Validación y conclusiones', '10'], ['10. Referencias', '11'], ['Apéndice. Reproducción y exposición', '12']], [420, 60], 9)
p('El objetivo del proyecto es detectar rectas en una imagen de bordes y comparar tres formas de distribuir los datos de la Transformada de Hough en CUDA. El trabajo conserva la discretización del código del curso: 90 ángulos, separados por 2 grados, y 100 intervalos de distancia. Cada píxel no nulo emite un voto por ángulo.')
p('La implementación entrega una imagen PNG con las rectas superpuestas a color, un CSV con sus parámetros y una bitácora de 30 tiempos por variante. La versión CPU sirve como referencia exacta y cada ejecución medida comprueba la igualdad de los 9000 contadores.')
p(f'En esta sesión, la media del kernel compartido fue {STATS["shared_ms"]["mean"]:.6f} ms frente a {STATS["global_ms"]["mean"]:.6f} ms del global. La aceleración relativa fue {STATS["global_over_shared"]:.2f}x. Estos resultados corresponden al tiempo del kernel en la RTX 3060, con las condiciones descritas en la metodología.')
page()

# 3. Algoritmo
heading(2, 'Algoritmo y coordenadas')
p('Una recta se representa por su distancia firmada rho respecto del origen y por el ángulo theta de su normal. Para un punto cartesiano (x, y), todos los parámetros que satisfacen rho = x cos(theta) + y sin(theta) describen rectas que pasan por ese punto. La transformación discretiza esa familia y acumula votos. Los máximos identifican familias de puntos alineados. [1, 5]')
p('Discretización del proyecto', 'H2Custom')
table([['Parámetro', 'Valor'], ['theta', '0, 2, 4, ..., 178 grados (90 bins)'], ['rMax', 'sqrt(ancho² + alto²) / 2 = 500 píxeles'], ['rScale', '2 rMax / 100 = 10 píxeles'], ['Acumulador', '100 x 90 = 9000 enteros (36000 bytes)']], [160, 320], 10)
p('El índice de un píxel en memoria es id = fila * ancho + columna. El kernel recupera la columna mediante id % ancho y la fila mediante id / ancho, con división entera. Entonces calcula xCoord = columna - ancho / 2 y yCoord = alto / 2 - fila. El desplazamiento coloca el origen en el centro y el cambio de signo hace que el eje y crezca hacia arriba.')
p('Por ejemplo, para 800 x 600, el píxel de columna 400 y fila 300 pasa a (0, 0); el de columna 400 y fila 299 pasa a (0, 1). Esta convención se usa tanto al votar como al dibujar las líneas, para evitar inversiones de orientación.')
p('Cada voto obtiene rIndex = int((rho + rMax) / rScale). El desplazamiento convierte las distancias negativas en índices no negativos. Un límite explícito al intervalo [0, 99] protege los extremos frente al redondeo. El índice lineal del acumulador es rIndex * 90 + thetaIndex.')
p('Paralelismo y costo', 'H2Custom')
p('La geometría usa bloques unidimensionales de 256 hilos y un hilo por píxel. El ID global es blockIdx.x * blockDim.x + threadIdx.x; el número de bloques es (N - 1) / 256 + 1. Para runway, N = 480000 y se lanzan 1875 bloques. El trabajo principal es proporcional a 90 veces el número de píxeles activos: 12456540 votos en esta imagen.')
p('Aunque cada hilo procesa un píxel distinto, varios píxeles pueden pertenecer a una misma recta y actualizar el mismo contador. Una suma ordinaria perdería incrementos concurrentes. Por ello las actualizaciones GPU emplean atomicAdd. La CPU recorre los píxeles secuencialmente y no requiere atómicos.')
page()

# 4. Constante
heading(3, 'Memoria global y constante')
p('La versión global almacena imagen, acumulador y tablas trigonométricas en memoria global. El host calcula una sola vez los 90 senos y 90 cosenos. Cada hilo consulta las tablas dentro del bucle angular y vota directamente en el acumulador global mediante atomicAdd.')
p('En la segunda versión, los arreglos constantCos y constantSin se declaran con __constant__ fuera de main. El host transfiere los datos con cudaMemcpyToSymbol y el kernel deja de recibir punteros a las tablas. La imagen y el acumulador permanecen en memoria global. Las dos tablas ocupan 720 bytes y no cambian durante la ejecución. [2]')
story.append(diagram('constant'))
p('Figura 1. Flujo funcional de la variante global + constante. Elaboración propia a partir de la implementación y del modelo de memoria CUDA. [2]', 'SmallText')
p('La caché constante favorece accesos uniformes: los hilos activos de un warp solicitan el mismo índice t durante cada iteración. Esta propiedad permite difundir un valor a múltiples hilos. Si los hilos solicitan direcciones distintas, el beneficio puede disminuir. La capacidad de la memoria y su caché dependen del dispositivo; no se debe asumir que toda lectura constante siempre será más rápida. [2, 3]')
p(f'En las 30 mediciones, la variante constante tardó {STATS["constant_ms"]["mean"]:.6f} ms de media, un 15.66% más que la global. Por tanto, mover únicamente las tablas no produjo una mejora en esta sesión. El tráfico de las tablas es pequeño frente a los votos atómicos, y la versión global también puede beneficiarse de cachés. Son explicaciones posibles: no se midieron transacciones ni ocupación para atribuir una causa única.')
p('La documentación moderna de NVIDIA se utiliza para interpretar los resultados. Las afirmaciones de capacidad o latencia de una GPU antigua no se trasladan automáticamente a la RTX 3060 utilizada en este experimento. [2, 3]')
page()

# 5. Compartida
heading(4, 'Memoria compartida y sincronización')
p('La tercera versión mantiene las tablas en constante y crea __shared__ int localAcc[9000] dentro del kernel. Cada bloque posee su propio acumulador de 36000 bytes. La inicialización se reparte entre los hilos: cada locID = threadIdx.x pone a cero las posiciones locID, locID + blockDim.x y así sucesivamente. Una primera barrera garantiza que todos los contadores estén listos antes de votar. [2, 4]')
story.append(diagram('shared'))
p('Figura 2. Acumulación cooperativa dentro de cada bloque y fusión final en memoria global. Elaboración propia. [2, 4]', 'SmallText')
p('Cada píxel activo vota con atomicAdd sobre localAcc. Los atómicos siguen siendo necesarios porque varios hilos del mismo bloque pueden elegir el mismo bin. Después, una segunda barrera evita que la fusión lea votos incompletos. Finalmente, los hilos recorren el acumulador con el mismo patrón por saltos y suman sus posiciones no nulas al acumulador global, también con atomicAdd, porque otros bloques fusionan simultáneamente.')
p('Los hilos que quedan fuera de la imagen en un bloque parcial no leen píxeles ni votan, pero sí inicializan, sincronizan y fusionan. Un return antes de cualquiera de las barreras invalidaría la cooperación del bloque. Por eso las barreras se encuentran fuera de la condición globalId &lt; ancho * alto. [4]')
p(f'La media observada fue {STATS["shared_ms"]["mean"]:.6f} ms, equivalente a {STATS["global_over_shared"]:.2f}x respecto del global. La agrupación local puede reducir contención global al combinar votos antes de fusionarlos. Su costo incluye poner a cero 9000 enteros por bloque, dos barreras y la fusión. Además, 36 kB por bloque consumen un recurso que limita la concurrencia. El balance depende de la imagen y del equipo; la mejora observada no demuestra un aumento de ocupación. [3]')
page()

# 6. Método
heading(5, 'Metodología y recursos')
table([['Elemento', 'Configuración registrada'], ['GPU', META['gpu']], ['Driver / CUDA', '592.82 / CUDA 13.1, nvcc 13.1.115'], ['Host', 'Ubuntu 26.04 en WSL2; GCC 14.3; Windows WDDM'], ['Compilación', '-O2, C++17, sm_86, --fmad=false, -ffp-contract=off'], ['Datos', '800 x 600, 138406 píxeles no nulos'], ['Repeticiones', '5 calentamientos y 30 mediciones por variante'], ['Memorias', '720 B constante; 36000 B compartida por bloque']], [145, 335], 9)
p('Se crea un par de CUDA events por variante. Cada repetición pone el acumulador en cero, registra el evento inicial, lanza el kernel, registra el evento final, espera ese evento y consulta cudaEventElapsedTime. El reinicio queda antes del evento inicial. La copia a CPU, la comparación y la escritura de archivos ocurren después del evento final. Así se mide ejecución del kernel, incluyendo su sincronización interna, en milisegundos. [1]')
p('Después de cada repetición medida se copian los 9000 bins y se exige igualdad exacta con la referencia CPU. Una discrepancia produce código de salida distinto de cero y evita publicar resultados aceptados. Todos los kernels y la CPU usan las mismas tablas float. La compilación desactiva la fusión multiplicación-suma para reducir diferencias de redondeo al discretizar rho.')
p('Los datos se recolectaron el 6 de noviembre de 2026 a las 18:21 en Guatemala. El orden fue global, constante y compartida. La GPU también atiende el escritorio; no se fijaron relojes ni se aisló térmicamente el equipo. Por ello se publican todos los valores, media, mediana, mínimo, máximo y desviación estándar muestral. La comparación es descriptiva para esta sesión.')
p('El uso de vectores en CPU, buffers de dispositivo y eventos con RAII libera los recursos al terminar su ámbito y también durante excepciones. Después de destruir esos recursos, main llama cudaDeviceReset. Se comprueban los estados de reserva, copia, eventos y lanzamiento. La lectura PGM rechaza datos incompletos, valores fuera de rango y dimensiones incompatibles con los índices de 32 bits.')
p('La instalación aislada usa cabeceras glibc 2.39 para compilar con CUDA 13.1 en Ubuntu 26.04, cuyo sistema tiene glibc 2.43. Esto resuelve el conflicto de declaraciones rsqrt/rsqrtf sin alterar las bibliotecas del sistema. scripts/setup_toolchain.py reproduce esa preparación desde descargas oficiales verificadas por SHA256.', 'SmallText')
page()

# 7. Estadísticos
heading(6, 'Resultados y análisis')
table([['Variante', 'Media ms', 'Mediana ms', 'Desv. ms', 'Rango ms'], *[[label, f'{STATS[key]["mean"]:.6f}', f'{STATS[key]["median"]:.6f}', f'{STATS[key]["stdev"]:.6f}', f'{STATS[key]["minimum"]:.3f} - {STATS[key]["maximum"]:.3f}'] for label, key in [('Global', 'global_ms'), ('Constante', 'constant_ms'), ('Compartida', 'shared_ms')]]], [112, 91, 91, 91, 95], 8.6)
d = Drawing(480, 245)
chart = VerticalBarChart()
chart.x, chart.y, chart.width, chart.height = 45, 35, 410, 185
chart.data = [[STATS[k]['mean'] for k in ['global_ms', 'constant_ms', 'shared_ms']]]
chart.categoryAxis.categoryNames = ['Global', 'Constante', 'Compartida']
chart.valueAxis.valueMin, chart.valueAxis.valueMax, chart.valueAxis.valueStep = 0, 1.6, .4
chart.categoryAxis.labels.fontName = chart.valueAxis.labels.fontName = 'Body'
chart.categoryAxis.labels.fontSize = chart.valueAxis.labels.fontSize = 9
chart.bars[0].fillColor = TEAL
chart.bars.strokeColor = None
chart.barLabels.fontName = 'BodyBold'
chart.barLabelFormat = '%.3f'
chart.barLabels.fontSize = 10
d.add(chart)
d.add(String(45, 232, 'Tiempo medio del kernel (ms), n = 30 por variante', fontName='BodyBold', fontSize=10, fillColor=NAVY))
story.append(d)
p('Figura 3. Comparación descriptiva de medias. Fuente: results/runway_timings.csv. La barra más baja corresponde a menor tiempo. Los tiempos individuales están en la página 9.', 'SmallText')
p('El cociente media global / media compartida es 1.5803. La disminución relativa del tiempo es 1 - 0.726596 / 1.148245 = 36.72%. La versión constante obtiene un cociente de 0.8646, lo que corresponde a 15.66% más tiempo que la global. Estos cocientes comparan kernels que producen el mismo acumulador.')
p('La mayor dispersión relativa corresponde a la versión global, con desviación muestral de 0.135710 ms. La mediana ayuda a describir su centro sin depender tanto de las mediciones extremas. La dispersión también muestra por qué una sola ejecución no basta para establecer un resultado de rendimiento.')
p('La referencia CPU tardó 26.825217 ms en una única ejecución. Ese dato sirve de contexto y de validación, pero no es una serie estadística comparable a las 30 mediciones GPU. Tampoco representa un speedup de toda la aplicación, porque el intervalo GPU excluye reservas, transferencias y generación del PNG.')
page()

# 8. Salida visual
heading(7, 'Visualización de líneas')
original = PILImage.open(ROOT / 'data/runway.pgm').convert('RGB')
buffer = BytesIO()
original.save(buffer, format='PNG')
buffer.seek(0)
story.append(Image(buffer, width=400, height=300))
p('Figura 4. Imagen original de bordes, conservada sin modificaciones. Fuente: archivo runway.pgm proporcionado por el curso.', 'SmallText')
story.append(Image(str(ROOT / 'results/runway_lines.png'), width=400, height=300))
p('Figura 5. Salida del programa. Doce rectas sobre la imagen original; votos &gt; 2772. El color distingue las rectas, sin codificar confianza ni clase semántica.', 'SmallText')
page()

# 9. Bitácora
heading(8, 'Bitácora de mediciones')
p('Treinta mediciones por variante mediante CUDA events. Todas las celdas contienen milisegundos. Las 90 ejecuciones medidas coincidieron exactamente con la CPU; los cinco calentamientos de cada variante no están en esta tabla.', 'SmallText')
table([['Repetición', 'Global (ms)', 'Constante (ms)', 'Compartida (ms)'], *[[row['run'], row['global_ms'], row['constant_ms'], row['shared_ms']] for row in ROWS]], [90, 130, 130, 130], 8.4, 17)
p('Fuente: results/runway_timings.csv. La desviación estándar publicada usa divisor n - 1. scripts/benchmark.py calcula el resumen directamente de estos valores y registra las huellas SHA256 del binario y de la imagen.', 'SmallText')
page()

# 10. Validación y conclusiones
heading(9, 'Validación y conclusiones')
p('Validación funcional', 'H2Custom')
p('La batería automática aprobó 23 casos. Cubre imagen negra, un solo píxel, 257 píxeles (bloque final parcial), imagen densa de dimensiones impares, puntos aleatorios, líneas verticales, horizontales y diagonales, y bytes que podrían confundirse con espacios en PGM binario. También rechaza cabeceras vacías, formatos no admitidos, dimensiones cero o negativas, datos truncados, profundidades de 16 bits y valores fuera de rango.')
p('Las pruebas geométricas comprueban picos esperados para líneas horizontales y verticales y el efecto de discretización sobre una diagonal de 45 grados. Además, verifican dimensiones, canal RGB, CRC de los chunks PNG y descompresión zlib. La referencia CPU comprueba conservación de votos: 90 por cada píxel activo.')
p('Compute Sanitizer intentó memcheck, racecheck y synccheck, pero WDDM rechazó la interfaz de depuración. Por ello no se afirma que la instrumentación haya demostrado ausencia de carreras o fugas. Los logs conservan el fallo y make sanitize permite repetir las verificaciones en un entorno compatible. La igualdad CPU/GPU y las pruebas funcionales sí se ejecutaron en hardware real.')
p('Interpretación visual y límites', 'H2Custom')
p('La selección ordena los bins por votos y acepta únicamente pesos estrictamente superiores al umbral. Suprime candidatos a una distancia de hasta 2 bins radiales y 4 bins angulares de una recta aceptada, incluida la equivalencia de ángulos cerca de 0/180 grados. Se limita la salida a 12 rectas. Este procedimiento reduce duplicados, pero no clasifica las rectas como bordes de pista ni distingue objetos.')
p('El dibujo usa el centro de cada bin radial. Con pasos de 10 píxeles puede existir un desplazamiento de hasta aproximadamente medio bin, además del error angular. Cada recta se prolonga hasta los límites de la imagen; el algoritmo no estima segmentos ni extremos. Un umbral configurable permite reducir rectas secundarias o aumentar sensibilidad.')
p('Conclusiones y recomendaciones', 'H2Custom')
p('La implementación satisface el cálculo del ID global, la comparación exacta de resultados y la salida visual. Las barreras fuera de la condición de píxel válido permiten procesar bloques incompletos correctamente. Los atómicos son necesarios tanto al votar dentro del bloque como al fusionar entre bloques.')
p('En esta sesión, la variante compartida redujo el tiempo medio del kernel un 36.72% frente a la global. La constante sola no mejoró el rendimiento. Para generalizar el resultado conviene repetir las series con orden aleatorio, controlar el estado térmico y medir contención y ocupación con un perfilador. Cambiar resolución de bins o tamaño de bloque requiere volver a comprobar tanto exactitud como uso de memoria.')
page()

# 11. Bibliografía
heading(10, 'Referencias')
refs = [
    ('[1] Harris, M. (2012, 7 de noviembre). How to Implement Performance Metrics in CUDA C/C++. NVIDIA Technical Blog.', 'https://developer.nvidia.com/blog/how-implement-performance-metrics-cuda-cc/'),
    ('[2] NVIDIA. (2025). CUDA C++ Programming Guide, versión 12.9.1. Jerarquía de memoria, calificadores __constant__/__shared__, atómicos y sincronización.', 'https://docs.nvidia.com/cuda/archive/12.9.1/cuda-c-programming-guide/index.html'),
    ('[3] NVIDIA. (2025). CUDA C++ Best Practices Guide, versión 12.9.1. Medición, memoria, cachés y ocupación.', 'https://docs.nvidia.com/cuda/archive/12.9.1/cuda-c-best-practices-guide/index.html'),
    ('[4] Harris, M. (2013, 28 de enero). Using Shared Memory in CUDA C/C++. NVIDIA Technical Blog.', 'https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/'),
]
for text, url in refs:
    p(text)
    p(f'<link href="{url}" color="#007F73">Consultar fuente oficial de NVIDIA</link>', 'SmallText')
p('[5] Galindo, S. (s. f.). Proyecto 3 - CUDA. Universidad del Valle de Guatemala. Guía de proyecto y rúbrica proporcionadas por el curso. Se utiliza la fecha de entrega indicada por el equipo: 11 de noviembre de 2026.')
p('[6] Barlas, G. (2014, diciembre). houghBase.cu, versión 1.0. Código fuente entregado con el proyecto; cabecera con licencia GNU GPL 3.0. La atribución se conserva en NOTICE.md y en el código.')
p('Los números entre corchetes remiten a esta bibliografía. Las descripciones de implementación y las figuras funcionales documentan el código del repositorio. Los resultados cuantitativos proceden exclusivamente de los archivos de la ejecución final, sin extrapolar rendimientos de otros equipos.', 'SmallText')
p('Material suplementario conservado', 'H2Custom')
p('El repositorio incluye el código CUDA y C++, la imagen original, el Makefile, pruebas automáticas, scripts de compilación y medición, las 90 mediciones sin resumir, metadatos del entorno, la salida PNG, los parámetros de las líneas y los logs de validación. El informe y la presentación se entregan junto con el código, conforme a la indicación de no enviar solamente un enlace.')
page()

# 12. Reproducción
p('Apéndice. Reproducción y exposición', 'H1Custom')
p('Compilación con CUDA instalado', 'H2Custom')
for text in ['make', 'make test', 'make benchmark RUNS=30', 'make sanitize']:
    p(escape(text), 'CodeCustom')
p('El valor predeterminado CUDA_ARCH=86 corresponde a la RTX 3060. Puede ajustarse a otra arquitectura soportada por el toolkit. HOST_CXX y EXTRA_NVCCFLAGS permiten seleccionar el compilador y sus rutas sin modificar el código.')
p('Preparación aislada para esta instalación WSL2', 'H2Custom')
for text in ['python3 scripts/setup_toolchain.py', 'source .toolchain/env.sh', 'make', 'make test']:
    p(escape(text), 'CodeCustom')
p('La preparación requiere make y g++-14. Descarga componentes oficiales CUDA 13.1 y cabeceras Ubuntu a .toolchain/, directorio ignorado por Git. No instala un driver Linux: WSL usa el driver NVIDIA de Windows.')
p('Ejecución y ajuste del umbral', 'H2Custom')
p('./build/hough data/runway.pgm --runs 30 \\<br/>  --threshold 4000 --max-lines 8 --output results/custom', 'CodeCustom')
p('La opción --threshold compara con &gt;, por lo que un peso exactamente igual al umbral no se dibuja. El prefijo determina los cuatro archivos de salida; el script benchmark agrega un resumen y el registro de ejecución. Para regenerar este informe: pip install reportlab pillow y python scripts/build_report.py.')
p('Organización de la exposición oral', 'H2Custom')
table([['Participación sugerida', 'Contenido'], ['dpatzan2', 'Algoritmo, coordenadas y sincronización del bloque'], ['G1LB3T0', 'Salida PNG, umbral, memorias y demostración'], ['Ihan-Marroquin', 'CUDA events, bitácora, resultados y límites']], [145, 335], 9.5)
p('La presentación ejecutiva incluye notas para apoyar la explicación. La demostración puede ejecutar make run y abrir results/runway_lines.png. Cada integrante debe poder explicar por qué varios píxeles compiten por un mismo bin y por qué todos los hilos alcanzan las barreras. La preparación y participación oral corresponden al equipo.')

doc = SimpleDocTemplate(str(OUT / 'informe.pdf'), pagesize=A4, rightMargin=54, leftMargin=54, topMargin=58, bottomMargin=50)
doc.build(story, onFirstPage=frame, onLaterPages=frame)
print(OUT / 'informe.pdf')
