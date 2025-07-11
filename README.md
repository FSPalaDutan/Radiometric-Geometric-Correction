# Corrección Geométrica y Radiométrica de Imágenes de Cielo

Este repositorio, `fspaladutan-radiometric-geometric-correction`, contiene scripts para la corrección geométrica y radiométrica de imágenes de cielo capturadas con cámaras de gran angular (140° y 180°). Incluye implementaciones de métodos de corrección geométrica (Kannala-Brandt, Scaramuzza, UCM, EUCM, Mei) y radiométrica (generación de imágenes HDR con Debevec, Mitsunaga y Robertson; corrección de viñeteado con Gao, Lopez-Fuentes y Zheng), junto con parámetros de calibración precalculados, datasets de prueba y ejemplos de imágenes originales y corregidas. El objetivo es facilitar la replicación y adaptación de estos métodos para aplicaciones en visión por computadora y predicción fotovoltaica.

## Estructura del Repositorio

El repositorio está organizado en tres carpetas principales:

- **`Geometric/`**: Scripts para corrección geométrica, organizados por modelos:
  - `EUCM, UCM y Mei/`
  - `K-B OpenCV/`
  - `Scaramuzza/`
- **`HDR/`**: Scripts para generar imágenes HDR y aplicar operadores de *tone mapping*.
- **`Vignetting/`**: Scripts para corrección de viñeteado (Gao, Lopez-Fuentes, Zheng).

Cada carpeta contiene scripts, parámetros de calibración precalculados y ejemplos de imágenes para pruebas.

## Métodos de Corrección Geométrica

Los métodos implementados mitigan distorsiones en imágenes de gran angular. Sus características principales son:

- **Kannala-Brandt (OpenCV)**: Alta calidad visual (PSNR: 23.63 dB, SSIM: 0.94), ideal para segmentación de nubes.
- **Scaramuzza**: Precisión geométrica superior (RPE: 0.159 píxeles), adecuado para reconstrucción 3D.
- **UCM, EUCM, Mei**: Equilibrio entre precisión y estabilidad, optimizados para lentes de 180° (RPE: 0.129–0.140 píxeles).

### Estructura de la Carpeta `Geometric/`

- **`EUCM, UCM y Mei/`**: Código C++ para calibración y rectificación. Dependencias: OpenCV, Ceres Solver, Eigen, Boost.
- **`K-B OpenCV/`**: Scripts en Python usando el módulo `fisheye` de OpenCV.
- **`Scaramuzza/`**: Scripts en Python con parámetros precalculados en `checkpoints/`.
- **`parametros_distorsion/`**: Parámetros de calibración para cada modelo y lente (140° y 180°).

### Instrucciones de Uso

#### Calibración

La calibración usa imágenes de tableros de ajedrez. Los parámetros precalculados permiten omitir este paso si se desea.

- **EUCM, UCM, Mei**:
  - **Configuración**: Crear un archivo JSON (e.g., `data/calib_example.json`) con rutas a imágenes de tableros, modelo de cámara (`eucm`, `ucm`, o `mei`) y características del tablero (e.g., 8 filas, 6 columnas, tamaño 0.03 m).
  - **Ejemplo**:
    ```bash
    cd Geometric/EUCM,\ UCM\ y\ Mei/build
    ./calib config_calibracion.json > resultados_calibracion.txt
    ```
  - **Detalles**: Genera parámetros en `resultados_calibracion.txt`. Compilar `calib` en `build/`.
  - **Salida**: Archivos de parámetros en texto.

- **Kannala-Brandt (OpenCV)**:
  - **Script**: Editar `calibrar.py` con directorio y tamaño del tablero (e.g., `calibrar_y_undistort("180", (6, 8))`).
  - **Ejemplo**:
    ```bash
    cd Geometric/K-B\ OpenCV/
    python calibrar.py
    ```

- **Scaramuzza**:
  - **Requisitos**:
    - Imágenes del tablero (JPG, PNG, etc.).
    - Número de filas y columnas (e.g., 6x8).
    - Tamaño de los cuadrados (e.g., 30 mm).
  - **Ejemplo**:
    ```bash
    python src/pyocamcalib/script/calibration_script.py "path/to/dataset" 6 8 --camera-name mi_camara --square-size 30 --no-show-results
    ```
  - **Detalles**: Usa `--camera-name` para nombrar el archivo y `--square-size` para el tamaño en mm.
  - **Salida**: Archivo JSON en `src/pyocamcalib/checkpoints/calibration/`.

#### Rectificación

Usa parámetros precalculados o generados para corregir imágenes.

- **EUCM, UCM, Mei**:
  - **Configuración**: Crear un JSON (e.g., `config_rectify.json`) con modelo, parámetros intrínsecos (de `parametros_distorsion/<modelo>/`), parámetros pinhole (ancho, alto, centro, FOV) y rutas de imágenes.
  - **Ejemplo**:
    ```bash
    cd Geometric/EUCM,\ UCM\ y\ Mei/build
    ./rectify_ucm_mei config_rectify.json directorio_salida
    ```
  - **Detalles**: FOVs sugeridos: 300, 400, 550. Imágenes guardadas por modelo, dataset y FOV.
  - **Salida**: Imágenes JPG en `directorio_salida`.

- **Kannala-Brandt (OpenCV)**:
  - **Script**: Configurar `corregir.py`.
  - **Ejemplo**:
    ```bash
    python corregir.py
    ```

- **Scaramuzza**:
  - **Requisitos**:
    - Archivo de calibración.
    - Imágenes (JPG, PNG, etc.).
    - FOV y dimensiones de salida.
  - **Ejemplo**:
    ```bash
    python src/pyocamcalib/script/projection_conversion_script.py "path/to/imagen.jpg" "path/to/checkpoints/calibration/calibration_SCARAMUZZA-mi_camara.json" 140 1000 1000
    ```
  - **Detalles**: Genera imágenes con sufijo (e.g., `imagen_fov140.jpg`). Organizar manualmente.
  - **Salida**: Imágenes JPG en la carpeta original.

## Métodos de Corrección Radiométrica

### Generación de Imágenes HDR

- **Métodos**: Debevec (mejor para nubes), Mitsunaga, Robertson.
- **Resultados**: Reducción de saturación (e.g., 56.32–59.26% en 140° con Reinhard).
- **Ubicación**: `HDR/`.

#### Instrucciones de Uso

- **Scripts**: `debevec.py`, `mitsunaga.py`, `robertson.py`.
- **Ejemplo**:
  ```bash
  cd HDR/
  python .\main.py .\Images\dataset1\ mitsunaga
  ```
- **Datasets**: `HDR/Images/140/` y `180/` (nublado, semi soleado, soleado).

### Corrección de Viñeteado

- **Métodos**:
  - **Lopez-Fuentes**: Alto PSNR en sintéticas (42.26 dB).
  - **Zheng**: Uniformidad periférica en reales (σ: 0.2047).
  - **Gao**: Rendimiento intermedio, mejor en el centro.

#### Instrucciones de Uso

- **Lopez-Fuentes**:
  - **Requisitos**:
    - `VignettingCorrection_color.exe` compilado.
    - Imagen JPG.
    - Centro óptico (`u0`, `v0`).
  - **Ejemplo**:
    ```bash
    .\VignettingCorrection_color.exe "ruta/a/imagen.jpg" 820 616
    ```
  - **Salida**: Imagen con sufijo `_corrected.jpg` en `sky_finales_lopez`.

- **Zheng**:
  - **Requisitos**:
    - `VignettingCorrection.exe` compilado.
    - Imagen JPG.
    - Centro óptico (`u0`, `v0`).
  - **Ejemplo**:
    ```bash
    .\VignettingCorrection.exe "ruta/a/imagen.jpg" 820 616
    ```
  - **Salida**: Imagen con sufijo `_corrected.jpg`.

- **Gao**:
  - **Requisitos**:
    - ROS (e.g., Noetic) con `camera_model` y `vignetting_model`.
    - Imágenes de tableros y a corregir (JPG/PNG).
    - Tablero (e.g., 6x8, 30 mm).
    - Centro óptico (`u0`, `v0`).
  - **Pasos**:
    1. Convertir a grises:
       ```bash
       convert imagen.jpg -colorspace Gray gray_imagen.jpg
       ```
    2. Calibración intrínseca:
       ```bash
       rosrun camera_model Calibration --camera-name mi_camara --input path/to/images/ -p gray_imagen_ -e jpg -w 8 -h 6 -s 30 --camera-model mei --opencv true --save_result true --view-results true
       ```
    3. Calibración de viñeteado:
       ```bash
       rosrun vignetting_model vignetting_calib --camera-name mi_camara --input path/to/images/ -p gray_imagen_ -e jpg -w 8 -h 6 --size 30 --opencv true --resize-scale 1.0 --center_x 820 --center_y 616 --save_result true --is_color false
       ```
    4. Corrección:
       ```bash
       rosrun vignetting_model vignetting_test --camera-name mi_camara --input path/to/images/ -p gray_imagen_ -e jpg -w 8 -h 6 --size 30 --opencv true --center_x 820 --center_y 616 --save_result true --is_color false --result_images_save_folder path/to/corrected_images
       ```
  - **Salida**: Imágenes en `path/to/corrected_images`.

## Requisitos y Dependencias

- **General**:
  - OpenCV 4.5+
  - Python 3.x (`numpy`, `opencv-python`)
- **Geometric/EUCM, UCM y Mei/**:
  - Ceres Solver 2.1+
  - Eigen 3.3+
  - Boost 1.74+
  - Glog
- **Vignetting/**:
  - Compilador C++ (e.g., g++)

## Resultados Esperados

- **Geométrica**: Imágenes con distorsión minimizada, bordes rectificados y alineación precisa.
- **HDR**: Detalles mejorados en regiones de alto contraste (nubes, sol).
- **Viñeteado**: Uniformidad radiométrica mejorada, especialmente en periferias.
