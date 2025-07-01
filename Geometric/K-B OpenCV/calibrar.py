import cv2
import numpy as np
import glob
import os
import statistics
import time

def calibrar_y_undistort(directorio, checkerboard, umbral_rmse=2.0):
    print(f"\n📂 Procesando carpeta: {directorio}")
    start_time = time.time()  # Iniciar tiempo total

    subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
    objp = np.zeros((checkerboard[0] * checkerboard[1], 1, 3), np.float32)
    objp[:, 0, :2] = np.mgrid[0:checkerboard[0], 0:checkerboard[1]].T.reshape(-1, 2)

    objpoints, imgpoints = [], []
    images = sorted(glob.glob(os.path.join(directorio, "*.jpg")))
    _img_shape = None

    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        if img is None:
            print(f"⚠️ No se pudo cargar la imagen: {fname}")
            continue

        if _img_shape is None:
            _img_shape = img.shape[:2]
        else:
            assert img.shape[:2] == _img_shape, "Todas las imágenes deben tener el mismo tamaño"

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(
            gray, checkerboard,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        if ret:
            corners_subpix = cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), subpix_criteria)
            imgpoints.append((idx, corners_subpix))
            objpoints.append((idx, objp.copy()))

    N_OK = len(objpoints)
    print(f"🔢 Imágenes válidas detectadas: {N_OK}")
    if N_OK < 3:
        print("❌ No hay suficientes imágenes para calibrar.")
        return

    detected_indices = [idx for idx, _ in objpoints]

    def calibrar_con_flags(flags):
        K = np.zeros((3, 3))
        D = np.zeros((4, 1))
        rvecs = [np.zeros((1, 1, 3)) for _ in range(N_OK)]
        tvecs = [np.zeros((1, 1, 3)) for _ in range(N_OK)]

        start_calib = time.time()
        rms, _, _, rvecs, tvecs = cv2.fisheye.calibrate(
            [o for _, o in objpoints], [i for _, i in imgpoints], _img_shape[::-1],
            K, D, rvecs, tvecs, flags,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
        )
        end_calib = time.time()
        calib_time = end_calib - start_calib
        print(f"⏱️ Tiempo de calibración: {calib_time:.2f} segundos")
        return K, D, rvecs, tvecs, calib_time

    flags_libres = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_FIX_SKEW
    K, D, rvecs, tvecs, calib_time = calibrar_con_flags(flags_libres)

    errores_individuales = {}
    for i, (idx, objp) in enumerate(objpoints):
        projected, _ = cv2.fisheye.projectPoints(objp, rvecs[i], tvecs[i], K, D)
        projected = projected.reshape(-1, 2)
        real = imgpoints[i][1].reshape(-1, 2)
        errores = np.linalg.norm(real - projected, axis=1)
        mean_error = np.mean(errores)
        std_dev = np.std(errores)
        errores_individuales[idx] = (mean_error, std_dev)

    mean_rmse = statistics.mean([e[0] for e in errores_individuales.values()])
    usar_coef_completos = mean_rmse <= umbral_rmse

    if not usar_coef_completos:
        print(f"⚠️ RMSE promedio ({mean_rmse:.2f} px) excede el umbral ({umbral_rmse:.2f} px). Se fijan K2, K3 y K4.")
        flags_limitados = flags_libres + cv2.fisheye.CALIB_FIX_K2 + cv2.fisheye.CALIB_FIX_K3 + cv2.fisheye.CALIB_FIX_K4
        K, D, rvecs, tvecs, calib_time = calibrar_con_flags(flags_limitados)

        errores_individuales = {}
        for i, (idx, objp) in enumerate(objpoints):
            projected, _ = cv2.fisheye.projectPoints(objp, rvecs[i], tvecs[i], K, D)
            projected = projected.reshape(-1, 2)
            real = imgpoints[i][1].reshape(-1, 2)
            errores = np.linalg.norm(real - projected, axis=1)
            mean_error = np.mean(errores)
            std_dev = np.std(errores)
            errores_individuales[idx] = (mean_error, std_dev)
        mean_rmse = statistics.mean([e[0] for e in errores_individuales.values()])

    print("\n📏 Error de reproyección por imagen (Mean ± STD):")
    for idx in range(len(images)):
        if idx in errores_individuales:
            mean_error, std_dev = errores_individuales[idx]
            print(f"📸 Imagen {idx+1:02d}: Mean RMS = {mean_error:.2f}, Std RMS = {std_dev:.2f} px")
        else:
            print(f"📸 Imagen {idx+1:02d}: Mean RMS = 0.00, Std RMS = 0.00 px")

    print(f"\n📊 Error de reproyección medio total: {mean_rmse:.2f}")
    print(f"📐 Tamaño de imagen: {_img_shape[::-1]}")
    print("🔧 Matriz K:\n", K)
    print("🔍 Coeficientes D:\n", D)

    # Guardar parámetros de calibración en un archivo .txt
    calibration_file = os.path.join(directorio, f"resultados_{os.path.basename(directorio)}.txt")
    with open(calibration_file, 'w') as f:
        f.write(f"image_size: {_img_shape[1]} {_img_shape[0]}\n")
        f.write("K:\n")
        np.savetxt(f, K, fmt='%.6f')
        f.write("D:\n")
        np.savetxt(f, D, fmt='%.6f')
    print(f"💾 Parámetros de calibración guardados en: {calibration_file}")

    parent_dir = os.path.basename(os.path.dirname(directorio))
    child_dir = os.path.basename(directorio)
    filename = f"errors_{parent_dir}_{child_dir}.txt"
    ruta_archivo = os.path.join(directorio, filename)

    with open(ruta_archivo, 'w', encoding='utf-8') as f:
        for idx in range(len(images)):
            if idx in errores_individuales:
                mean_error, std_dev = errores_individuales[idx]
                f.write(f"Image {idx+1}: Mean RMS = {mean_error:.2f}, Std RMS = {std_dev:.2f}\n")
            else:
                f.write(f"Image {idx+1}: Mean RMS = 0.00, Std RMS = 0.00\n")

    print(f"💾 Resultados guardados en: {ruta_archivo}")

    output_root = os.path.join(directorio, "salida")
    os.makedirs(output_root, exist_ok=True)

    if usar_coef_completos:
        for bal in [0, 1]:
            subfolder = os.path.join(output_root, f"balance{bal}")
            os.makedirs(subfolder, exist_ok=True)
            new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, _img_shape[::-1], np.eye(3), balance=bal)
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, _img_shape[::-1], cv2.CV_16SC2)
            for fname in images:
                img = cv2.imread(fname)
                undistorted = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)
                nombre_img = os.path.splitext(os.path.basename(fname))[0]
                salida = os.path.join(subfolder, f"corregida_{nombre_img}.jpg")
                cv2.imwrite(salida, undistorted)
                print(f"💾 Guardada: {salida}")
    else:
        scales = [0.8, 1, 1.2]
        for scale in scales:
            subfolder = os.path.join(output_root, f"scale{str(scale).replace('.', '_')}")
            os.makedirs(subfolder, exist_ok=True)
            new_K = K.copy()
            new_K[0, 0] *= scale
            new_K[1, 1] *= scale
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, _img_shape[::-1], cv2.CV_16SC2)
            for fname in images:
                img = cv2.imread(fname)
                undistorted = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)
                nombre_img = os.path.splitext(os.path.basename(fname))[0]
                salida = os.path.join(subfolder, f"{nombre_img}_undistorted.png")
                cv2.imwrite(salida, undistorted)
                print(f"💾 Guardada: {salida}")

    # Calcular y registrar tiempo total
    end_time = time.time()
    total_time = end_time - start_time
    print(f"⏱️ Tiempo total de procesamiento: {total_time:.2f} segundos")

    # Guardar tiempo total en execution_times.txt en la carpeta del dataset
    dataset_dir = os.path.dirname(directorio)  # e.g., "dataset_1"
    tiempos_file = os.path.join(dataset_dir, "execution_times.txt")
    with open(tiempos_file, 'a', encoding='utf-8') as f:
        f.write(f"Dataset {directorio}: {calib_time:.2f} seconds\n")

# Ejecutar
#calibrar_y_undistort("seleccionadas_opencv", (6, 8))        #Usado para la de 140
calibrar_y_undistort("seleccionadas_opencv_180", (6, 8))    #Usado para la de 180