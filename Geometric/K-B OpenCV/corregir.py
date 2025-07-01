import cv2
import numpy as np
import glob
import os

def correct_images(calibration_file, images_dir, output_dir, balance_values=[0, 0.5, 1], scales=[0.8, 1, 1.2]):
    """
    Corrige imágenes usando balance o escalas en el archivo de calibración.
    
    Args:
        calibration_file (str): Ruta al archivo de calibración.
        images_dir (str): Directorio con las imágenes a corregir.
        output_dir (str): Directorio donde se guardarán las imágenes corregidas.
        balance_values (list): Lista de valores de balance (default: [0, 0.5, 1]).
        scales (list): Lista de valores de escala (default: [0.8, 1, 1.2]).
    """
    print(f"Buscando imágenes en: {os.path.abspath(images_dir)}")
    images = sorted(glob.glob(os.path.join(images_dir, "*.jpg")))  # Ajusta la extensión si es necesario
    if not images:
        print("❌ No se encontraron imágenes en el directorio especificado.")
        return

    # Leer los parámetros de calibración
    with open(calibration_file, 'r') as f:
        lines = f.readlines()
        image_size = tuple(map(int, lines[0].split()[1:]))
        K = np.loadtxt(lines[2:5])
        D = np.loadtxt(lines[6:10]).reshape(4, 1)

    # Determinar si usar escalas o balance basado en D[3]
    if D[3] == 0:
        print("🔍 D[3] es cero. Usando escalas para la corrección.")
        for scale in scales:
            subfolder = os.path.join(output_dir, f"scale{str(scale).replace('.', '_')}")
            os.makedirs(subfolder, exist_ok=True)
            new_K = K.copy()
            new_K[0, 0] *= scale  # Escalar el término focal en x
            new_K[1, 1] *= scale  # Escalar el término focal en y
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, image_size, cv2.CV_16SC2)
            for fname in images:
                img = cv2.imread(fname)
                if img is None:
                    print(f"⚠️ No se pudo cargar la imagen: {fname}")
                    continue
                undistorted = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)
                nombre_img = os.path.splitext(os.path.basename(fname))[0]
                salida = os.path.join(subfolder, f"{nombre_img}_undistorted.png")
                cv2.imwrite(salida, undistorted)
                print(f"💾 Guardada: {salida}")
    else:
        print("🔍 D[3] no es cero. Usando balance para la corrección.")
        for balance in balance_values:
            subfolder = os.path.join(output_dir, f"balance{balance}")
            os.makedirs(subfolder, exist_ok=True)
            new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, image_size, np.eye(3), balance=balance)
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, image_size, cv2.CV_16SC2)
            for fname in images:
                img = cv2.imread(fname)
                if img is None:
                    print(f"⚠️ No se pudo cargar la imagen: {fname}")
                    continue
                undistorted = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)
                nombre_img = os.path.splitext(os.path.basename(fname))[0]
                salida = os.path.join(subfolder, f"corregida_{nombre_img}.jpg")
                cv2.imwrite(salida, undistorted)
                print(f"💾 Guardada: {salida}")

# Ejemplo de uso
if __name__ == "__main__":
    calibration_file = "seleccionadas_opencv_180/resultados_seleccionadas_opencv_180.txt"
    sets = ["cielo1", "cielo2", "cielo3"]  # Lista de conjuntos
    base_images_dir = "seleccionadas_opencv_180/cielo_180/"  # Directorio base de imágenes
    base_output_dir = "seleccionadas_opencv_180/cielo_180/corrected/"  # Directorio base de salida
    balance_values = [0, 0.5, 1]
    scales = [0.4, 0.6, 1]  # Valores de escala cuando D[3] == 0

    for set_name in sets:
        images_dir = os.path.join(base_images_dir, set_name)  # Ruta de entrada para cada conjunto
        set_output_dir = os.path.join(base_output_dir, set_name)  # Ruta de salida para cada conjunto
        correct_images(calibration_file, images_dir, set_output_dir, balance_values, scales)