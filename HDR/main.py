import sys
import os
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
from utils import load_img, read_exposure_times, progress
from debevec import response_curve_Debevec
from mitsunaga import response_curve_Mitsunaga
from robertson import robertson_method
from tone_mapping import tone_mapping, save_hdr
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='cv2')

def command_validate():
    """
    Valida los argumentos de línea de comandos.
    Returns:
        bool: True si los argumentos son válidos, False en caso contrario.
    """
    if len(sys.argv) != 3:
        print('Usage: python main.py <image folder> <method: debevec|mitsunaga|robertson>')
        return False
    return True

def radiance_map(g, imgs, ln_t, w, status):
    """
    Construye un mapa de radiancia para un canal.
    Args:
        g: Curva de respuesta logarítmica.
        imgs: Lista de imágenes para un canal.
        ln_t: Logaritmo de los tiempos de exposición.
        w: Pesos para los valores de píxeles.
        status: Mensaje para la barra de progreso.
    Returns:
        ln_E: Mapa de radiancia logarítmico.
    """
    Z = np.array([img.flatten() for img in imgs])
    acc_E = np.zeros(Z[0].shape)
    ln_E = np.zeros(Z[0].shape)
    pixels, imgs_count = Z.shape[1], Z.shape[0]
    for i in range(pixels):
        progress(i, pixels, status)
        acc_w = 0
        for j in range(imgs_count):
            z = Z[j][i]
            acc_E[i] += w[z] * (g[z] - ln_t[j])
            acc_w += w[z]
        ln_E[i] = acc_E[i] / acc_w if acc_w > 0 else 0
    return ln_E

def construct_hdr(img_list, curve_list, exposures):
    """
    Construye una imagen HDR a partir de imágenes y curvas de respuesta.
    Args:
        img_list: Lista de imágenes por canal (B, G, R).
        curve_list: Lista de curvas de respuesta por canal.
        exposures: Tiempos de exposición.
    Returns:
        Imagen HDR en formato float32.
    """
    img_size = img_list[0][0].shape
    w = [i if i < 127.5 else 255-i for i in range(256)]
    hdr = np.zeros((img_size[0], img_size[1], 3), 'float32')
    vexp = np.vectorize(lambda x: math.exp(x))
    for i in range(3):
        E = radiance_map(curve_list[i], img_list[i], np.log(exposures), w, f'channel {"bgr"[i]}')
        hdr[..., i] = np.reshape(vexp(E), img_size)
        print('')
    return hdr

def main():
    """
    Orquesta el proceso de creación de imágenes HDR y tone mapping.
    """
    if not command_validate():
        return

    img_dir = sys.argv[1]
    method = sys.argv[2].lower()

    # Cargar imágenes y extraer canales
    print("Cargando imágenes ...")
    files, exposures = read_exposure_times(img_dir)
    imgs = [cv2.imread(os.path.join(img_dir, f)) for f in files]
    imgs_b = [img[:, :, 0] for img in imgs]
    imgs_g = [img[:, :, 1] for img in imgs]
    imgs_r = [img[:, :, 2] for img in imgs]

    # Mapa de métodos para curvas de respuesta
    method_map = {
        'debevec': lambda imgs, exp: [response_curve_Debevec(ch, exp)[0] for ch in imgs],
        'mitsunaga': lambda imgs, exp: [response_curve_Mitsunaga(ch, exp) for ch in imgs],
        'robertson': lambda imgs, exp: robertson_method(imgs, exp)
    }

    if method not in method_map:
        print("Método no reconocido")
        return

    print(f"Aplicando método {method} ...")
    curve_list = method_map[method]([imgs_b, imgs_g, imgs_r], exposures)
    suffix = f'_{method[0]}'

        # Graficar curvas de respuesta
    plt.figure(figsize=(12, 4))
    for i, color in enumerate(['b', 'g', 'r']):
        plt.subplot(1, 3, i+1)
        plt.plot(curve_list[i], color=color, label=f'Canal {["B", "G", "R"][i]}')
        plt.title(f'Canal {["B", "G", "R"][i]}')
        plt.xlabel('Valor de píxel (Z)')
        plt.ylabel('Log Exposición (g(Z))')
        plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, f'response_curves{suffix}.png'))
    plt.close()
    print(f"Curvas de respuesta guardadas en response_curves{suffix}.png")

    # Construir y guardar HDR
    print("Construyendo HDR ...")
    hdr = construct_hdr([imgs_b, imgs_g, imgs_r], curve_list, exposures)
    
    # Validar imagen HDR
    if np.any(np.isnan(hdr)) or np.any(np.isinf(hdr)):
        print("Advertencia: La imagen HDR contiene valores NaN o infinitos. Normalizando...")
        hdr = np.nan_to_num(hdr, nan=0.0, posinf=np.max(hdr[np.isfinite(hdr)]), neginf=0.0)
    
    print("Guardando mapa de radiancia ...")
    plt.figure(figsize=(12, 8))
    plt.imshow(np.log(cv2.cvtColor(hdr, cv2.COLOR_BGR2GRAY) + 1e-5), cmap='jet')
    plt.colorbar()
    plt.savefig(os.path.join(img_dir, f'radiance-map{suffix}.png'))
    plt.close()
    print("Mapa de radiancia guardado")

    print("Guardando archivo HDR ...")
    save_hdr(hdr, os.path.join(img_dir, f'output{suffix}.hdr'))
    print("Archivo HDR guardado")

    # Tone mapping
    methods = ['Reinhard', 'Mantiuk', 'global', 'durandanddorsy', 'Drago']
    results_dir = os.path.join(img_dir, 'Results')
    os.makedirs(results_dir, exist_ok=True)

    for tm_method in methods:
        print(f"Aplicando tone mapping - {tm_method} ...")
        ldr = tone_mapping(hdr, tm_method)
        if ldr is not None:
            output_path = os.path.join(results_dir, f'{tm_method}_ldr{suffix}.jpg')
            cv2.imwrite(output_path, ldr)
            print(f"Guardada imagen LDR: {output_path}")
        else:
            print(f"Falló tone mapping con {tm_method}")
        

    # Graficar curvas de respuesta
    plt.figure(figsize=(12, 4))
    for i, color in enumerate(['b', 'g', 'r']):
        plt.subplot(1, 3, i+1)
        plt.plot(curve_list[i], color=color, label=f'Canal {["B", "G", "R"][i]}')
        plt.title(f'Canal {["B", "G", "R"][i]}')
        plt.xlabel('Valor de píxel (Z)')
        plt.ylabel('Log Exposición (g(Z))')
        plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, f'response_curves{suffix}.png'))
    plt.close()
    print(f"Curvas de respuesta guardadas en response_curves{suffix}.png")

if __name__ == '__main__':
    main()