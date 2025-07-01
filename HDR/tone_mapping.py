import cv2
import numpy as np
import math
from scipy.ndimage import gaussian_filter1d
import scipy.signal
import scipy.interpolate

def bilateral_approximation(data, edge, sigmaS, sigmaR):
    """
    Realiza una aproximación bilateral para suavizado de imágenes.
    Args:
        data: Imagen de entrada (matriz 2D).
        edge: Imagen de bordes para guiar el filtro.
        sigmaS: Desviación estándar espacial.
        sigmaR: Desviación estándar de rango.
    Returns:
        Imagen suavizada.
    """
    inputHeight, inputWidth = data.shape
    edgeMax = np.amax(edge)
    edgeMin = np.amin(edge)

    paddingXY = 5
    paddingZ = 5

    downsampledWidth = math.floor((inputWidth - 1) / sigmaS) + 2 * paddingXY + 1
    downsampledHeight = math.floor((inputHeight - 1) / sigmaS) + 2 * paddingXY + 1
    downsampledDepth = math.floor((edgeMax - edgeMin) / sigmaR) + 2 * paddingZ + 1

    gridData = np.zeros((downsampledHeight, downsampledWidth, downsampledDepth))
    gridWeights = np.zeros((downsampledHeight, downsampledWidth, downsampledDepth))

    (jj, ii) = np.meshgrid(range(inputWidth), range(inputHeight))
    di = np.around(ii / sigmaS) + paddingXY + 1
    dj = np.around(jj / sigmaS) + paddingXY + 1
    dz = np.around((edge - edgeMin) / sigmaR) + paddingZ + 1

    for k in range(0, dz.size):
        gridData[int(di.flat[k]), int(dj.flat[k]), int(dz.flat[k])] += data.flat[k]
        gridWeights[int(di.flat[k]), int(dj.flat[k]), int(dz.flat[k])] += 1

    kernelWidth = 5
    kernelHeight = 5
    kernelDepth = 5

    (gridX, gridY, gridZ) = np.meshgrid(range(0, int(kernelWidth)), range(0, int(kernelHeight)), range(0, int(kernelDepth)))
    gridX -= math.floor(kernelWidth / 2)
    gridY -= math.floor(kernelHeight / 2)
    gridZ -= math.floor(kernelDepth / 2)
    gridRSquared = (gridX * gridX + gridY * gridY) + (gridZ * gridZ)
    kernel = np.exp(-0.5 * gridRSquared)

    blurredGridWeights = scipy.signal.fftconvolve(gridWeights, kernel, mode='same')
    blurredGrid = scipy.signal.fftconvolve(gridData, kernel, mode='same')

    blurredGridWeights = np.where(blurredGridWeights == 0, 0.1, blurredGridWeights)
    normalizedBlurredGrid = blurredGrid / blurredGridWeights
    normalizedBlurredGrid = np.where(blurredGridWeights < -1, 0, normalizedBlurredGrid)

    (jj, ii) = np.meshgrid(range(0, inputWidth), range(0, inputHeight))
    di = (ii / sigmaS) + paddingXY + 1
    dj = (jj / sigmaS) + paddingXY + 1
    dz = (edge - edgeMin) / sigmaR + paddingZ + 1

    return scipy.interpolate.interpn(
        (range(0, normalizedBlurredGrid.shape[0]), range(0, normalizedBlurredGrid.shape[1]), range(0, normalizedBlurredGrid.shape[2])),
        normalizedBlurredGrid,
        (di, dj, dz)
    )

def gammaCorrection(x):
    """
    Aplica corrección gamma a un valor.
    Args:
        x: Valor de entrada.
    Returns:
        Valor corregido.
    """
    if x <= 0.0031308:
        return 12.92 * x
    return 1.055 * x ** (1 / 2.2) - 0.055

def durandanddorsy(img):
    """
    Aplica tone mapping usando el método de Durand y Dorsey.
    Args:
        img: Imagen HDR en formato RGB (float32).
    Returns:
        Imagen LDR en formato uint8.
    """
    height, width = img.shape[:2]
    img = img / np.max(img)  # Normalizar para evitar valores extremos
    epsilon = 0.000001
    val = 0.114 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.299 * img[:, :, 2] + epsilon
    log_val = np.log(val)

    space_sigma = min(width, height) / 16
    range_sigma = (np.amax(log_val) - np.amin(log_val)) / 10
    imgg = bilateral_approximation(log_val, log_val, space_sigma, range_sigma)
    gamma = 0.45
    detailed_channel = np.exp(gamma * imgg + np.subtract(log_val, imgg))
    detailed_channel = detailed_channel.astype('float32')

    out = np.zeros(img.shape)
    out[:, :, 0] = img[:, :, 0] * (detailed_channel / val)
    out[:, :, 1] = img[:, :, 1] * (detailed_channel / val)
    out[:, :, 2] = img[:, :, 2] * (detailed_channel / val)

    gammafun = np.vectorize(gammaCorrection)
    outt = np.clip(gammafun(out) * 255, 0, 255).astype('uint8')
    return outt

def tone_mapping(hdr, method='global', d=1e-5, a=0.5):
    """
    Aplica tone mapping a una imagen HDR.
    Args:
        hdr: Imagen HDR en formato float32.
        method: Método de tone mapping ('global', 'Drago', 'Reinhard', 'Mantiuk', 'durandanddorsy').
        d: Valor pequeño para evitar log(0) en el método global.
        a: Parámetro de ajuste para el método global.
    Returns:
        Imagen LDR en formato uint8 o None si el método no es soportado.
    """
    # Normalizar HDR para evitar valores no válidos
    hdr = np.nan_to_num(hdr, nan=0.0, posinf=np.max(hdr[np.isfinite(hdr)]), neginf=0.0)
    
    if method == 'global':
        Lw = hdr
        Lw_bar = np.exp(np.mean(np.log(d + Lw)))
        Lm = (a / Lw_bar) * Lw
        Lm_white = np.max(Lm)
        Ld = (Lm * (1 + (Lm / (Lm_white ** 2)))) / (1 + Lm)
        ldr = np.clip(np.array(Ld * 255), 0, 255).astype(np.uint8)
    elif method == 'Drago':
        tonemapDrago = cv2.createTonemapDrago(1.0, 0.7)
        ldr = tonemapDrago.process(hdr)
        ldr = np.clip(ldr * 255, 0, 255).astype(np.uint8)
    elif method == 'Reinhard':
        tonemapReinhard = cv2.createTonemapReinhard(1.5, 0, 0, 0)
        ldr = tonemapReinhard.process(hdr)
        ldr = np.clip(ldr * 255, 0, 255).astype(np.uint8)
    elif method == 'Mantiuk':
        tonemapMantiuk = cv2.createTonemapMantiuk(gamma=2.2, scale=0.85, saturation=1.2)
        ldr = tonemapMantiuk.process(hdr)
        ldr = np.clip(ldr * 255, 0, 255).astype(np.uint8)
    elif method == 'durandanddorsy':
        ldr = durandanddorsy(hdr)
    else:
        print('Método de tone mapping no soportado.')
        return None
    return ldr

def save_hdr(hdr, filename):
    """
    Guarda una imagen HDR en formato RADIANCE (.hdr).
    Args:
        hdr: Imagen HDR en formato float32.
        filename: Ruta del archivo de salida.
    """
    image = np.zeros((hdr.shape[0], hdr.shape[1], 3), 'float32')
    image[..., 0] = hdr[..., 2]  # R
    image[..., 1] = hdr[..., 1]  # G
    image[..., 2] = hdr[..., 0]  # B
    f = open(filename, 'wb')
    f.write(b"#?RADIANCE\n# Made with Python & Numpy\nFORMAT=32-bit_rle_rgbe\n\n")
    header = f'-Y {image.shape[0]} +X {image.shape[1]}\n'
    f.write(bytes(header, encoding='utf-8'))
    brightest = np.maximum(np.maximum(image[..., 0], image[..., 1]), image[..., 2])
    mantissa = np.zeros_like(brightest)
    exponent = np.zeros_like(brightest)
    np.frexp(brightest, mantissa, exponent)
    scaled_mantissa = mantissa * 256.0 / brightest
    rgbe = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
    rgbe[..., 0:3] = np.around(image[..., 0:3] * scaled_mantissa[..., None])
    rgbe[..., 3] = np.around(exponent + 128)
    rgbe.flatten().tofile(f)
    f.close()