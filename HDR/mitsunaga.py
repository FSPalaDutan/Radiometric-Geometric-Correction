import numpy as np
import math
import cv2
from debevec import response_curve_Debevec  # Necesitas tener esta función implementada
from robertson import robertson_method      # Necesitas tener esta función implementada

def solve_c(M, R, N, I_max=1.0):
    """
    Resuelve los coeficientes del polinomio c_n para un orden N dado las razones de exposición R.
    """
    Q = len(M)  # Número de imágenes
    P = len(M[0])  # Número de píxeles
    rows = []
    targets = []
    
    for q in range(Q - 1):
        R_q = R[q]
        for p in range(P):
            M_pq = M[q][p]
            M_pq1 = M[q + 1][p]
            a = []
            for n in range(N):
                a_n = (M_pq ** n - R_q * M_pq1 ** n) - (M_pq ** N - R_q * M_pq1 ** N)
                a.append(a_n)
            b_pq = I_max * (M_pq ** N - R_q * M_pq1 ** N)
            rows.append(a)
            targets.append(-b_pq)
    
    A = np.array(rows, dtype=np.float32)
    b = np.array(targets, dtype=np.float32)
    
    # Resolver el sistema de mínimos cuadrados
    c_prime, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    c = list(c_prime)
    
    # Calcular c_N usando la restricción f(1) = I_max
    s = sum(c)
    c_N = I_max - s
    c.append(c_N)
    
    return c

def compute_error(c, M, R, N, I_max=1.0):
    """
    Calcula el error para los coeficientes c dados.
    """
    Q = len(M)
    P = len(M[0])
    error = 0
    for q in range(Q - 1):
        R_q = R[q]
        for p in range(P):
            M_pq = M[q][p]
            M_pq1 = M[q + 1][p]
            f_pq = sum(c[n] * M_pq ** n for n in range(N + 1))
            f_pq1 = sum(c[n] * M_pq1 ** n for n in range(N + 1))
            if f_pq1 > 1e-6:
                ratio = f_pq / f_pq1
                error += (ratio - R_q) ** 2
    return error

def response_curve_Mitsunaga(imgs, exposures, reference_method='debevec', N_range=(7, 10), max_iter=55, tol=1e-4):
    """
    Calcula la curva de respuesta logarítmica usando el método de Mitsunaga.

    Args:
        imgs: Lista de imágenes para un canal específico.
        exposures: Lista de tiempos de exposición.
        reference_method: 'debevec' o 'robertson' para elegir la curva de referencia.
        N_range: Rango de grados del polinomio a probar.
        max_iter: Número máximo de iteraciones.
        tol: Tolerancia para la convergencia.
    Returns:
        g_scaled: Curva de respuesta ajustada y escalada al rango de la referencia (vector de 256 elementos).
    """
    # Seleccionar la curva de referencia
    if reference_method == 'debevec':
        reference_curve = response_curve_Debevec(imgs, exposures)[0]  # Tomamos solo g
    elif reference_method == 'robertson':
        reference_curve = robertson_method([imgs], exposures)[0]  # Solo un canal
    else:
        raise ValueError("Método no válido. Usa 'debevec' o 'robertson'.")
    
    # Asegurarse de que reference_curve sea un vector unidimensional
    reference_curve = np.array(reference_curve).flatten()
    if len(reference_curve) != 256:
        raise ValueError(f"reference_curve debe tener 256 elementos, tiene {len(reference_curve)}")
    
    # Ordenar imágenes y exposiciones según tiempos de exposición
    sorted_indices = np.argsort(exposures)
    sorted_exposures = [exposures[i] for i in sorted_indices]
    sorted_imgs = [imgs[i] for i in sorted_indices]
    
    # Redimensionar imágenes para cálculos más rápidos
    Z = [cv2.resize(img, (25, 25)).flatten() / 255.0 for img in sorted_imgs]
    Q = len(Z)  # Número de imágenes
    P = len(Z[0])  # Número de píxeles
    
    # Calcular razones de exposición iniciales
    R = [sorted_exposures[q] / sorted_exposures[q + 1] for q in range(Q - 1)]
    
    # Lista para almacenar errores y coeficientes
    errors = []
    c_list = []
    
    # Probar diferentes grados del polinomio
    for N in range(N_range[0], N_range[1] + 1):
        c_prev = None
        I_max = 1.0
        current_R = R.copy()
        
        for k in range(max_iter):
            # Resolver coeficientes
            c = solve_c(Z, current_R, N, I_max)
            
            # Definir función de respuesta
            def f(M):
                return sum(c[n] * M ** n for n in range(N + 1))
            
            # Actualizar razones de exposición
            new_R = []
            for q in range(Q - 1):
                sum_ratio = 0
                count = 0
                for p in range(P):
                    M_pq = Z[q][p]
                    M_pq1 = Z[q + 1][p]
                    f_pq1 = f(M_pq1)
                    if f_pq1 > 1e-6:
                        ratio = f(M_pq) / f_pq1
                        sum_ratio += ratio
                        count += 1
                R_q = sum_ratio / count if count > 0 else current_R[q]
                new_R.append(R_q)
            
            # Verificar convergencia
            if c_prev is not None:
                diff = max(abs(c[n] - c_prev[n]) for n in range(N + 1))
                if diff < tol:
                    break
            c_prev = c.copy()
            current_R = new_R
        
        # Calcular error
        error = compute_error(c, Z, R, N, I_max)
        
        # Almacenar N, error y coeficientes
        errors.append((N, error))
        c_list.append(c)
    
    # Filtrar N con error > 100
    valid_errors = [e for e in errors if e[1] > 100]
    
    if not valid_errors:
        raise ValueError("Ningún N tiene un error mayor a 100.")
    
    # Seleccionar el N con el menor error entre los que cumplen la condición
    best_N, best_error = min(valid_errors, key=lambda x: x[1])
    best_c = c_list[best_N - N_range[0]]  # Ajustar índice para obtener los coeficientes correspondientes
    
    print(f"Mejor N: {best_N} con error: {best_error}")
    
    # Calcular g(Z) = ln(f(Z / 255)) para Z de 0 a 255
    g = []
    for Z in range(256):
        M = Z / 255.0
        f_M = sum(best_c[n] * M ** n for n in range(best_N + 1))
        f_M = max(f_M, 1e-6)  # Evitar logaritmo de cero
        g.append(math.log(f_M))
    
    # Ajustar para que g(127) = 0
    offset = g[127]
    g_adjusted = [g_z - offset for g_z in g]
    
    min_ref = min(reference_curve)
    max_ref = max(reference_curve)
    min_g = min(g_adjusted)
    max_g = max(g_adjusted)
    range_g = max_g - min_g if max_g != min_g else 1e-6  # Evitar división por cero
    g_scaled = [(min_ref + (max_ref - min_ref) * (g_z - min_g) / range_g) for g_z in g_adjusted]
    
    return g_scaled