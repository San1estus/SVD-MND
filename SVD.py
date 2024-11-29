import numpy as np

def svd(A):
    m, n = A.shape
    if np.all(A == np.zeros((m,n))):
       return np.zeros((m,n)), np.zeros((m,n)), np.zeros((n,n))

    # Paso 1: Calcular A^T A y A A^T.
    ata = A.T @ A
    aat = A @ A.T

    # Paso 2: Calcular valores y vectores propios.
    valores_ata, vectores_ata = valores_vectores_propios(ata)
    valores_aat, vectores_aat = valores_vectores_propios(aat)

    # Paso 3: Ordenar los valores y vectores propios en orden descendente.
    orden_ata = np.argsort(valores_ata)[::-1]
    orden_aat = np.argsort(valores_aat)[::-1]

    valores_singulares = np.sqrt(np.maximum(0, valores_aat[orden_aat]))  # Los valores singulares son la raíz de los valores propios.
    vectores_ata = vectores_ata[:, orden_ata]
    vectores_aat = vectores_aat[:, orden_aat]

    # Paso 4: Selección de vectores propios y construcción de matrices
    U = vectores_aat[:, :m]
    Sigma = np.zeros((m, n))
    np.fill_diagonal(Sigma, valores_singulares[:min(m, n)])
    V = vectores_ata

    # Paso 5: Corregir signos en V
    U, Sigma, V = svd_signo(U, Sigma, V, A)
    return U, Sigma, V.T



def svd_signo(U, Sigma, V, A):
    m, n = A.shape

    U_interes = U[:, :min(m, n)]
    Sigma_fix = Sigma[:min(m, n), :min(m, n)]

    for i in range(min(V.shape[1], U.shape[1])):
        # Reconstrucción parcial para verificar consistencia
        if not np.allclose(A @ V[:, i], U_interes @ (Sigma_fix[:, i]), atol=1e-6):
            V[:, i] *= -1  # Invertimos el signo del vector si es necesario

    return U, Sigma, V


def matrizHouseholder(a):
    e = np.zeros_like(a)
    e[0] = np.linalg.norm(a)
    v = a + np.sign(a[0]) * e
    if np.linalg.norm(v) < 1e-8:
        return np.eye(len(a))
    v = v / np.linalg.norm(v)
    H = np.eye(len(a)) - 2 * np.outer(v, v)
    return H



def factorizacionQR(A):
    m, n = A.shape
    Q = np.eye(m)
    R = A.copy()

    for k in range(n):
        # Crear H_k
        Hk = np.eye(m)
        H = matrizHouseholder(R[k:, k])
        Hk[k:, k:] = H

        # Aplicamos Hk a R y acumulamos en Q
        R = Hk @ R
        Q = Q @ Hk

    return Q, R


def descomposicion_schur(A, max_iter=1000, tol=1e-9):
    n = A.shape[0]
    Q_total = np.eye(n)
    T = A.copy()

    for _ in range(max_iter):
        Q, R = factorizacionQR(T)
        T = R @ Q
        Q_total = Q_total @ Q

        if np.allclose(T[np.tril_indices(n, -1)], 0, atol=tol):
            break

    return Q_total, T


def valores_vectores_propios(A):
    Q, T = descomposicion_schur(A)
    valores_propios = np.diag(T)  # Valores propios en la diagonal de T
    vectores_propios = Q
    return valores_propios, vectores_propios