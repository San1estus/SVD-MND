from SVD import svd
import numpy as np

#Compara los resultados obtenido por nuestra implementacion de la SVD encontra de la implementacion de numpy

def comparar_svd(codigo_personalizado, matriz):
    # Resultado de la implementación personalizada
    U_custom, Sigma_custom, Vt_custom = codigo_personalizado(matriz)

    # Resultado de numpy.linalg.svd
    U_np, Sigma_np, Vt_np = np.linalg.svd(matriz)

    D1 = np.zeros((len(matriz),len(matriz[0])))
    for i in range(len(Sigma_np)):
      D1[i,i] += Sigma_np[i]

    # Reconstrucción de A para ambos casos
    A_reconstruida_np = U_np @ D1 @ Vt_np
    A_reconstruida_custom = U_custom @ Sigma_custom @ Vt_custom

    print("Comparación para la matriz:")
    print(matriz)
    print("\nReconstrucción personalizada:")
    print(A_reconstruida_custom)
    print("\nReconstrucción con numpy:")
    print(A_reconstruida_np)

    print("\nReconstrucción de A:")
    print(np.allclose(A_reconstruida_custom, matriz, atol=1e-6))
    print("-" * 50)

# Casos de prueba
matrices_prueba = [
    np.array([[1, 0], [0, 1]]),                          # Matriz identidad
    np.array([[3, 1], [2, 4]]),                          # Matriz 2x2 simple
    np.array([[2, 3, 4, 5, 6], [4, 4, 5, 6, 7], [0, 3, 6, 7, 8], [0, 0, 2, 8, 9], [0, 0, 0, 1, 10]]), #Matriz 5x5 (Tarea 7)
    np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),         # Matriz singular
    np.random.rand(4, 4),                                # Matriz aleatoria 4x4
    np.random.rand(5, 3),                                # Matriz no cuadrada (más filas que columnas)
    np.random.rand(3, 5),                                # Matriz no cuadrada (más columnas que filas)
    np.random.rand(10, 10),                              # Matriz grande
    np.random.rand(50, 50),                              # Matriz grande
    np.diag([10, 1, 0.1, 0.01]),                         # Matriz diagonal con valores decrecientes
    np.zeros((3, 3)),                                    # Matriz nula
]

for matriz in matrices_prueba:
    comparar_svd(svd, matriz)

