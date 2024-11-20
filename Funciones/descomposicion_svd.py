import numpy as np 
from tabulate import tabulate 

## Algoritmo de Gauss 
##  Necesario en: Potencia inverso 
def algoritmo_gauss(A, b):
    Ab = np.hstack([A, b.reshape(-1, 1)])
    n = Ab.shape[0] 

    
    for i in range(n):
        for j in range(i + 1, n):
            
            if Ab[i, i] == 0: 
                raise ValueError("No se puede realizar la division por 0") 
                return 
                
            cociente = Ab[j, i] / Ab[i, i]
            Ab[j] -= cociente * Ab[i]

    x = np.zeros(n)

   
    for i in range(n - 1, -1, -1):
        cumulate = Ab[i, -1] 
        for j in range(i + 1, n):
            cumulate -= Ab[i, j] * x[j]  
        x[i] = cumulate / Ab[i, i]  

    return Ab, x 

## Algoritmo de la potencia inversa
##  Necesario en SVD  

def met_potencia_inverso(A, v, tol=1e-10, N=1500, val_delta = None):  
    A = np.asarray(A, dtype=np.float64) 
    v = np.asarray(v, dtype=np.float64)  

    if len(A.shape) != 2: 
        raise ValueError("La matriz A debe ser bidimensional.")

    if A.shape[0] != A.shape[1]: 
        raise ValueError("La matriz A debe ser cuadrada.")
    
    n = A.shape[0] 

    if v.ndim != 1 or v.size != n:
        raise ValueError("v debe ser un vector de tamaño 1xN, donde N es el número de filas de A.")

    #Normalizacion del vector inicial 
    v_k = v/np.linalg.norm(v) 
    delta = val_delta  if val_delta is not None else np.random.rand()

    error_arr = []
    mu_arr = []

    for _ in range(N):     
        ## Calculamos la matriz B = (A -\lambda I)  
        B = A - delta * np.eye(n) 

        ## Resolvemos el sistema (A - \lambda I)x = v_k 
        _, x = algoritmo_gauss(B.copy(), v_k)   

        #Normalizamos el candidato 
        x = x/np.linalg.norm(x) 

        omega = v_k/np.linalg.norm(x)

        rho = x.T @ omega 

        mu = delta + rho 
        mu_arr.append(mu) 

        r = omega - rho*x 

        error = np.linalg.norm(x - v_k)
        error_arr.append(error)

        if error < tol:  
            break 

        v_k = x 
        
    return mu, v_k 
        
    
## Descomposicion QR
##      Necesaria en obtencion de eigenvalores de la matriz 
def fac_QR_householder(A): 
    A = np.asarray(A)  
    m, n = A.shape
    Q = np.eye(m)
    R = A.copy() 

    for j in np.arange(n): 
        x = R[j:m, j] 
        e = np.zeros(m-j)
        e[0] = np.linalg.norm(x) 
        v = x - e 

        v = v / np.linalg.norm(v) 

        H = np.eye(m)
        H[j:m, j:m] -= 2 * np.outer(v, v)

        R = H @ R  
        Q = Q @ H.T 

    return Q, R 



## Obtencion de eigenvalores de la matriz 
##      Necesaria en descomposicion SVD 


def eig_mat(A, tau=1e-9, max_iter = 50): 
    k = 0 
    n_A_size = len(A) 
    V = np.eye(n_A_size)  

    print_indexes = [3, 10, 20, 34] 
    A_k1 = A.copy() 

    flag = True 

    for k in range(max_iter):
        Q_k, R_k = fac_QR_householder(A_k1) 

        Q_k = np.asarray(Q_k)
        R_k = np.asarray(R_k) 

        A_k1 = np.asarray(R_k @ Q_k)
        V = np.asarray(V @ Q_k)
        if k in print_indexes: 
            print("A_", k, '\n', tabulate(A_k1), end='\n') 
        
        Q_diag = np.diag(Q_k) 
        Q_no_diag = Q_k - np.diag(np.diag(Q_k))  

        flag_diag = np.all(np.abs(1 - np.abs(Q_diag)) < tau) 
        flag_no_diag = np.all(np.abs(Q_no_diag) < tau) 

        flag = flag_diag and flag_no_diag 

        if flag: 
            break 

    return np.diag(A_k1), V


## Función principal de la descomposición SVD
def svd_decomposition(A):
    m, n = A.shape


    Q1, R = fac_QR_householder(A)
    Q2, B = fac_QR_householder(R.T)  
    B = B.T  


    singular_values, V = eig_mat(B)

  
    U = Q1 @ Q2.T

    Sigma = np.zeros((m, n))
    np.fill_diagonal(Sigma, singular_values)

    V_t = V.T

    return U, Sigma, V_t
