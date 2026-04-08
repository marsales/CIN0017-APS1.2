import numpy as np

def gauss_seidel(A, b, max_iter=50, tol=1e-6, lim_divergencia=1e10):
    """
    Tenta resolver Ax = b usando Gauss-Seidel.
    Retorna uma tupla: (Sucesso/Falha, Numero_Iteracoes, Mensagem)
    """
    n = len(b)
    x = np.zeros(n)
    
    # Critério de Falha 1: Zero na diagonal principal
    for i in range(n):
        if A[i, i] == 0:
            return False, 0, "Falha imediata: Divisão por zero (elemento da diagonal é 0)."
            
    for k in range(1, max_iter + 1):
        x_old = x.copy()
        
        for i in range(n):
            # Soma dos elementos já calculados na iteração atual (k)
            soma_atual = np.dot(A[i, :i], x[:i])
            # Soma dos elementos da iteração anterior (k-1)
            soma_ant = np.dot(A[i, i+1:], x_old[i+1:])
            
            # Fórmula de Gauss-Seidel
            x[i] = (b[i] - soma_atual - soma_ant) / A[i, i]
            
        # Cálculo do Erro: Norma infinita (maior diferença absoluta)
        erro = np.max(np.abs(x - x_old))
        
        # Critério de Convergência
        if erro < tol:
            return True, k, f"Convergiu com sucesso."
            
        # Critério de Falha 2: Divergência (valores explodiram)
        if erro > lim_divergencia or np.any(np.isnan(x)) or np.any(np.isinf(x)):
            return False, k, f"Divergiu criticamente (Erro atingiu {erro:.2e})."
            
    return False, max_iter, "Não convergiu dentro do limite máximo de iterações."

# Definição dos três sistemas da imagem

# Set One: Note o 0 na posição A[1,1] (pois não há 'y' na 2ª equação)
A1 = np.array([[9, 3, 1], 
               [-6, 0, 8], 
               [2, 5, -1]], dtype=float)
b1 = np.array([13, 8, 6], dtype=float)

# Set Two
A2 = np.array([[1, 1, 6], 
               [1, 5, -1], 
               [4, 2, -2]], dtype=float)
b2 = np.array([8, 5, 4], dtype=float)

# Set Three
A3 = np.array([[-3, 4, 5], 
               [-2, 2, -4], 
               [0, 2, -1]], dtype=float)
b3 = np.array([6, -3, 1], dtype=float)

# Execução dos Testes
sistemas = [("Set One", A1, b1), ("Set Two", A2, b2), ("Set Three", A3, b3)]

for nome, A, b in sistemas:
    print(f"--- {nome} ---")
    sucesso, iteracoes, msg = gauss_seidel(A, b)
    print(f"Iterações: {iteracoes}")
    print(f"Resultado: {msg}\n")