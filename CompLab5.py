import numpy as np

def gauss_eliminacao_pivoteamento(A, b):
    """
    Resolve o sistema linear Ax = b usando Eliminação de Gauss com Pivoteamento Parcial.
    """
    # Converte para arrays do tipo float para evitar truncamento em divisões
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(b)
    
    # Fase 1: Eliminação com Pivoteamento Parcial
    for k in range(n - 1):
        # 1.1 Identificar o pivô (maior valor absoluto na coluna k, da linha k para baixo)
        max_index = np.argmax(np.abs(A[k:n, k])) + k
        
        # 1.2 Trocar as linhas em A e b se o pivô não estiver na linha atual (k)
        if max_index != k:
            # Troca as linhas na matriz A
            A[[k, max_index]] = A[[max_index, k]]
            # Troca as linhas no vetor b
            b[[k, max_index]] = b[[max_index, k]]
            
        # Verifica se a matriz é singular (pivô zero após a troca)
        if abs(A[k, k]) < 1e-12:
            raise ValueError("Pivô nulo encontrado. O sistema não tem solução única.")
            
        # 1.3 Eliminação (Zerar os elementos abaixo do pivô)
        for i in range(k + 1, n):
            mult = A[i, k] / A[k, k]
            # Atualiza a linha i (somente da coluna k em diante para otimizar)
            A[i, k:] = A[i, k:] - mult * A[k, k:]
            # Atualiza o vetor b
            b[i] = b[i] - mult * b[k]
            
    # Fase 2: Substituição Reversa (Back substitution)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        if abs(A[i, i]) < 1e-12:
            raise ValueError("Elemento diagonal nulo encontrado durante a substituição.")
        
        # x_i = (b_i - soma(A_ij * x_j)) / A_ii
        x[i] = (b[i] - np.dot(A[i, i+1:], x[i+1:])) / A[i, i]
        
    return x

# ==========================================
# Testando o programa com o sistema fornecido
# ==========================================
#  x1 + 2x2 -  x3 = 4
# 5x1 + 2x2 + 2x3 = 18
# -3x1 + 5x2 -  x3 = 2

A_teste = [
    [1, 2, -1],
    [5, 2, 2],
    [-3, 5, -1]
]

b_teste = [4, 18, 2]

# Executando a função
solucao = gauss_eliminacao_pivoteamento(A_teste, b_teste)

print("Matriz A:")
print(np.array(A_teste))
print("\nVetor b:", b_teste)
print("-" * 30)
print(f"Solução encontrada: x1 = {solucao[0]:.0f}, x2 = {solucao[1]:.0f}, x3 = {solucao[2]:.0f}")

# Opcional: Verificação rápida usando o solver nativo do NumPy para validar o resultado
solucao_numpy = np.linalg.solve(A_teste, b_teste)
print(f"Solução nativa (np.linalg.solve): {solucao_numpy}")