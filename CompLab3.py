import numpy as np
from scipy.optimize import brentq

'''
A fórmula de sensibilidade Δr ≈ -ε·g(r)/f'(r) estima o deslocamento da raiz
causado por uma pequena perturbação ε na equação. Quanto mais plana f for em r,
maior o deslocamento — indicando um problema mal-condicionado.
'''
def f_pert(x, pert):
    '''Função perturbada'''
    return (1 + pert)*x**3 - 3*x**2 + x - 3

def f(x):
    return x**3 - 3*x**2 + x - 3

def df(x):  
    return 3*x**2 - 6*x + 1

def g(x):
    return x**3
r = 3       
pert = 1e-3  

delta_r = -pert * g(r) / df(r)
r_aproximado = r + delta_r

r_numerico = brentq(f_pert, 2.5, 3.5, args=(pert,))

print(f"Raiz original: {r:.10f} | Δr: {delta_r:.10f} | Aprox: {r_aproximado:.10f} | Numérica: {r_numerico:.10f} | Erro: {abs(r_aproximado - r_numerico):.2e}")
