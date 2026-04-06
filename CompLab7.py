import numpy as np
from scipy.optimize import brentq

# DADOS do problema 

t = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
y = np.array([6.2, 9.5, 12.3, 13.9, 14.6, 13.5, 13.3, 12.7, 12.4, 11.9], dtype=float)

# MODELO: y(t) = c1 * t * exp(c2 * t)
# Linearizando 
# ln(y) - ln(t) = ln(c1) + c2*t

b_lin = np.log(y) - np.log(t)
# Matriz A: coluna de 1s e coluna de t 
A = np.column_stack([np.ones(len(t)), t])
print("MATRIZ A:")
print(A)
print("\nVetor b linearizado (ln(y) - ln(t)):")
print(np.round(b_lin, 4))

# RESOLVENDO via np.linalg.lstsq 
coeffs, residuals, rank, s = np.linalg.lstsq(A, b_lin, rcond=None)
c1_til = coeffs[0]
c2 = coeffs[1]
c1 = np.exp(c1_til)
print("Solucao via np.linalg.lstsq:")
print(f" c~1 = ln(c1) = {c1_til:.4f}")
print(f" c2 = {c2:.4f}")
print(f"\nCoeficientes do modelo y = c1 * t * exp(c2 * t):")
print(f" c1 = {c1:.4f}")
print(f" c2 = {c2:.4f}")

# RESIDUOS e RMSE 

y_fit = c1 * t * np.exp(c2 * t)
r = y - y_fit
SE = np.sum(r**2)
RMSE = np.sqrt(SE / len(t))
print("Residuo r = b - Ax:")
print(np.round(r, 4))
print(f"SE = {SE:.4f}")
print(f"RMSE = {RMSE:.4f}")

# MAXIMO
# dy/dt = c1*exp(c2*t)*(1 + c2*t) = 0 => t_max = -1/c2

t_max = -1 / c2
y_max = c1 * t_max * np.exp(c2 * t_max)
print(f"Maximo estimado:")
print(f" t_max = {t_max:.4f} horas")
print(f" y_max = {y_max:.4f} ng/ml")

# MEIA-VIDA: y(t) = y_max/2, para t > t_max
# Usando brentq do scipy 

def f_half(tt):
 return c1 * tt * np.exp(c2 * tt) - y_max / 2
t_half = brentq(f_half, t_max, 100)
print(f"\nMeia-vida:")
print(f" t_half = {t_half:.4f} horas")
print(f" Tempo do pico ate meia-vida: {t_half - t_max:.4f} horas")

# FAIXA TERAPEUTICA: 4 <= y(t) <= 15
# Usando brentq do scipy para encontrar as raizes

def f_entra(tt):
 return c1 * tt * np.exp(c2 * tt) - 4
def f_sai(tt):
 return c1 * tt * np.exp(c2 * tt) - 4
t_entra = brentq(f_entra, 0.01, t_max)
t_sai = brentq(f_sai, t_max, 100)
print("Faixa terapeutica (4 - 15 ng/ml):")
print(f" Entra na faixa em: t = {t_entra:.4f} horas")
print(f" Sai da faixa em: t = {t_sai:.4f} horas")
print(f" Tempo dentro da faixa: {t_sai - t_entra:.4f} horas")