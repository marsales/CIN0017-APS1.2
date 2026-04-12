import math
import matplotlib.pyplot as plt
import numpy as np

R = 4
V_desejado = 50

def f_newton(h):
    return (math.pi * h**2 / 3) * (3*R - h) - V_desejado

def df_newton(h):
    return math.pi * h * (2*R - h)

def method_Newton(f_newton, df_newton, X_inicial, Tolerancia, Max_Iteracoes):
    ''' Para o problema do tanque esférico:
    1 - Colocar o chute inicial na reta tangente
    2 - Encontrar o novo ponto
    3 - Repetir até satisfazer a condição
    '''
    func_x = f_newton(X_inicial)
    der_func_x = df_newton(X_inicial)

    for i in range(Max_Iteracoes):
        Xn = X_inicial - func_x / der_func_x
        
        # erro relativo (%)
        error = abs((Xn - X_inicial) / Xn) * 100

        if error <= Tolerancia:
            print(f'A raiz aproximada é: {Xn:.8f}')
            break

        print(f'i: {i+1} | error: {error:.6f}% | X: {Xn:.8f}')
        
        X_inicial = Xn
        func_x = f_newton(X_inicial)
        der_func_x = df_newton(X_inicial)

method_Newton(f_newton, df_newton, 2, 0, 3)
