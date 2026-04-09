import math
import matplotlib.pyplot as plt
import numpy as np

# X0 = Chute inicial
# Xi+1 = Xi - (f(Xi)/f'(Xi))

def f_newton(x):
    return x**3 + x - 1

def df_newton(x):
    return 3*x**2 + 1

def rf_fixed_point_convergente(x):
    return 1 / (x**2 + 1)

def method_Newton(f_newton, df_newton, X_inicial, Tolerancia, Max_Iteracoes):
    ''' Para a equação x^3 + x + 1 Faremos:
    1 - Colcar o chute inicial na reta tangente e ver onde ela cruza o eixo x
    2 - Colocar o novo chute na reta tangente e ver onde ela cruza o eixo x
    3 - e seguir nos passos 1 e 2 até que a condição de parada seja satisfeita
    '''
    func_x = f_newton(X_inicial)
    der_func_x = df_newton(X_inicial)

    for i in range(Max_Iteracoes):
        Xn = X_inicial - func_x/der_func_x
        error = abs(Xn - X_inicial)

        if error <= Tolerancia:
            print(f'A raiz aproximada é: {Xn:.8f}')
            break

        print(f'i: {i} | error: {error:.2e} | X: {Xn:.8f}')
        X_inicial = Xn
        func_x = f_newton(X_inicial)
        der_func_x = df_newton(X_inicial)


def method_fixed_point(f_fixed_point, rf_fixed_point, X_inicial, Tolerancia, Max_Iteracoes):
    for i in range(Max_Iteracoes):
        X1 = rf_fixed_point(X_inicial)
        error = abs(X1 - X_inicial)

        print(f'i: {i} | error: {error:.2e} | X: {X1:.8f}')

        if error <= Tolerancia:
            print(f'\nA raiz aproximada é: {X1:.8f}')
            break

        X_inicial = X1
    else:
        print(f'Não convergiu em {Max_Iteracoes} iterações. Último X: {X1:.8f}')

method_Newton(f_newton, df_newton, 0, 0.000000000000000001, 10)
method_fixed_point(f_fixed_point, rf_fixed_point_convergente, 0.5, 0.001, 20)




  
