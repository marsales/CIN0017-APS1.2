import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

v = np.array([10, 20, 30, 40, 50, 60, 70, 80], dtype=float)
F = np.array([25, 70, 380, 550, 610, 1220, 830, 1450], dtype=float)

print("CL1-8 - Regressao: Forca vs Velocidade do Vento")
print(f"{'v (m/s)':<10} {'F (N)':<10}")
print("-" * 20)
for vi, fi in zip(v, F):
    print(f"{vi:<10.0f} {fi:<10.0f}")


# =============================================================================
# (a) REGRESSAO LINEAR: F = a + b*v
# =============================================================================
print("\n" + "=" * 60)
print("(a) REGRESSAO LINEAR: F = a + b*v")
print("=" * 60)

A_lin = np.column_stack([np.ones_like(v), v])

coef_a, res_a, rank_a, sv_a = np.linalg.lstsq(A_lin, F, rcond=None)
a0, a1 = coef_a

F_pred_a = a0 + a1 * v

SS_res_a = np.sum((F - F_pred_a) ** 2)
SS_tot   = np.sum((F - np.mean(F)) ** 2)
R2_a     = 1 - SS_res_a / SS_tot

print(f"  Equacao:  F = {a0:.4f} + {a1:.4f}*v")
print(f"  R2      = {R2_a:.6f}")
print(f"  ||e||   = {np.sqrt(SS_res_a):.4f} N")


# =============================================================================
# (b) MODELO DE POTENCIA VIA LOG: F = c * v^alpha
# =============================================================================
print("\n" + "=" * 60)
print("(b) MODELO DE POTENCIA VIA TRANSFORMACAO LOG: F = c * v^alpha")
print("=" * 60)

ln_v = np.log(v)
ln_F = np.log(F)

A_log = np.column_stack([np.ones_like(ln_v), ln_v])
coef_b, _, _, _ = np.linalg.lstsq(A_log, ln_F, rcond=None)
ln_c, alpha_b = coef_b
c_b = np.exp(ln_c)

F_pred_b = c_b * v ** alpha_b

SS_res_b = np.sum((F - F_pred_b) ** 2)
R2_b     = 1 - SS_res_b / SS_tot

print(f"  Equacao:  F = {c_b:.6f} * v^{alpha_b:.6f}")
print(f"  R2      = {R2_b:.6f}")
print(f"  ||e||   = {np.sqrt(SS_res_b):.4f} N")


# =============================================================================
# (c) REGRESSAO NAO-LINEAR: F = c * v^alpha  (Levenberg-Marquardt)
# =============================================================================
print("\n" + "=" * 60)
print("(c) REGRESSAO NAO-LINEAR: F = c * v^alpha")
print("    Metodo: Levenberg-Marquardt")
print("=" * 60)

def residual(params):
    c, alpha = params
    return c * v ** alpha - F

x0 = np.array([c_b, alpha_b])

sol = least_squares(residual, x0, method='lm')
c_nl, alpha_nl = sol.x

F_pred_c = c_nl * v ** alpha_nl

SS_res_c = np.sum((F - F_pred_c) ** 2)
R2_c     = 1 - SS_res_c / SS_tot

print(f"  Equacao:  F = {c_nl:.6f} * v^{alpha_nl:.6f}")
print(f"  R2      = {R2_c:.6f}")
print(f"  ||e||   = {np.linalg.norm(sol.fun):.4f} N")


# =============================================================================
# COMPARACAO FINAL
# =============================================================================
print("\n" + "=" * 60)
print("COMPARACAO DOS MODELOS")
print("=" * 60)
print(f"{'Modelo':<30} {'R2':>10} {'||e|| (N)':>14}")
print("-" * 56)
print(f"{'(a) Reta':<30} {R2_a:>10.6f} {np.sqrt(SS_res_a):>14.4f}")
print(f"{'(b) Potencia (log)':<30} {R2_b:>10.6f} {np.sqrt(SS_res_b):>14.4f}")
print(f"{'(c) Potencia (nao-linear)':<30} {R2_c:>10.6f} {np.linalg.norm(sol.fun):>14.4f}")


# =============================================================================
# GRAFICO
# =============================================================================
v_plot = np.linspace(8, 82, 400)

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
fig.suptitle(
    "CL1-8 - Regressao: Forca vs Velocidade do Vento\n"
    "Complementos de Matematica - Aula 2-8 (Least Squares II)",
    fontsize=13, fontweight='bold'
)

cores = ['#2196F3', '#4CAF50', '#F44336']
labels_modelo = [
    f"(a) Reta\nF = {a0:.1f} + {a1:.2f}*v\nR2 = {R2_a:.4f}",
    f"(b) Potencia (log)\nF = {c_b:.3f}*v^{alpha_b:.3f}\nR2 = {R2_b:.4f}",
    f"(c) Gauss-Newton\nF = {c_nl:.3f}*v^{alpha_nl:.3f}\nR2 = {R2_c:.4f}",
]
curvas = [
    a0 + a1 * v_plot,
    c_b * v_plot ** alpha_b,
    c_nl * v_plot ** alpha_nl,
]

for ax, cor, label, curva in zip(axes, cores, labels_modelo, curvas):
    ax.scatter(v, F, color='black', zorder=5, label='Dados', s=60)
    ax.plot(v_plot, curva, color=cor, linewidth=2.5, label=label)
    ax.set_xlabel("Velocidade v (m/s)", fontsize=11)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(5, 85)

axes[0].set_ylabel("Forca F (N)", fontsize=11)

plt.tight_layout()
plt.savefig("cl1_8_grafico.png", dpi=150, bbox_inches='tight')
plt.show()