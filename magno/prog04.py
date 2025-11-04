import numpy as np #type: ignore
import matplotlib.pyplot as plt #type: ignore
from scipy.linalg import toeplitz #type: ignore
from scipy.stats import skew #type: ignore

# ==========================================================
# 1️⃣ Função Yule–Walker com pₘ variável
# ==========================================================
def yule_walker_par_variable_p(Z, p_by_month, n_meses=12):
    """
    Estima modelos PAR(pₘ) com ordem variável por mês usando equações de Yule–Walker.
    """
    phis_by_month = {}
    sigma2_by_month = {}
    gamma_by_month = {}

    for m in range(1, n_meses + 1):
        p_m = p_by_month.get(m, 1)
        Z_m = Z[(m - 1)::n_meses]
        N = len(Z_m)
        mu_m = np.mean(Z_m)

        # autocovariâncias γ_k até lag pₘ
        gamma = np.array([
            np.sum((Z_m[:N - k] - mu_m) * (Z_m[k:] - mu_m)) / N
            for k in range(p_m + 1)
        ])
        gamma_by_month[m] = gamma.copy()

        if p_m == 0:
            phis_by_month[m] = np.array([])
            sigma2_by_month[m] = gamma[0]
            continue

        R = toeplitz(gamma[:-1])
        r = gamma[1:]
        phi = np.linalg.solve(R, r)
        sigma2 = gamma[0] - np.dot(phi, r)

        phis_by_month[m] = phi
        sigma2_by_month[m] = sigma2

    return phis_by_month, sigma2_by_month, gamma_by_month


# ==========================================================
# 2️⃣ Visualização da matriz de autocovariâncias Yule–Walker
# ==========================================================
def plot_yule_walker_matrix(gamma_by_month):
    """
    Plota a matriz de autocovariâncias (Yule–Walker) para cada mês.
    Cada linha representa as autocovariâncias até a ordem pₘ.
    """
    max_p = max(len(g) - 1 for g in gamma_by_month.values())
    M = len(gamma_by_month)
    G = np.full((M, max_p + 1), np.nan)

    for m, gamma in gamma_by_month.items():
        G[m - 1, :len(gamma)] = gamma

    plt.figure(figsize=(10, 5))
    im = plt.imshow(G, aspect='auto', cmap='coolwarm', interpolation='nearest')
    plt.colorbar(im, label="Autocovariância γₖ")
    plt.xticks(range(max_p + 1), [f"lag {k}" for k in range(max_p + 1)])
    plt.yticks(range(M), ["Jan","Fev","Mar","Abr","Mai","Jun","Jul","Ago","Set","Out","Nov","Dez"])
    plt.title("Matriz de Yule–Walker (Autocovariâncias mensais)")
    plt.xlabel("Defasagem (k)")
    plt.ylabel("Mês")
    plt.tight_layout()
    plt.show()


# ==========================================================
# 3️⃣ Funções auxiliares para estatísticas mensais
# ==========================================================
def mean_by_month(series, n_meses=12):
    return np.array([np.mean(series[(m - 1)::n_meses]) for m in range(1, n_meses + 1)])

def std_by_month(series, n_meses=12):
    return np.array([np.std(series[(m - 1)::n_meses], ddof=0) for m in range(1, n_meses + 1)])

def skew_by_month(series, n_meses=12):
    return np.array([skew(series[(m - 1)::n_meses]) for m in range(1, n_meses + 1)])

def acf1_by_month(series, n_meses=12):
    acf1 = []
    for m in range(1, n_meses + 1):
        X = series[(m - 1)::n_meses]
        if len(X) < 2:
            acf1.append(np.nan)
            continue
        acf1.append(np.corrcoef(X[:-1], X[1:])[0,1])
    return np.array(acf1)

if __name__ == "__main__":
    from numpy.random import default_rng #type: ignore
    rng = default_rng(1234)

    # Exemplo: série de teste (substitua pela sua 'sinal')
    pwd = "/home/joao/Documentos/SeriesTemporais/REE/data/ena_hist_subsystem.csv"
    sinal = np.genfromtxt(pwd, delimiter=';', skip_header=1, usecols=(2))

    # Dados básicos
    n_anos = len(sinal) // 12
    anos = np.arange(1931, 1931 + n_anos)
    meses_nomes = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 
               'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dec']

    # --- Estatísticas mensais originais
    mu_orig = mean_by_month(sinal)
    std_orig = std_by_month(sinal)
    mus = {m: mu_orig[m - 1] for m in range(1, 13)}
    sigmas = {m: std_orig[m - 1] for m in range(1, 13)}

    # --- Normalização
    Z = np.array([(sinal[t] - mus[(t % 12) + 1]) / sigmas[(t % 12) + 1] for t in range(len(sinal))])

    # --- Definir ordens por mês (pₘ)
    p_by_month = {
        1: 3,  2: 4,  3: 4,  4: 3,
        5: 2,  6: 2,  7: 3,  8: 3,
        9: 4, 10: 4, 11: 3, 12: 2
    }

    # --- Estimar PAR(pₘ)
    phis_by_month, sigma2_by_month, gamma_by_month = yule_walker_par_variable_p(Z, p_by_month)

    # --- Visualizar matriz Yule–Walker
    plot_yule_walker_matrix(gamma_by_month)

    # --- Definir ruído lognormal padronizado
    def resid_sampler(size):
        sigma_ln = 0.6
        mu_ln = -0.5 * sigma_ln**2
        eps = rng.lognormal(mean=mu_ln, sigma=sigma_ln, size=size)
        eps -= np.mean(eps)
        eps /= np.std(eps)
        return eps

    # --- Simular série sintética
    from prog03 import simulate_par_normalizado
    Z_sim = simulate_par_normalizado(
        phis_by_month=phis_by_month,
        resid_sampler=resid_sampler,
        sigmas=sigmas,
        n_years=n_anos,
        seed=42,
        n_meses=12
    )

    # --- Desnormalizar
    Y_sim = np.array([mus[(t % 12) + 1] + sigmas[(t % 12) + 1] * Z_sim[t] for t in range(len(Z_sim))])

    # --- Estatísticas mensais (histórico vs sintético)
    mu_sim = mean_by_month(Y_sim)
    std_sim = std_by_month(Y_sim)
    skew_orig = skew_by_month(sinal)
    skew_sim = skew_by_month(Y_sim)
    acf1_orig = acf1_by_month(sinal)
    acf1_sim = acf1_by_month(Y_sim)

    # --- Impressão resumida (tabela textual)
    meses = np.array(["Jan","Fev","Mar","Abr","Mai","Jun","Jul","Ago","Set","Out","Nov","Dez"])
    print("\nResumo mensal (histórico vs sintético):")
    print(" Mês |   μ_orig   μ_sim   σ_orig   σ_sim   skew_orig  skew_sim  acf1_orig  acf1_sim")
    for i in range(12):
        print(f" {meses[i]:>3s} | {mu_orig[i]:8.1f} {mu_sim[i]:8.1f} "
              f"{std_orig[i]:8.1f} {std_sim[i]:8.1f} "
              f"{skew_orig[i]:9.3f} {skew_sim[i]:9.3f} "
              f"{acf1_orig[i]:9.3f} {acf1_sim[i]:9.3f}")

    # ==========================================================
    # 5️⃣ Gráficos comparando momentos
    # ==========================================================
    x = np.arange(12)
    width = 0.35
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Momentos Estatísticos Mensais - Histórico (azul) vs Sintético (laranja)")

    # Média
    axs[0,0].bar(x - width/2, mu_orig, width, color='tab:blue', alpha=0.7)
    axs[0,0].bar(x + width/2, mu_sim, width, color='tab:orange', alpha=0.8)
    axs[0,0].set_title("Média (μ)")
    axs[0,0].set_xticks(x); axs[0,0].set_xticklabels(meses)

    # Desvio padrão
    axs[0,1].bar(x - width/2, std_orig, width, color='tab:blue', alpha=0.7)
    axs[0,1].bar(x + width/2, std_sim, width, color='tab:orange', alpha=0.8)
    axs[0,1].set_title("Desvio padrão (σ)")
    axs[0,1].set_xticks(x); axs[0,1].set_xticklabels(meses)

    # Assimetria
    axs[1,0].bar(x - width/2, skew_orig, width, color='tab:blue', alpha=0.7)
    axs[1,0].bar(x + width/2, skew_sim, width, color='tab:orange', alpha=0.8)
    axs[1,0].axhline(0, color='black', linewidth=0.8)
    axs[1,0].set_title("Assimetria (Skewness)")
    axs[1,0].set_xticks(x); axs[1,0].set_xticklabels(meses)

    # Autocorrelação lag-1
    axs[1,1].bar(x - width/2, acf1_orig, width, color='tab:blue', alpha=0.7)
    axs[1,1].bar(x + width/2, acf1_sim, width, color='tab:orange', alpha=0.8)
    axs[1,1].axhline(0, color='black', linewidth=0.8)
    axs[1,1].set_title("Autocorrelação de 1ª ordem (ρ₁)")
    axs[1,1].set_xticks(x); axs[1,1].set_xticklabels(meses)

    plt.tight_layout(rect=[0,0,1,0.96])
    plt.show()
