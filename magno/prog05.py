import numpy as np #type: ignore
from matplotlib import pyplot as plt #type: ignore
from typing import Tuple
from scipy.linalg import toeplitz #type:ignore
from scipy.stats import skew#type:ignore
from scipy.stats import norm #type: ignore
from numpy.random import default_rng #type: ignore
 

# --- utilitários ---
def month_indices(n_years: int, m: int, n_meses: int = 12):
    """Retorna os índices (0-based) no vetor mensal para o mês m (1..12)."""
    return [(t * n_meses) + (m - 1) for t in range(n_years)]
def reshape_series(serie: np.ndarray, n_meses: int = 12) -> Tuple[np.ndarray,int]:
    """Garante array 1D e retorna (serie, n_anos)."""
    serie = np.asarray(serie, dtype=float).flatten()
    if len(serie) % n_meses != 0:
        raise ValueError("Comprimento da série não é múltiplo de n_meses.")
    n_anos = len(serie) // n_meses
    return serie, n_anos

# --- estatísticas mensais (consistentes com Eq. 3.1-3.3 da tese) ---
def mu_m(serie: np.ndarray, m: int, n_meses: int = 12) -> float:
    serie, n_anos = reshape_series(serie, n_meses)
    vals = serie[month_indices(n_anos, m, n_meses)]
    return np.mean(vals)
def sigma2_m(serie: np.ndarray, m: int, n_meses: int = 12) -> float:
    serie, n_anos = reshape_series(serie, n_meses)
    vals = serie[month_indices(n_anos, m, n_meses)]
    # variância populacional conforme sua implementação original
    mu = np.mean(vals)
    return np.mean((vals - mu) ** 2)
def sigma_m(serie: np.ndarray, m: int, n_meses: int = 12) -> float:
    return np.sqrt(sigma2_m(serie, m, n_meses))
def assim_m(serie: np.ndarray, m: int, n_meses: int = 12) -> float:
    serie, n_anos = reshape_series(serie, n_meses)
    vals = serie[month_indices(n_anos, m, n_meses)]
    # use scipy skew (bias=True para compatibilidade com seu script)
    return skew(vals, bias=True)
def mu_t(serie: np.ndarray, t: int, n_meses: int = 12) -> float:
    """Média anual no ano t (t=1..N), conforme sua função original (Eq. 3.5)."""
    serie, n_anos = reshape_series(serie, n_meses)
    if not (1 <= t <= n_anos):
        raise IndexError("t fora do intervalo de anos")
    start = (t - 1) * n_meses
    return np.mean(serie[start:start + n_meses])
def mu_global(serie: np.ndarray) -> float:
    serie, _ = reshape_series(serie)
    return np.mean(serie)
def sigma_global(serie: np.ndarray) -> float:
    serie, _ = reshape_series(serie)
    return np.std(serie, ddof=0)

# --- funções de autocorrelação por mês (FAC conforme eq. (3.15)) ---
def acf_month(serie: np.ndarray, m: int, lag_max: int = 11, n_meses: int = 12):
    """Calcula FAC para o mês m para k=1..lag_max (defasagens em meses) conforme a tese."""
    serie, n_anos = reshape_series(serie, n_meses)
    vals = serie[month_indices(n_anos, m, n_meses)]
    N = len(vals)
    mu = np.mean(vals)
    gamma = np.array([np.sum((vals[:N - k] - mu) * (vals[k:] - mu)) / N for k in range(lag_max + 1)])
    # ACf normalizado
    return gamma / gamma[0]

# --- Yule-Walker mensal (resolve sistema Toeplitz) para PAR(p) via método dos momentos ---
def yule_walker_month(serie: np.ndarray, m: int, p: int, n_meses: int = 12):
    """
    Retorna (phi, sigma2_e) para o mês m, ordem p.
    phi: array de tamanho p com phi_{m,1}...phi_{m,p} (Eq. 3.10)
    sigma2_e: variância do ruído (resíduos)
    """
    acf = acf_month(serie, m, lag_max=p, n_meses=n_meses)
    # montar matriz Toeplitz com autocovariâncias (populacionais)
    # usamos autocovariância: gamma_k = acf[k] * var
    var_m = sigma2_m(serie, m, n_meses)
    gamma = acf * var_m
    # Toeplitz T where T[i,j] = gamma[|i-j|]
    R = toeplitz(gamma[:p])
    r = gamma[1:p+1]  # vetor RHS (gamma1..gammap)
    phi = np.linalg.solve(R, r)
    # ruído branco: sigma2_e = gamma0 - phi^T r
    sigma2_e = gamma[0] - np.dot(phi, r)
    return phi, sigma2_e

# --- gerar série PAR(p) (simulação direta da Eq. 3.10) ---
def simulate_par(phis_by_month: dict, resid_sampler, n_years: int, seed: int = None, n_meses: int = 12):
    """
    phis_by_month: dict[m] = array([phi1, phi2, ..., phip_m])  (m = 1..12)
    resid_sampler: callable(size) -> array de resíduos (pode ser amostrador bootstrap)
    retorna vetor mensal de comprimento n_years * n_meses
    """
    rng = np.random.default_rng(seed)
    T = n_years * n_meses
    out = np.zeros(T)
    # precisamos de um burn-in: inicializamos primeiros p meses com média histórica 0 (se normalizado)
    max_p = max(len(v) for v in phis_by_month.values())
    # inicializar primeiros max_p valores com zeros (se trabalhar com série normalizada!)
    # depois geraremos resíduos com resid_sampler
    res = resid_sampler(T)
    for t in range(T):
        # identificar mês m (1..12)
        m = (t % n_meses) + 1
        phis = phis_by_month.get(m, np.array([]))
        p = len(phis)
        if p == 0:
            out[t] = res[t]
            continue
        s = 0.0
        for k in range(1, p + 1):
            idx = t - k
            y_lag = out[idx] if idx >= 0 else 0.0
            s += phis[k - 1] * y_lag
        out[t] = s + res[t]
    return out

def simulate_par_normalizado(phis_by_month, resid_sampler, sigmas, n_years, seed=None, n_meses=12):
    if seed is not None:
        np.random.seed(seed)  # garante reprodutibilidade global

    T = n_years * n_meses
    out = np.zeros(T)
    max_p = max(len(v) for v in phis_by_month.values())
    res = resid_sampler(T)  # amostra resíduos (usa NumPy global)

    for t in range(T):
        m = (t % n_meses) + 1
        phis = phis_by_month.get(m, np.array([]))
        p = len(phis)
        s = 0.0
        for k in range(1, p + 1):
            idx = t - k
            if idx < 0:
                continue
            m_lag = ((t - k) % n_meses) + 1
            scale = sigmas[m_lag] / sigmas[m]
            s += phis[k - 1] * scale * out[idx]
        out[t] = s + res[t]
    return out

def par_normalizado(serie: np.ndarray, phis_by_month: dict, mus: dict, sigmas: dict, n_meses: int = 12):
    """
    Modelo PAR(p) conforme Eq. (3.11)
    
    Parâmetros:
        serie : array-like
            Série original Y_{t} (valores mensais consecutivos)
        phis_by_month : dict
            Dicionário com coeficientes por mês: {mês: [phi1, phi2, ..., phi_p_m]}
        mus : dict
            Médias mensais {mês: mu_m}
        sigmas : dict
            Desvios-padrão mensais {mês: sigma_m}
        n_meses : int
            Número de meses no ciclo (default 12)
    
    Retorna:
        Z : np.ndarray
            Série normalizada Z_{m,t}
        Z_model : np.ndarray
            Série ajustada (modelo PAR normalizado)
        eps_norm : np.ndarray
            Resíduos normalizados (epsilon_{m,t}/sigma_m)
    """
    serie = np.asarray(serie, dtype=float)
    T = len(serie)
    Z = np.zeros(T)
    Z_model = np.zeros(T)
    eps_norm = np.zeros(T)

    #Normalização mensal
    for t in range(T):
        m = (t % n_meses) + 1
        Z[t] = (serie[t] - mus[m]) / sigmas[m]

    #Aplicação da forma normalizada do modelo (Eq. 3.11)
    for t in range(T):
        m = (t % n_meses) + 1
        phis = phis_by_month.get(m, [])
        if not phis:
            continue
        s = 0.0
        for k, phi in enumerate(phis, start=1):
            idx = t - k
            if idx < 0:
                continue  # sem histórico anterior suficiente
            m_lag = ((t - k) % n_meses) + 1
            s += phi * (sigmas[m_lag] / sigmas[m]) * Z[idx]
        Z_model[t] = s
        eps_norm[t] = Z[t] - Z_model[t]

    return Z, Z_model, eps_norm

def yule_walker_par(serie: np.ndarray, p: int, n_meses: int = 12):
    """
    Estima os coeficientes phi_{m,k} e variâncias sigma²_epsilon,m do modelo PAR(p)
    conforme as equações de Yule–Walker (seção 3.3.2).

    Parâmetros:
        serie : array-like
            Série mensal Y_{t} (valores consecutivos)
        p : int
            Ordem do modelo (p_m = p para todos os meses, por simplicidade)
        n_meses : int
            Número de meses por ciclo (default 12)

    Retorna:
        phis_by_month : dict
            {m: np.ndarray com phi_{m,1..p}}
        sigma2_by_month : dict
            {m: variância dos resíduos sigma²_{ε,m}}
        Z : np.ndarray
            Série normalizada (Eq. 3.11)
    """
    serie = np.asarray(serie, dtype=float)
    n_anos = len(serie) // n_meses

    # Calcular médias e desvios por mês ---
    mus = {m: mu_m(serie, m, n_meses) for m in range(1, n_meses + 1)}
    sigmas = {m: sigma_m(serie, m, n_meses) for m in range(1, n_meses + 1)}

    # Normalizar a série conforme Eq. (3.11) ---
    #    (usando a função que já implementamos)
    Z, _, _ = par_normalizado(serie, phis_by_month={}, mus=mus, sigmas=sigmas, n_meses=n_meses)

    # Estimar phi e sigma² via Yule–Walker (Eq. 3.10) ---
    phis_by_month = {}
    sigma2_by_month = {}

    for m in range(1, n_meses + 1):
        # extrair sub-série do mês m
        idx = np.arange(m - 1, len(Z), n_meses)
        Zm = Z[idx]
        N = len(Zm)
        mu_z = np.mean(Zm)

        # autocovariâncias até lag p
        gamma = np.array([
            np.sum((Zm[:N - k] - mu_z) * (Zm[k:] - mu_z)) / N for k in range(p + 1)
        ])

        # construir matriz Toeplitz e vetor de autocovariâncias
        R = toeplitz(gamma[:-1])
        r = gamma[1:]

        # resolver sistema Yule–Walker
        phi_m = np.linalg.solve(R, r)
        sigma2_e = gamma[0] - np.dot(phi_m, r)

        phis_by_month[m] = phi_m
        sigma2_by_month[m] = sigma2_e

    return phis_by_month, sigma2_by_month, Z

rng = default_rng(2025)  # seed reprodutível
def resid_sampler(size):
    # parâmetros da lognormal subjacente (da Normal)
    sigma_ln = 0.3
    mu_ln = -0.5 * sigma_ln**2  # garante média ≈ 1

    # gera lognormal positiva
    L = rng.lognormal(mean=mu_ln, sigma=sigma_ln, size=size)

    # centraliza e padroniza (média 0, desvio 1)
    L -= np.mean(L)
    L /= np.std(L)

    return L

# from scipy.linalg import toeplitz
# Função Yule-Walker com p_m variável
def yule_walker_par_variable_p(Z, p_by_month, n_meses=12):
    """
    Estima modelos PAR(p_m) com ordem variável por mês, usando equações de Yule-Walker.

    Parâmetros:
    -----------
    Z : array-like
        Série temporal (normalizada, tipo Z_t)
    p_by_month : dict[int, int]
        Dicionário {mês: ordem p_m} com p_m específico de cada mês (1..12)
    n_meses : int
        Número de meses do ciclo (default = 12)

    Retorna:
    --------
    phis_by_month : dict[mês -> np.array(phi_1..phi_p_m)]
    sigma2_by_month : dict[mês -> float]
    gamma_by_month : dict[mês -> np.array(autocovariâncias)]
    """
    phis_by_month = {}
    sigma2_by_month = {}
    gamma_by_month = {}

    n_anos = len(Z) // n_meses

    for m in range(1, n_meses + 1):
        p_m = p_by_month.get(m, 1)
        Z_m = Z[(m - 1)::n_meses]
        N = len(Z_m)
        mu_m = np.mean(Z_m)

        # autocovariâncias γ_k até lag p_m
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

# Função de visualização da matriz de Yule-Walker
def plot_yule_walker_matrix(gamma_by_month):
    """
    Plota a matriz de autocovariâncias (Yule–Walker) para cada mês.
    Cada linha do gráfico representa as autocovariâncias até a ordem p_m.

    Usa escala de cores (heatmap) para visualizar intensidade e sinal.
    """
    # construir matriz para visualização
    max_p = max(len(g) - 1 for g in gamma_by_month.values())
    M = len(gamma_by_month)
    G = np.full((M, max_p + 1), np.nan)

    for m, gamma in gamma_by_month.items():
        G[m - 1, :len(gamma)] = gamma

    plt.figure(figsize=(10, 5))
    im = plt.imshow(G, aspect='auto', cmap='coolwarm', interpolation='nearest')
    plt.colorbar(im, label="Autocovariância γ_k")
    plt.xticks(range(max_p + 1), [f"lag {k}" for k in range(max_p + 1)])
    plt.yticks(range(M), ["Jan","Fev","Mar","Abr","Mai","Jun","Jul","Ago","Set","Out","Nov","Dez"])
    plt.title("Matriz de Yule–Walker (Autocovariâncias mensais)")
    plt.xlabel("Defasagem (k)")
    plt.ylabel("Mês")
    plt.tight_layout()
    plt.show()

# ========================================== main script ==========================================
# pegando serie da ena_hist_subsystem.csv.

# eventdate;ena_mwmed_norte;ena_mwmed_nordeste;ena_mwmed_sul;ena_mwmed_seco
# 1931-01-01;   10764.6;    14125.1;    6946.44;    56728.43
# 1931-02-01;   13701.7;    13168.4;    3042.57;    86455.12
# 1931-03-01;   22321.1;    18892.4;    3239.21;    88431.05
# 1931-04-01;   22017.0;    20906.9;    2193.65;    64029.48
# 1931-05-01;   10086.2;    14299.6;    17016.9;    42569.05
# 1931-06-01;   5925.28;    7186.41;    15968.4;    31874.17
# ... 
# 620 observações no total (1931-01 a 2010-12)

pwd = "/home/joao/Documentos/SeriesTemporais/REE/data/ena_hist_subsystem.csv"
sinal = np.genfromtxt(pwd, delimiter=';', skip_header=1, usecols=(1))

# Dados básicos
n_anos = len(sinal) // 12
anos = np.arange(1931, 1931 + n_anos)
meses_nomes = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 
               'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dec']

if __name__ == "__main__":
    n_meses = 12
    sinal = np.asarray(sinal, dtype=float)
    n_anos = len(sinal) // n_meses

    # === Estatísticas básicas ===
    mus = {m: mu_m(sinal, m, n_meses) for m in range(1, 13)}
    sigmas = {m: sigma_m(sinal, m, n_meses) for m in range(1, 13)}

    # Série normalizada (Z_t)
    Z, _, _ = par_normalizado(sinal, {}, mus, sigmas, n_meses=n_meses)

    # === Definição de p_m (ordem variável por mês) ===
    p_by_month = {
        1: 3,  2: 4,  3: 4,  4: 3,
        5: 2,  6: 2,  7: 3,  8: 3,
        9: 4, 10: 4, 11: 3, 12: 2
    }

    # === Estimar modelo PAR(p_m) ===
    phis_by_month, sigma2_by_month, gamma_by_month = yule_walker_par_variable_p(Z, p_by_month, n_meses=n_meses)

    # === Visualizar matriz de autocovariâncias ===
    plot_yule_walker_matrix(gamma_by_month)

    # === Resíduos lognormais padronizados ===
    def resid_sampler(size):
        sigma_ln = 0.4
        mu_ln = -0.5 * sigma_ln**2
        eps = np.random.lognormal(mean=mu_ln, sigma=sigma_ln, size=size)
        eps -= np.mean(eps)
        eps /= np.std(eps)
        return eps

    # === Simulação PAR(p_m) ===
    Z_sim = simulate_par_normalizado(
        phis_by_month=phis_by_month,
        resid_sampler=resid_sampler,
        sigmas=sigmas,
        n_years=n_anos,
        seed=42,
        n_meses=n_meses
    )

    # === Desnormalização ===
    Y_sim = np.array([
        mus[(t % n_meses) + 1] + sigmas[(t % n_meses) + 1] * Z_sim[t]
        for t in range(len(Z_sim))
    ])

    # === Estatísticas mensais ===
    meses = np.arange(1, n_meses + 1)
    meses_labels = ["Jan","Fev","Mar","Abr","Mai","Jun","Jul","Ago","Set","Out","Nov","Dez"]

    def mean_by_month(series): return np.array([np.mean(series[(m - 1)::n_meses]) for m in meses])
    def std_by_month(series): return np.array([np.std(series[(m - 1)::n_meses]) for m in meses])
    def skew_by_month(series): return np.array([skew(series[(m - 1)::n_meses]) for m in meses])
    def acf1_by_month(series):
        acf1 = []
        for m in meses:
            vals = series[(m - 1)::n_meses]
            acf1.append(np.corrcoef(vals[:-1], vals[1:])[0, 1] if len(vals) > 1 else np.nan)
        return np.array(acf1)

    mu_orig, mu_sim = mean_by_month(sinal), mean_by_month(Y_sim)
    std_orig, std_sim = std_by_month(sinal), std_by_month(Y_sim)
    skew_orig, skew_sim = skew_by_month(sinal), skew_by_month(Y_sim)
    acf1_orig, acf1_sim = acf1_by_month(sinal), acf1_by_month(Y_sim)

    print("\nResumo mensal (histórico vs sintético):")
    print("Mês | μ_orig μ_sim σ_orig σ_sim skew_orig skew_sim acf1_orig acf1_sim")
    for i in range(n_meses):
        print(f"{meses_labels[i]:>3s} | {mu_orig[i]:8.1f} {mu_sim[i]:8.1f} "
              f"{std_orig[i]:8.1f} {std_sim[i]:8.1f} "
              f"{skew_orig[i]:9.3f} {skew_sim[i]:9.3f} "
              f"{acf1_orig[i]:9.3f} {acf1_sim[i]:9.3f}")

    # === Gráficos comparativos ===
    x = np.arange(n_meses)
    width = 0.35
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Momentos Estatísticos Mensais - Histórico (azul) vs Sintético (laranja) [PAR(p_m)]")

    # Média
    axs[0, 0].bar(x - width/2, mu_orig, width, color='tab:blue', alpha=0.7)
    axs[0, 0].bar(x + width/2, mu_sim, width, color='tab:orange', alpha=0.8)
    axs[0, 0].set_title("Média (μ)"); axs[0, 0].set_xticks(x); axs[0, 0].set_xticklabels(meses_labels)

    # Desvio padrão
    axs[0, 1].bar(x - width/2, std_orig, width, color='tab:blue', alpha=0.7)
    axs[0, 1].bar(x + width/2, std_sim, width, color='tab:orange', alpha=0.8)
    axs[0, 1].set_title("Desvio padrão (σ)"); axs[0, 1].set_xticks(x); axs[0, 1].set_xticklabels(meses_labels)

    # Assimetria
    axs[1, 0].bar(x - width/2, skew_orig, width, color='tab:blue', alpha=0.7)
    axs[1, 0].bar(x + width/2, skew_sim, width, color='tab:orange', alpha=0.8)
    axs[1, 0].axhline(0, color='black', linewidth=0.8)
    axs[1, 0].set_title("Assimetria (Skewness)"); axs[1, 0].set_xticks(x); axs[1, 0].set_xticklabels(meses_labels)

    # Autocorrelação
    axs[1, 1].bar(x - width/2, acf1_orig, width, color='tab:blue', alpha=0.7)
    axs[1, 1].bar(x + width/2, acf1_sim, width, color='tab:orange', alpha=0.8)
    axs[1, 1].axhline(0, color='black', linewidth=0.8)
    axs[1, 1].set_title("Autocorrelação lag-1 (ρ₁)"); axs[1, 1].set_xticks(x); axs[1, 1].set_xticklabels(meses_labels)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()




