import numpy as np #type: ignore
from matplotlib import pyplot as plt #type: ignore
from typing import Tuple
from scipy.linalg import toeplitz #type:ignore
from scipy.stats import skew#type:ignore
from scipy.stats import norm #type: ignore
 

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



# ========================================== main script ==========================================
# pegando serie da ena_hist_subsystem.csv.
pwd = "/home/joao/Documentos/SeriesTemporais/REE/data/ena_hist_subsystem.csv"
sinal = np.genfromtxt(pwd, delimiter=';', skip_header=1, usecols=(2))

# Dados básicos
n_anos = len(sinal) // 12
anos = np.arange(1931, 1931 + n_anos)
meses_nomes = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 
               'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dec']

if __name__ == "__main__":
    # Definir ordem do modelo
    p = 12
    n_meses = 12
    n_anos = len(sinal) // n_meses

    # Estimar coeficientes PAR(p)
    phis_by_month, sigma2_by_month, Z = yule_walker_par(sinal, p=p)

    # Calcular estatísticas mensais
    mus = {m: mu_m(sinal, m) for m in range(1, 13)}
    sigmas = {m: sigma_m(sinal, m) for m in range(1, 13)}

    # Definir gerador de resíduos (normal padrão)
    resid_sampler = lambda size: np.random.normal(0, 1, size)
    
    # Definir gerador de residuos (log normal)
    # resid_sampler = lambda size: np.random.lognormal(mean=0, sigma=1, size=size)

    # Simular série normalizada
    Z_sim = simulate_par_normalizado(
        phis_by_month=phis_by_month,   #já definida.
        resid_sampler=resid_sampler,
        sigmas=sigmas,
        n_years=len(sinal)//12,
        seed=42
    )

    # Desnormalizar
    Y_sim = np.array([
        mus[(t % 12) + 1] + sigmas[(t % 12) + 1] * Z_sim[t]
        for t in range(len(Z_sim))
    ])

    # # Plot
    # plt.figure(figsize=(10,5))
    # plt.plot(sinal, label="Histórico", color="tab:blue", alpha=0.6)
    # plt.plot(Y_sim, label="Simulado (PAR(p))", color="tab:orange", alpha=0.8)
    # plt.legend(); plt.grid(); plt.tight_layout()
    # plt.show()

    import pandas as pd #type: ignore
    from scipy.stats import ks_2samp, skew #type: ignore

    # 4) Estatísticas originais por mês
    months = list(range(1, n_meses+1))
    mu_orig = [mu_m(sinal, m) for m in months]
    std_orig = [sigma_m(sinal, m) for m in months]
    skew_orig = [assim_m(sinal, m) for m in months]
    acf1_orig = [acf_month(sinal, m, lag_max=1, n_meses=n_meses)[1] for m in months]

    # 5) Simulação (usar bootstrap ou normal; exemplo: normal padrão)
    # escolha: resid_sampler = lambda size: np.random.normal(0,1,size)
    resid_sampler = lambda size: np.random.default_rng(12345).normal(0, 1, size)
    sigmas = {m: std_orig[m-1] for m in months}
    mus = {m: mu_orig[m-1] for m in months}

    # 6) Estatísticas simuladas por mês
    mu_sim = [np.mean(Y_sim[(m-1)::n_meses]) for m in months]
    std_sim = [np.std(Y_sim[(m-1)::n_meses], ddof=0) for m in months]
    skew_sim = [skew(Y_sim[(m-1)::n_meses], bias=True) for m in months]
    acf1_sim = [acf_month(Y_sim, m, lag_max=1, n_meses=n_meses)[1] for m in months]

    # 7) KS teste mês-a-mês
    ks_pvalues = []
    for m in months:
        orig_vals = sinal[(m-1)::n_meses]
        sim_vals  = Y_sim[(m-1)::n_meses]
        _, pval = ks_2samp(orig_vals, sim_vals)
        ks_pvalues.append(pval)

    # 8) Montar DataFrame resumo
    df = pd.DataFrame({
        "month": months,
        "month_name": ['Jan','Fev','Mar','Abr','Mai','Jun','Jul','Ago','Set','Out','Nov','Dec'],
        "mu_orig": mu_orig, "mu_sim": mu_sim,
        "std_orig": std_orig, "std_sim": std_sim,
        "skew_orig": skew_orig, "skew_sim": skew_sim,
        "acf1_orig": acf1_orig, "acf1_sim": acf1_sim,
        "ks_pvalue": ks_pvalues
    })
    df["mu_diff_abs"] = (df["mu_orig"] - df["mu_sim"]).abs()
    df["std_rel_err"] = (df["std_orig"] - df["std_sim"]).abs() / df["std_orig"].replace(0, np.nan)

    print("\nResumo mensal (primeiras linhas):")
    print(df.head(12).to_string(index=False))

    # 9) Gráficos
    plt.figure(figsize=(12,4))
    plt.plot(sinal, label="Histórico", alpha=0.7)
    plt.plot(Y_sim, label="Sintético (PAR)", alpha=0.8)
    plt.title(f"Histórico vs Sintético - PAR({p})")
    plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

    # Boxplots mensais
    orig_monthly = [sinal[(m-1)::n_meses] for m in months]
    sim_monthly  = [Y_sim[(m-1)::n_meses] for m in months]

    plt.figure(figsize=(12,5))
    positions_orig = np.arange(1,13) - 0.15
    positions_sim  = np.arange(1,13) + 0.15
    plt.boxplot(orig_monthly, positions=positions_orig, widths=0.25)
    plt.boxplot(sim_monthly, positions=positions_sim, widths=0.25)
    plt.xticks(range(1,13), df["month_name"])
    plt.title("Boxplots mensais: histórico (esq) vs sintético (dir)")
    plt.grid(axis="y"); plt.tight_layout(); plt.show()

    # nomes dos meses
    meses = meses_nomes
    x = np.arange(len(meses))  # posições das barras
    width = 0.35               # largura das barras

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Comparação dos Momentos Estatísticos Mensais - Histórico vs Sintético (PAR)", fontsize=14)

    # Média ---
    axs[0, 0].bar(x - width/2, df["mu_orig"], width, label='Histórico', color='tab:blue', alpha=0.7)
    axs[0, 0].bar(x + width/2, df["mu_sim"],  width, label='Sintético', color='tab:orange', alpha=0.8)
    axs[0, 0].set_title("Média mensal (μ)")
    axs[0, 0].set_xticks(x)
    axs[0, 0].set_xticklabels(meses)
    axs[0, 0].legend()

    # Desvio padrão ---
    axs[0, 1].bar(x - width/2, df["std_orig"], width, label='Histórico', color='tab:blue', alpha=0.7)
    axs[0, 1].bar(x + width/2, df["std_sim"],  width, label='Sintético', color='tab:orange', alpha=0.8)
    axs[0, 1].set_title("Desvio padrão mensal (σ)")
    axs[0, 1].set_xticks(x)
    axs[0, 1].set_xticklabels(meses)
    axs[0, 1].legend()

    # Assimetria ---
    axs[1, 0].bar(x - width/2, df["skew_orig"], width, label='Histórico', color='tab:blue', alpha=0.7)
    axs[1, 0].bar(x + width/2, df["skew_sim"],  width, label='Sintético', color='tab:orange', alpha=0.8)
    axs[1, 0].set_title("Assimetria mensal (Skewness)")
    axs[1, 0].set_xticks(x)
    axs[1, 0].set_xticklabels(meses)
    axs[1, 0].axhline(0, color='black', linewidth=0.8)
    axs[1, 0].legend()

    # Autocorrelação lag-1 ---
    axs[1, 1].bar(x - width/2, df["acf1_orig"], width, label='Histórico', color='tab:blue', alpha=0.7)
    axs[1, 1].bar(x + width/2, df["acf1_sim"],  width, label='Sintético', color='tab:orange', alpha=0.8)
    axs[1, 1].set_title("Autocorrelação de 1ª ordem (ρ₁)")
    axs[1, 1].set_xticks(x)
    axs[1, 1].set_xticklabels(meses)
    axs[1, 1].axhline(0, color='black', linewidth=0.8)
    axs[1, 1].legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


