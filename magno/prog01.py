import numpy as np #type: ignore
from scipy.stats import skew #type: ignore
from matplotlib import pyplot as plt #type: ignore
from scipy.stats import norm #type: ignore

def mu_m(serie, m, n_meses=12):

# Calcula a média mensal (Eq. 3.1) para um mês específico m.
# Argumentos:
#     serie   : (array_like) Série histórica de ENA organizada em sequência (ano a ano, mês a mês).m : int
#     m       : (int)        Índice do mês (0 = janeiro, 11 = dezembro).
#     n_meses : (int),       opcional, número de meses no período (default=12).

# Retorna:
#     mu_m : (float)         Média da ENA para o mês m.

    serie = np.asarray(serie, dtype=float)
    n_anos = len(serie) // n_meses
    indices = [(t-1)*n_meses + (m-1) for t in range(1, n_anos+1)]
    valores = serie[indices]
    result = np.mean(valores)
    return result
def sigma2_m(serie, m, n_meses=12):

# Calcula a variância mensal (Eq. 3.2) para um mês específico m.
# Parametros:
#     serie    : (array_like) Série histórica de ENA organizada em sequência (ano a ano, mês a mês).
#     m        : (int)        Índice do mês (1 = janeiro, 12 = dezembro).
#     n_meses  : (int)        opcional, número de meses no período (default=12).

# Retorna:
#     sigma2_m : (float)      Variância da ENA para o mês m.

    serie = np.asarray(serie, dtype=float)
    n_anos = len(serie) // n_meses

    # índices do mês m em cada ano
    indices = [(t-1)*n_meses + (m-1) for t in range(1, n_anos+1)]
    valores = serie[indices]

    mu_m = np.mean(valores) #tirar a media com mu_m
    sigma2_m = np.mean((valores - mu_m)**2)  # variância populacional

    return sigma2_m
def sigma_m(serie, m, n_meses=12):

# Calcula o desvio padrão mensal (Eq. 3.3) para um mês específico m.
# Parametros:
#     serie    : (array_like) Série histórica de ENA organizada em sequência (ano a ano, mês a mês).
#     m        : (int)        Índice do mês (1 = janeiro, 12 = dezembro).
#     n_meses  : (int)        opcional, número de meses no período (default=12).

# Retorna:
#     sigma_m : (float)       Desvio padrão da ENA para o mês m.

    sigma2 = sigma2_m(serie, m, n_meses)
    sigma_m = np.sqrt(sigma2)
    return sigma_m
def assim_m(serie, m, n_meses=12):
# Calcula o desvio padrão mensal (Eq. 3.3) para um mês específico m.
# Parametros:
#     serie    : (array_like) Série histórica de ENA organizada em sequência (ano a ano, mês a mês).
#     m        : (int)        Índice do mês (1 = janeiro, 12 = dezembro).
#     n_meses  : (int)        opcional, número de meses no período (default=12).

# Retorna:
#     assimetria : (float)    Assimetria da ENA para o mês m.
    
    serie = np.asarray(serie, dtype=float)
    N = len(serie) // n_meses

    # índices do mes m em cada ano
    indices = [(t-1)*n_meses + (m-1) for t in range(1, N+1)]
    valores = serie[indices]

    mu = mu_m(serie, m, n_meses)
    sigma = sigma_m(serie, m, n_meses)

    numerador = 0.0
    for x in valores:
        numerador += (x-mu)**3

    denominador = ((N-1)*(N-2)/N)*(sigma**3)
    
    return(numerador/denominador)
def mu_t(serie, t, n_meses=12):    
# Calcula o desvio padrão mensal (Eq. 3.5) para um mês específico m.
# Parametros:
#     serie          : (array_like) Série histórica de ENA organizada em sequência (ano a ano, mês a mês).
#     t              : (int)        ano a ser analisado (1 = primeiro ano, 2 = segundo ano, etc.).
#     n_meses        : (int)        opcional, número de meses no período (default=12).

# Retorna:
#     media do ano t : (float)    Assimetria da ENA para o mês t.

    inicio = (t-1)*n_meses
    fim = inicio+n_meses
    
    valores_ano = serie[inicio:fim]
    
    return (sum(valores_ano)/n_meses)
def mu_global(serie):
# Calcula a media anual (Eq. 3.6)
    return np.mean(serie)
def sigma_global(serie):
# Calcula o desvio padrão anual (Eq. 3.7)
    return np.std(serie, ddof=0)
def assim_global(serie):
    return skew(serie, bias=True)
def correlacao(serie):
    soma = 0.0
    for i in range(1, (len(serie)-1)):
        soma += (mu_t(serie, i) - mu_global(serie))*(mu_t(serie, i+1) - mu_global(serie))
    
    soma = soma/len(serie)
    return soma/((sigma_global(serie))**2)

# pegando serie da ena_hist_subsystem.csv.
pwd = "/home/joao/Documentos/SeriesTemporais/REE/data/ena_hist_subsystem.csv"
sinal = np.genfromtxt(pwd, delimiter=';', skip_header=1, usecols=(2))

# Dados básicos
n_anos = len(sinal) // 12
anos = np.arange(1931, 1931 + n_anos)
meses_nomes = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 
               'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dec']

# 1. SÉRIE TEMPORAL COMPLETA
plt.figure(figsize=(15, 6))
tempo = np.arange(len(sinal))
plt.plot(tempo, sinal, 'b-', alpha=0.7, linewidth=1)
media_global = mu_global(sinal)
std_global = sigma_global(sinal)
plt.axhline(media_global, color='red', linestyle='--', linewidth=2, 
           label=f'Média: {media_global:.1f}')
plt.fill_between(tempo, media_global - std_global, media_global + std_global,
                alpha=0.2, color='gray', label='±1 Desvio Padrão')
plt.title('Série Temporal da ENA - Subsistema Sul')
plt.xlabel('Meses desde 1931')
plt.ylabel('ENA (MWmed)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 2. HISTOGRAMA GLOBAL
plt.figure(figsize=(12, 6))
plt.hist(sinal, bins=50, alpha=0.7, density=True, edgecolor='black')
x = np.linspace(np.min(sinal), np.max(sinal), 1000)
y = norm.pdf(x, media_global, std_global)
plt.plot(x, y, 'r-', linewidth=2, label='Distribuição Normal')
plt.axvline(media_global, color='red', linestyle='--', linewidth=2)
plt.title('Histograma da ENA Mensal')
plt.xlabel('ENA (MWmed)')
plt.ylabel('Densidade')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 3. BOXPLOT POR MÊS
dados_por_mes = [sinal[i::12] for i in range(12)]
medias_mensais = [mu_m(sinal, i+1) for i in range(12)]

plt.figure(figsize=(14, 6))
plt.boxplot(dados_por_mes, labels=meses_nomes)
plt.plot(range(1, 13), medias_mensais, 'ro-', linewidth=2, markersize=6, label='Médias')
plt.title('Distribuição da ENA por Mês')
plt.xlabel('Mês')
plt.ylabel('ENA (MWmed)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 4. ESTATÍSTICAS MENSIAIS
medias = [mu_m(sinal, i+1) for i in range(12)]
stds = [sigma_m(sinal, i+1) for i in range(12)]

# 4.1 Médias mensais
plt.figure(figsize=(12, 6))
plt.bar(range(1, 13), medias, alpha=0.7, edgecolor='black')
plt.axhline(media_global, color='red', linestyle='--', label='Média Global')
plt.xticks(range(1, 13), meses_nomes)
plt.title('Médias Mensais da ENA')
plt.xlabel('Mês')
plt.ylabel('ENA Média (MWmed)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 4.2 Desvios padrão mensais
plt.figure(figsize=(12, 6))
plt.bar(range(1, 13), stds, alpha=0.7, color='orange', edgecolor='black')
plt.xticks(range(1, 13), meses_nomes)
plt.title('Desvios Padrão Mensais da ENA')
plt.xlabel('Mês')
plt.ylabel('Desvio Padrão (MWmed)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 5. SÉRIE ANUAL
medias_anuais = [np.mean(sinal[i*12:(i+1)*12]) for i in range(n_anos)]

plt.figure(figsize=(15, 6))
plt.plot(anos, medias_anuais, 'bo-', linewidth=2, markersize=4)
plt.axhline(media_global, color='red', linestyle='--', label='Média Global')

# Tendência linear
coef = np.polyfit(anos, medias_anuais, 1)
tendencia = np.poly1d(coef)
plt.plot(anos, tendencia(anos), 'r-', linewidth=2, 
         label=f'Tendência: y = {coef[0]:.3f}x + {coef[1]:.1f}')

plt.title('Média Anual da ENA')
plt.xlabel('Ano')
plt.ylabel('ENA Média Anual (MWmed)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 6. MATRIZ DE CORRELAÇÃO
corr_matrix = np.zeros((12, 12))
for i in range(12):
    for j in range(12):
        corr_matrix[i, j] = np.corrcoef(sinal[i::12], sinal[j::12])[0, 1]

plt.figure(figsize=(10, 8))
plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar(label='Correlação')
plt.xticks(range(12), meses_nomes, rotation=45)
plt.yticks(range(12), meses_nomes)
plt.title('Correlação entre Meses')

for i in range(12):
    for j in range(12):
        plt.text(j, i, f'{corr_matrix[i, j]:.2f}', 
                 ha='center', va='center', fontsize=8,
                 color='white' if abs(corr_matrix[i, j]) > 0.5 else 'black')
plt.tight_layout()
plt.show()

# 7. COMPARAÇÃO COM DISTRIBUIÇÃO NORMAL
plt.figure(figsize=(12, 6))
plt.hist(sinal, bins=50, alpha=0.7, density=True, edgecolor='black', label='Dados')
x = np.linspace(np.min(sinal), np.max(sinal), 1000)
plt.plot(x, norm.pdf(x, media_global, std_global), 'r-', linewidth=2, 
         label=f'N(μ={media_global:.1f}, σ={std_global:.1f})')
plt.title('Comparação com Distribuição Normal')
plt.xlabel('ENA (MWmed)')
plt.ylabel('Densidade')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 8. SÉRIE POR DÉCADAS
decadas = np.arange(1930, 2011, 10)
cores = plt.cm.tab10(np.linspace(0, 1, len(decadas)))

plt.figure(figsize=(15, 8))
for i, decada in enumerate(decadas):
    mask = (anos >= decada) & (anos < decada + 10)
    if np.any(mask):
        anos_decada = anos[mask]
        dados_decada = np.concatenate([sinal[i*12:(i+1)*12] for i in range(n_anos) if mask[i]])
        tempo_decada = np.arange(len(dados_decada)) + (decada - 1930) * 120
        plt.plot(tempo_decada, dados_decada, alpha=0.7, 
                label=f'{decada}s', color=cores[i])

plt.axhline(media_global, color='red', linestyle='--', linewidth=2)
plt.title('ENA por Década')
plt.xlabel('Meses desde 1930')
plt.ylabel('ENA (MWmed)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 9. SCATTER PLOT MENSAL (exemplo: Janeiro vs Julho)
janeiro = sinal[0::12]
julho = sinal[6::12]

plt.figure(figsize=(10, 8))
plt.scatter(janeiro, julho, alpha=0.6, s=30)
plt.xlabel('ENA Janeiro (MWmed)')
plt.ylabel('ENA Julho (MWmed)')
plt.title('Relação entre Janeiro e Julho')
plt.grid(True, alpha=0.3)

# Linha de tendência
coef = np.polyfit(janeiro, julho, 1)
tendencia = np.poly1d(coef)
x_fit = np.linspace(np.min(janeiro), np.max(janeiro), 100)
plt.plot(x_fit, tendencia(x_fit), 'r-', linewidth=2,
         label=f'y = {coef[0]:.3f}x + {coef[1]:.1f}')

plt.legend()
plt.tight_layout()
plt.show()

# ESTATÍSTICAS RESUMO
print("="*60)
print("RESUMO ESTATÍSTICO - ENA SUBSISTEMA SUL")
print("="*60)
print(f"Período: 1931-2010 ({n_anos} anos)")
print(f"Média global: {media_global:.1f} MWmed")
print(f"Desvio padrão global: {std_global:.1f} MWmed")
print(f"Coef. de variação: {(std_global/media_global*100):.1f}%")
print(f"Assimetria: {skew(sinal, bias=True):.3f}")

print(f"\nMês com maior média: {meses_nomes[np.argmax(medias)]} ({np.max(medias):.1f} MWmed)")
print(f"Mês com menor média: {meses_nomes[np.argmin(medias)]} ({np.min(medias):.1f} MWmed)")
print(f"Mês com maior variabilidade: {meses_nomes[np.argmax(stds)]}")

corr_serial = np.corrcoef(medias_anuais[:-1], medias_anuais[1:])[0, 1]
print(f"Correlação serial anual: {corr_serial:.3f}")
print(f"Tendência anual: {coef[0]:.4f} MWmed/ano")