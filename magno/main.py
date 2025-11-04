import numpy as np #type: ignore
from scipy.stats import skew #type: ignore
import prog01 as pg
import matplotlib.pyplot as plt #type: ignore
import seaborn as sns #type: ignore
from matplotlib.dates import YearLocator, MonthLocator, DateFormatter #type: ignore
import datetime #type: ignore
import pandas as pd #type: ignore
from scipy.stats import norm #type: ignore

# pegando serie da ena_hist_subsystem.csv.
pwd = "/home/joao/Documentos/SeriesTemporais/REE/data/ena_hist_subsystem.csv"
sinal = np.genfromtxt(pwd, delimiter=';', skip_header=1, usecols=(2))


print(f"Tamanho da série: {len(sinal)}")
print(f"Período: {len(sinal)//12} anos")
print(f"Média global: {pg.mu_global(sinal):.3f}")
print(f"Desvio padrão global: {pg.sigma_global(sinal):.3f}")
print(f"Assimetria global: {pg.assim_global(sinal):.3f}")
print(f"Correlação serial anual: {pg.correlacao(sinal):.3f}")

for mes in range(0, 12):
    print("-"*30)
    print(f"\nPara mes (m={mes}):")
    print(f"Média: {pg.mu_m(sinal, mes):.3f}")
    print(f"Desvio padrão: {pg.sigma_m(sinal, mes):.3f}")
    print(f"Assimetria: {pg.assim_m(sinal, mes):.3f}")

# Carregar dados
pwd = "ena_hist_subsystem.csv"
data = pd.read_csv(pwd, delimiter=';')
data['eventdate'] = pd.to_datetime(data['eventdate'])
sinal = data['ena_mwmed_sul'].values

# Preparar dados
meses = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 
         'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dec']
n_anos = len(sinal) // 12
anos = range(1931, 1931 + n_anos)

# 1. SÉRIE TEMPORAL COMPLETA
plt.figure(figsize=(15, 8))
plt.plot(data['eventdate'], sinal, alpha=0.7, linewidth=1, label='ENA Mensal', color='steelblue')
plt.axhline(y=pg.mu_global(sinal), color='red', linestyle='--', 
            linewidth=2, label=f'Média Global: {pg.mu_global(sinal):.1f} MWmed')
plt.fill_between(data['eventdate'], 
                 pg.mu_global(sinal) - pg.sigma_global(sinal),
                 pg.mu_global(sinal) + pg.sigma_global(sinal),
                 alpha=0.2, color='gray', label='±1 Desvio Padrão')
plt.title('Série Temporal da ENA - Subsistema Sul (1931-2010)', fontsize=16, fontweight='bold')
plt.xlabel('Ano', fontsize=14)
plt.ylabel('ENA (MWmed)', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('serie_temporal_ena.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. HISTOGRAMA GLOBAL
plt.figure(figsize=(12, 8))
plt.hist(sinal, bins=50, alpha=0.7, density=True, 
         color='lightblue', edgecolor='black', linewidth=0.5)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 1000)
mu, sigma = pg.mu_global(sinal), pg.sigma_global(sinal)
p = pg.norm.pdf(x, mu, sigma)
plt.plot(x, p, 'r-', linewidth=3, label=f'Normal(μ={mu:.1f}, σ={sigma:.1f})')
plt.axvline(mu, color='red', linestyle='--', linewidth=2, label=f'Média: {mu:.1f}')
plt.title('Distribuição da ENA Mensal - Histograma Global', fontsize=16, fontweight='bold')
plt.xlabel('ENA (MWmed)', fontsize=14)
plt.ylabel('Densidade de Probabilidade', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('histograma_global_ena.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. BOXPLOT POR MÊS
dados_por_mes = [sinal[i::12] for i in range(12)]
medias_mensais = [pg.mu_m(sinal, i+1) for i in range(12)]

plt.figure(figsize=(14, 8))
box = plt.boxplot(dados_por_mes, labels=meses, patch_artist=True)
# Colorir os boxes
colors = plt.cm.Set3(np.linspace(0, 1, 12))
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
plt.plot(range(1, 13), medias_mensais, 'ro-', linewidth=3, 
         markersize=8, label='Médias Mensais', markerfacecolor='red')
plt.title('Distribuição da ENA por Mês - Boxplot', fontsize=16, fontweight='bold')
plt.xlabel('Mês', fontsize=14)
plt.ylabel('ENA (MWmed)', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('boxplot_mensal_ena.png', dpi=300, bbox_inches='tight')
plt.show()

# 4. ESTATÍSTICAS MENSIAIS
estatisticas_mensais = {
    'Mês': meses,
    'Média': [pg.mu_m(sinal, i+1) for i in range(12)],
    'Desvio Padrão': [pg.sigma_m(sinal, i+1) for i in range(12)],
    'Assimetria': [pg.assim_m(sinal, i+1) for i in range(12)]
}

df_estatisticas = pd.DataFrame(estatisticas_mensais)

# 4.1 MÉDIAS MENSIAIS
plt.figure(figsize=(12, 6))
bars = plt.bar(meses, df_estatisticas['Média'], alpha=0.8, color='skyblue', edgecolor='black')
plt.axhline(y=pg.mu_global(sinal), color='red', linestyle='--', 
            linewidth=2, label=f'Média Global: {pg.mu_global(sinal):.1f}')
plt.title('Médias Mensais da ENA', fontsize=16, fontweight='bold')
plt.xlabel('Mês', fontsize=14)
plt.ylabel('ENA Média (MWmed)', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('medias_mensais_ena.png', dpi=300, bbox_inches='tight')
plt.show()

# 4.2 DESVIOS PADRÃO MENSIAIS
plt.figure(figsize=(12, 6))
bars = plt.bar(meses, df_estatisticas['Desvio Padrão'], alpha=0.8, color='orange', edgecolor='black')
plt.title('Desvios Padrão Mensais da ENA', fontsize=16, fontweight='bold')
plt.xlabel('Mês', fontsize=14)
plt.ylabel('Desvio Padrão (MWmed)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('desvios_mensais_ena.png', dpi=300, bbox_inches='tight')
plt.show()

# 4.3 ASSIMETRIA MENSAL
plt.figure(figsize=(12, 6))
bars = plt.bar(meses, df_estatisticas['Assimetria'], alpha=0.8, color='lightgreen', edgecolor='black')
plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
plt.title('Assimetria Mensal da ENA', fontsize=16, fontweight='bold')
plt.xlabel('Mês', fontsize=14)
plt.ylabel('Coeficiente de Assimetria', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('assimetria_mensal_ena.png', dpi=300, bbox_inches='tight')
plt.show()

# 5. SÉRIE ANUAL COM TENDÊNCIA
medias_anuais = [pg.mu_t(sinal, t) for t in range(n_anos)]

plt.figure(figsize=(15, 8))
plt.plot(anos, medias_anuais, 'o-', linewidth=2, markersize=5, 
         color='blue', label='Média Anual', alpha=0.8)
plt.axhline(y=pg.mu_global(sinal), color='red', linestyle='--', 
            linewidth=2, label=f'Média Global: {pg.mu_global(sinal):.1f}')

# Linha de tendência
z = np.polyfit(anos, medias_anuais, 1)
p = np.poly1d(z)
plt.plot(anos, p(anos), 'r-', linewidth=3, 
         label=f'Tendência: y = {z[0]:.3f}x + {z[1]:.1f}')

plt.title('Média Anual da ENA com Tendência Linear', fontsize=16, fontweight='bold')
plt.xlabel('Ano', fontsize=14)
plt.ylabel('ENA Média Anual (MWmed)', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('serie_anual_tendencia_ena.png', dpi=300, bbox_inches='tight')
plt.show()

# 6. MATRIZ DE CORRELAÇÃO ENTRE MESES
corr_matrix = np.zeros((12, 12))
for i in range(12):
    for j in range(12):
        corr_matrix[i, j] = np.corrcoef(sinal[i::12], sinal[j::12])[0, 1]

plt.figure(figsize=(12, 10))
im = plt.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
plt.colorbar(im, label='Coeficiente de Correlação', shrink=0.8)
plt.xticks(range(12), meses, rotation=45, fontsize=12)
plt.yticks(range(12), meses, fontsize=12)
plt.title('Matriz de Correlação entre Meses', fontsize=16, fontweight='bold')

# Adicionar valores na matriz
for i in range(12):
    for j in range(12):
        color = 'white' if abs(corr_matrix[i, j]) > 0.5 else 'black'
        plt.text(j, i, f'{corr_matrix[i, j]:.2f}', 
                 ha='center', va='center', fontsize=10, 
                 color=color, fontweight='bold')

plt.tight_layout()
plt.savefig('matriz_correlacao_meses.png', dpi=300, bbox_inches='tight')
plt.show()

# 7. COMPARAÇÃO DISTRIBUIÇÃO NORMAL vs REAL
plt.figure(figsize=(12, 8))
# Histograma dos dados
n, bins, patches = plt.hist(sinal, bins=50, alpha=0.7, density=True, 
                           color='lightblue', edgecolor='black', 
                           linewidth=0.5, label='Dados Reais')

# Distribuição normal teórica
x = np.linspace(min(sinal), max(sinal), 1000)
mu, sigma = pg.mu_global(sinal), pg.sigma_global(sinal)
plt.plot(x, norm.pdf(x, mu, sigma), 'r-', linewidth=3, 
         label=f'Distribuição Normal\n(μ={mu:.1f}, σ={sigma:.1f})')

plt.title('Comparação com Distribuição Normal', fontsize=16, fontweight='bold')
plt.xlabel('ENA (MWmed)', fontsize=14)
plt.ylabel('Densidade de Probabilidade', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('comparacao_normal_ena.png', dpi=300, bbox_inches='tight')
plt.show()

# 8. SÉRIE TEMPORAL POR DÉCADAS
data['ano'] = data['eventdate'].dt.year
data['decada'] = (data['ano'] // 10) * 10

plt.figure(figsize=(15, 10))
decadas = sorted(data['decada'].unique())
colors = plt.cm.viridis(np.linspace(0, 1, len(decadas)))

for i, decada in enumerate(decadas):
    mask = data['decada'] == decada
    plt.plot(data.loc[mask, 'eventdate'], 
             data.loc[mask, 'ena_mwmed_sul'], 
             alpha=0.7, linewidth=1, 
             label=f'{decada}s', color=colors[i])

plt.axhline(y=pg.mu_global(sinal), color='red', linestyle='--', 
            linewidth=2, label=f'Média Global')
plt.title('Série Temporal da ENA por Década', fontsize=16, fontweight='bold')
plt.xlabel('Ano', fontsize=14)
plt.ylabel('ENA (MWmed)', fontsize=14)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('serie_por_decada_ena.png', dpi=300, bbox_inches='tight')
plt.show()

# RESULTADOS NUMÉRICOS
print("="*70)
print("ANÁLISE ESTATÍSTICA DA ENA - SUBSISTEMA SUL")
print("="*70)
print(f"Período analisado: {1931}-{2010} ({n_anos} anos completos)")
print(f"Total de meses: {len(sinal)}")
print(f"Média Global: {pg.mu_global(sinal):.1f} MWmed")
print(f"Desvio Padrão Global: {pg.sigma_global(sinal):.1f} MWmed")
print(f"Coeficiente de Variação: {(pg.sigma_global(sinal)/pg.mu_global(sinal)*100):.1f}%")
print(f"Assimetria Global: {pg.assim_global(sinal):.3f}")
print(f"Correlação Serial Anual: {pg.correlacao(sinal):.3f}")
print(f"Mês com maior média: {meses[np.argmax(medias_mensais)]} ({max(medias_mensais):.1f} MWmed)")
print(f"Mês com menor média: {meses[np.argmin(medias_mensais)]} ({min(medias_mensais):.1f} MWmed)")
print(f"Mês com maior variabilidade: {meses[np.argmax(df_estatisticas['Desvio Padrão'])]}")
print(f"Mês com maior assimetria: {meses[np.argmax(df_estatisticas['Assimetria'])]}")
print(f"Inclinação da tendência: {z[0]:.4f} MWmed/ano")

# Salvar estatísticas em CSV
df_estatisticas.to_csv('estatisticas_mensais_ena.csv', index=False, float_format='%.3f')
print("\nEstatísticas mensais salvas em 'estatisticas_mensais_ena.csv'")