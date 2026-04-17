# noshow-predictor / notebooks / 02_eda.py
# Parte 3 — Análise Exploratória de Dados (EDA)
# Execute com: python notebooks/02_eda.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ── CONFIGURAÇÃO ───────────────────────────────────────────────── 
CAMINHO_DATASET = Path('data') / 'raw' / 'KaggleV2-May-2016.csv'
PASTA_ASSETS    = Path('assets')
PASTA_ASSETS.mkdir(exist_ok=True)

sns.set_theme(style='whitegrid', palette='Blues_d') 
plt.rcParams['figure.dpi'] = 120
plt.rcParams['font.family'] = 'sans-serif'
if not CAMINHO_DATASET.exists():
    print(f'ERRO: dataset não encontrado em {CAMINHO_DATASET}') 
    print('Execute a partir da raiz do projeto.')
    exit(1)

df = pd.read_csv(CAMINHO_DATASET)

# Converter No-show para numérico (antecipando o pré-processamento) 
df['noshow'] = (df['No-show'] == 'Yes').astype(int)

# Calcular dias de antecedência (feature importante!)
df['ScheduledDay']    = pd.to_datetime(df['ScheduledDay']) 
df['AppointmentDay']  = pd.to_datetime(df['AppointmentDay']) 
df['dias_antecedencia'] = (df['AppointmentDay'] - df['ScheduledDay']).dt.days

print('Dataset carregado e preparado.')
print(f'  Taxa geral de no-show: {df["noshow"].mean()*100:.1f}%')
print()

# ── GRÁFICO 1 — DISTRIBUIÇÃO DA VARIÁVEL-ALVO ──────────────────── 
print('Gerando gráfico 1 — Distribuição da variável-alvo...')
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

contagem   = df['noshow'].value_counts()
percentual = df['noshow'].value_counts(normalize=True) * 100

axes[0].bar(['Compareceu (0)', 'Não compareceu (1)'],
            contagem.values, color=['#2E86C1', '#C0392B'], edgecolor='white') 
axes[0].set_title('Contagem absoluta', fontsize=13, fontweight='bold') 
axes[0].set_ylabel('Número de agendamentos')
for i, v in enumerate(contagem.values):
    axes[0].text(i, v + 500, f'{v:,}', ha='center', fontweight='bold')
    
axes[1].bar(['Compareceu (0)', 'Não compareceu (1)'],
            percentual.values, color=['#2E86C1', '#C0392B'], edgecolor='white')
axes[1].set_title('Proporção (%)', fontsize=13, fontweight='bold') 
axes[1].set_ylabel('Percentual (%)')
axes[1].set_ylim(0, 100)
for i, v in enumerate(percentual.values):
    axes[1].text(i, v + 1, f'{v:.1f}%', ha='center', fontweight='bold')
    
plt.suptitle('Distribuição da Variável-Alvo: No-Show', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(PASTA_ASSETS / 'grafico_01_distribuicao_alvo.png', bbox_inches='tight')
plt.close()

# ── GRÁFICO 2 — DISTRIBUIÇÃO DE IDADES ─────────────────────────── 
print('Gerando gráfico 2 — Distribuição de idades...')
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(df['Age'], bins=50, color='#2E86C1', edgecolor='white', alpha=0.8)
axes[0].set_title('Distribuição de Idades (geral)', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Idade (anos)')
axes[0].set_ylabel('Frequência')
axes[0].axvline(df['Age'].mean(), color='red', linestyle='--', label=f'Média: {df["Age"].mean():.1f}')
axes[0].legend()

df[df['noshow'] == 0]['Age'].hist(bins=40, ax=axes[1], alpha=0.6, color='#2E86C1', label='Compareceu', density=True)
df[df['noshow'] == 1]['Age'].hist(bins=40, ax=axes[1], alpha=0.6, color='#C0392B', label='Não compareceu', density=True) 
axes[1].set_title('Distribuição de Idades por Grupo', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Idade (anos)')
axes[1].set_ylabel('Densidade')
axes[1].legend()

plt.suptitle('Análise de Idade dos Pacientes', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(PASTA_ASSETS / 'grafico_02_distribuicao_idade.png', bbox_inches='tight')
plt.close()

# ── GRÁFICO 3 — DIAS DE ANTECEDÊNCIA ───────────────────────────── 
print('Gerando gráfico 3 — Dias de antecedência...')
df_valido = df[df['dias_antecedencia'] >= 0].copy()

bins_ant   = [0, 1, 7, 15, 30, 60, 200]
labels_ant = ['0-1d', '2-7d', '8-15d', '16-30d', '31-60d', '60+d'] 
df_valido['faixa'] = pd.cut(df_valido['dias_antecedencia'],
                            bins=bins_ant, labels=labels_ant)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

grupos = [df_valido[df_valido['noshow'] == 0]['dias_antecedencia'], df_valido[df_valido['noshow'] == 1]['dias_antecedencia']]
bp = axes[0].boxplot(grupos, labels=['Compareceu', 'Não compareceu'], patch_artist=True) 
bp['boxes'][0].set_facecolor('#2E86C1') 
bp['boxes'][1].set_facecolor('#C0392B')
axes[0].set_title('Antecedência por Grupo', fontsize=13, fontweight='bold') 
axes[0].set_ylabel('Dias entre agendamento e consulta')

taxa_faixa = df_valido.groupby('faixa', observed=True)['noshow'].mean() * 100 
axes[1].bar(taxa_faixa.index, taxa_faixa.values, color='#2E86C1', edgecolor='white')
axes[1].set_title('Taxa de No-Show por Antecedência', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Dias de antecedência')
axes[1].set_ylabel('Taxa de no-show (%)')
for i, v in enumerate(taxa_faixa.values):
    axes[1].text(i, v + 0.5, f'{v:.1f}%', ha='center', fontsize=10)
    
plt.suptitle('Impacto do Tempo de Antecedência no No-Show', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(PASTA_ASSETS / 'grafico_03_antecedencia.png', bbox_inches='tight')
plt.close()

# ── GRÁFICO 4 — PARADOXO DO SMS ────────────────────────────────── 
print('Gerando gráfico 4 — Paradoxo do SMS...')
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

taxa_sms = df.groupby('SMS_received')['noshow'].mean() * 100 
axes[0].bar(['Não recebeu SMS', 'Recebeu SMS'], taxa_sms.values, color=['#2E86C1', '#C0392B'], edgecolor='white', width=0.5)
axes[0].set_title('Taxa de No-Show por SMS', fontsize=13, fontweight='bold') 
axes[0].set_ylabel('Taxa de no-show (%)')
axes[0].set_ylim(0, 35)
for i, v in enumerate(taxa_sms.values):
    axes[0].text(i, v + 0.5, f'{v:.1f}%', ha='center', fontweight='bold', fontsize=12)

df_valido['recebeu_sms'] = df_valido['SMS_received'].map({0: 'Não', 1: 'Sim'})
taxa_sms_ant = df_valido.groupby(['faixa', 'recebeu_sms'], observed=True)['noshow'].mean().unstack() * 100
taxa_sms_ant.plot(kind='bar', ax=axes[1], color=['#2E86C1', '#C0392B'], edgecolor='white', width=0.7)
axes[1].set_title('No-Show por Antecedência e SMS', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Faixa de antecedência')
axes[1].set_ylabel('Taxa de no-show (%)')
axes[1].legend(title='Recebeu SMS', labels=['Não', 'Sim']) 
axes[1].tick_params(axis='x', rotation=30)

plt.suptitle('O Paradoxo do SMS', fontsize=14, fontweight='bold') 
plt.tight_layout()
plt.savefig(PASTA_ASSETS / 'grafico_04_paradoxo_sms.png', bbox_inches='tight')
plt.close()

# ── GRÁFICO 5 — NO-SHOW POR CONDIÇÃO DE SAÚDE ──────────────────── 
print('Gerando gráfico 5 — No-show por condição de saúde...')
condicoes = ['Hipertension', 'Diabetes', 'Alcoholism', 'Scholarship']
rotulos   = ['Hipertensão', 'Diabetes', 'Alcoolismo', 'Bolsa Família']

fig, axes = plt.subplots(2, 2, figsize=(13, 10))
axes = axes.flatten()

for i, (col, rotulo) in enumerate(zip(condicoes, rotulos)):
    taxa = df.groupby(col)['noshow'].mean() * 100
    bars = axes[i].bar(['Não tem', 'Tem'], taxa.values, color=['#2E86C1', '#C0392B'], edgecolor='white', width=0.5) 
    axes[i].set_title(f'No-Show: {rotulo}', fontsize=12, fontweight='bold') 
    axes[i].set_ylabel('Taxa de no-show (%)')
    axes[i].set_ylim(0, 35)
    
    for bar, v in zip(bars, taxa.values):
        axes[i].text(bar.get_x() + bar.get_width() / 2, v + 0.3, f'{v:.1f}%', ha='center', fontweight='bold')
        
plt.suptitle('Taxa de No-Show por Condição de Saúde / Perfil Social', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(PASTA_ASSETS / 'grafico_05_condicoes_saude.png', bbox_inches='tight')
plt.close()

# ── GRÁFICO 6 — HEATMAP DE CORRELAÇÃO ──────────────────────────── 
print('Gerando gráfico 6 — Heatmap de correlação...')
colunas_corr = ['noshow', 'Age', 'dias_antecedencia', 'SMS_received', 'Scholarship', 'Hipertension', 'Diabetes', 'Alcoholism', 'Handcap']

corr_matrix = df[colunas_corr].corr()
mask = np.zeros_like(corr_matrix, dtype=bool) 
mask[np.triu_indices_from(mask)] = True

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax, mask=mask, linewidths=0.5, square=True, cbar_kws={'shrink': 0.8})
ax.set_title('Heatmap de Correlação com a Variável-Alvo', fontsize=13, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(PASTA_ASSETS / 'grafico_06_heatmap_correlacao.png', bbox_inches='tight')
plt.close()

# ── GRÁFICO 7 — TOP BAIRROS ─────────────────────────────────────── 
print('Gerando gráfico 7 — Top bairros por taxa de no-show...')
bairros = df.groupby('Neighbourhood').agg(total=('noshow', 'count'), taxa =('noshow', 'mean')).reset_index()
bairros['taxa'] = bairros['taxa'] * 100
bairros_validos = bairros[bairros['total'] >= 100]
top_noshow = bairros_validos.nlargest(15, 'taxa')

fig, ax = plt.subplots(figsize=(12, 6))
ax.barh(top_noshow['Neighbourhood'], top_noshow['taxa'], color='#C0392B', edgecolor='white', alpha=0.85) 
ax.axvline(df['noshow'].mean() * 100, color='navy', linestyle='--', label=f'Média geral: {df["noshow"].mean()*100:.1f}%') 
ax.set_title('Top 15 Bairros — Maior Taxa de No-Show (mín. 100 agendamentos)', fontsize=12, fontweight='bold')
ax.set_xlabel('Taxa de no-show (%)')
ax.legend()
plt.tight_layout()
plt.savefig(PASTA_ASSETS / 'grafico_07_top_bairros.png', bbox_inches='tight') 
plt.close()

# ── CORRELAÇÕES COM A VARIÁVEL-ALVO ──────────────────────────────
print()
print('--- Correlações com noshow (ordenadas) ---')
corrs = corr_matrix['noshow'].drop('noshow').sort_values(ascending=False) 
print(corrs.round(4).to_string())

print()
print('=' * 60)
print('EDA concluída! 7 gráficos salvos em assets/') 
print('Prossiga para a Parte 4.')
print('=' * 60)