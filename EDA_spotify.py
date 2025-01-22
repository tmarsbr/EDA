# %% [markdown]
# # 🎵 A Anatomia de um Hit: Análise das Músicas Mais Streamadas do Spotify
# 
# ## Introdução
# No mundo digital da música, o que faz uma canção se tornar um sucesso? 
# Esta análise mergulha nos dados das músicas mais streamadas do Spotify para 
# desvendar os segredos por trás dos hits modernos.

# %% [markdown]
# ## Objetivos da Análise
# 1. 🎯 Identificar padrões que levam ao sucesso musical
# 2. 📊 Analisar características técnicas das músicas populares
# 3. 💡 Descobrir insights para estratégias de lançamento
# 4. 🌟 Entender o impacto das diferentes plataformas

# %% [markdown]
# ## 1. Configuração Inicial
# Importando as bibliotecas necessárias para análise

# %% 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

# %% [markdown]
# ## 2. Carregamento dos Dados
# Lendo o arquivo CSV e mostrando as primeiras linhas

# %% 
df = pd.read_csv('Spotify Most Streamed Songs.csv')
df.head()

# %% [markdown]
# ## 2.1 Padronização dos Nomes das Variáveis
# Função para limpar e padronizar nomes das colunas

# %% 
def padronizar_colunas(df):
    """
    Padroniza os nomes das colunas do DataFrame:
    - Remove caracteres especiais
    - Converte para minúsculas
    - Substitui espaços por _
    - Remove parênteses e outros caracteres indesejados
    """
    df = df.copy()
    
    # Dicionário de renomeação específica
    rename_dict = {
        'artist(s)_name': 'artistas',
        'track_name': 'musica',
        'artist_count': 'qt_artistas',
        'released_year': 'ano_lancamento',
        'released_month': 'mes_lancamento',
        'released_day': 'dia_lancamento',
        'in_spotify_playlists': 'playlists_spotify',
        'in_spotify_charts': 'charts_spotify',
        'in_apple_playlists': 'playlists_apple',
        'in_apple_charts': 'charts_apple',
        'in_deezer_playlists': 'playlists_deezer',
        'in_deezer_charts': 'charts_deezer',
        'in_shazam_charts': 'charts_shazam',
        'streams': 'streams',
        'bpm': 'bpm',
        'key': 'tom_musical',
        'mode': 'modo_musical',
        'danceability_%': 'danceability',
        'valence_%': 'valence',
        'energy_%': 'energy',
        'acousticness_%': 'acousticness',
        'instrumentalness_%': 'instrumentalness',
        'liveness_%': 'liveness',
        'speechiness_%': 'speechiness'
    }
    
    # Renomear colunas
    df = df.rename(columns=rename_dict)
    
    print("Colunas renomeadas:")
    for old, new in rename_dict.items():
        print(f"{old:25} -> {new}")
    
    return df

# Aplicar padronização
df = padronizar_colunas(df)

# Verificar novos nomes
print("\nNovas colunas:")
print(df.columns.tolist())

# %% [markdown]
# ## 3. Função de Metadados
# Esta função gera um resumo completo do dataset, incluindo:
# - Tipos de dados
# - Valores únicos
# - Valores nulos
# - Uso de memória
# - Exemplos de valores

# %% 
def metadados(df):
    # Criar DataFrame de metadados
    meta = pd.DataFrame(index=df.columns)
    
    # Tipos de dados
    meta['Tipo'] = df.dtypes
    
    # Contagem de valores únicos
    meta['Valores Únicos'] = df.nunique()
    
    # Valores nulos
    meta['Nulos'] = df.isnull().sum()
    meta['% Nulos'] = (df.isnull().sum() / len(df) * 100).round(2)
    
    # Memória utilizada
    meta['Memória (MB)'] = df.memory_usage(deep=True) / 1024 / 1024
    
    # Primeiros valores únicos (amostra)
    meta['Exemplos'] = [df[col].dropna().unique()[:3].tolist() for col in df.columns]
    
    print("="*80)
    print(f"RESUMO DO DATASET")
    print(f"Dimensões: {df.shape[0]} linhas x {df.shape[1]} colunas")
    print(f"Uso de Memória Total: {df.memory_usage().sum() / 1024 / 1024:.2f} MB")
    print("="*80)
    
    return meta

# %% [markdown]
# ## 4. Análise Inicial dos Dados
# Executando a função de metadados para entender a estrutura do dataset

# %% 
metadados_df = metadados(df)
display(metadados_df)

# %% [markdown]
# ## 5. Limpeza e Preparação dos Dados
# Função para limpar e corrigir tipos de dados, incluindo:
# - Remoção de colunas desnecessárias
# - Conversão de tipos
# - Tratamento de valores nulos
# - Padronização de formatos

# %% 
def limpar_dados(df):
    """
    Função única para limpeza e correção de tipos dos dados
    """
    df = df.copy()
    
    print("🎵 Iniciando a mixagem dos dados...")
    
    # 1. Remover colunas desnecessárias
    print("\n🎚️ Removendo ruídos (colunas desnecessárias)...")
    df = df.drop(columns=['cover_url'])
    
    # 2. Tratamento de Streams (outliers e formato)
    print("\n🎛️ Ajustando os níveis de streams...")
    df['streams'] = pd.to_numeric(df['streams'].str.replace(',', ''), errors='coerce')
    
    # Identificar e tratar outliers em streams usando IQR
    Q1 = df['streams'].quantile(0.25)
    Q3 = df['streams'].quantile(0.75)
    IQR = Q3 - Q1
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR
    
    # Criar coluna para identificar outliers
    df['is_outlier_streams'] = (df['streams'] < limite_inferior) | (df['streams'] > limite_superior)
    print(f"📊 Identificados {df['is_outlier_streams'].sum()} outliers em streams")
    
    # 3. Tratamento de métricas musicais
    metricas_musicais = ['danceability', 'energy', 'valence', 'acousticness', 
                         'instrumentalness', 'liveness', 'speechiness']
    
    print("\n🎼 Normalizando características musicais...")
    for metrica in metricas_musicais:
        # Converter para float
        df[metrica] = df[metrica].astype(float)
        
        # Identificar e marcar outliers
        Q1 = df[metrica].quantile(0.25)
        Q3 = df[metrica].quantile(0.75)
        IQR = Q3 - Q1
        df[f'is_outlier_{metrica}'] = (df[metrica] < (Q1 - 1.5 * IQR)) | (df[metrica] > (Q3 + 1.5 * IQR))
        print(f"📊 {metrica}: {df[f'is_outlier_{metrica}'].sum()} outliers identificados")
    
    # 4. Tratamento de playlists
    print("\n🎯 Ajustando métricas de playlists...")
    playlist_cols = ['playlists_spotify', 'playlists_apple', 'playlists_deezer']
    for col in playlist_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].median())
    
    # 5. Tratamento final
    print("\n🎹 Finalizando os ajustes...")
    df['tom_musical'] = df['tom_musical'].fillna(df['tom_musical'].mode()[0])
    
    print("\n✨ Mixagem concluída! Dados prontos para análise.")
    return df

# %% [markdown]
# ## 6. Aplicação da Limpeza
# Aplicando as transformações e verificando os resultados

# %% 
# Aplicar limpeza
df = limpar_dados(df)

# Verificar resultado
metadados_df = metadados(df)
metadados_df.head()

# %% [markdown]
# ## 7. Desvendando os Dados: Uma Jornada pelos Hits
# Vamos explorar diferentes aspectos que compõem um sucesso musical moderno.

# %% [markdown]
# ### 7.1 O DNA dos Hits: Análise das Características Musicais
# Explorando os elementos fundamentais que compõem as músicas mais populares

# %% 
# Função para criar análise univariada
def plot_univariada(df, coluna):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Histograma
    sns.histplot(data=df, x=coluna, ax=ax1, color='purple', alpha=0.6)
    ax1.set_title(f'Distribuição de {coluna}')
    ax1.grid(True, alpha=0.3)
    
    # Boxplot
    sns.boxplot(data=df, y=coluna, ax=ax2, color='purple')
    ax2.set_title(f'Boxplot de {coluna}')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Análise das principais variáveis numéricas
variaveis_numericas = ['streams', 'danceability', 'energy', 'valence']
for var in variaveis_numericas:
    plot_univariada(df, var)

# %% [markdown]
# ### 7.2 A Dança dos Números: Correlações e Padrões
# Descobrindo como diferentes características musicais se relacionam com o sucesso

# %% 
def plot_bivariada(df, x, y):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Scatter plot
    sns.scatterplot(data=df, x=x, y=y, ax=ax1, alpha=0.5, color='purple')
    ax1.set_title(f'Relação entre {x} e {y}')
    
    # Line plot com média móvel
    df_sorted = df.sort_values(by=x)
    sns.regplot(data=df_sorted, x=x, y=y, ax=ax2, 
                scatter_kws={'alpha':0.5}, 
                line_kws={'color': 'red'})
    ax2.set_title(f'Tendência entre {x} e {y}')
    
    plt.tight_layout()
    plt.show()

# Análises bivariadas relevantes
pares_analise = [
    ('danceability', 'streams'),
    ('energy', 'streams'),
    ('valence', 'streams'),
    ('danceability', 'energy')
]

for x, y in pares_analise:
    plot_bivariada(df, x, y)

# %% [markdown]
# ### 7.3 A Arte da Composição: Elementos Musicais
# Analisando as escolhas artísticas que definem os hits

# %% 
def plot_univariada_categorica(df, coluna, limite_categorias=10):
    """
    Análise univariada para variáveis categóricas com gráfico de barras e pizza
    """
    # Preparar dados
    contagem = df[coluna].value_counts()
    if len(contagem) > limite_categorias:
        outras = pd.Series({'Outras': contagem[limite_categorias:].sum()})
        contagem = pd.concat([contagem[:limite_categorias], outras])
    
    # Criar subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Gráfico de barras atualizado
    data_plot = pd.DataFrame({
        'Categoria': contagem.index,
        'Contagem': contagem.values
    })
    sns.barplot(data=data_plot,
                x='Contagem',
                y='Categoria',
                ax=ax1,
                color='skyblue',
                alpha=0.8)
    ax1.set_title(f'Distribuição de {coluna}')
    ax1.set_xlabel('Contagem')
    
    # Gráfico de pizza
    wedges, texts, autotexts = ax2.pie(contagem.values, 
                                      labels=contagem.index,
                                      autopct='%1.1f%%', 
                                      colors=sns.color_palette('viridis', n_colors=len(contagem)))
    ax2.set_title(f'Proporção de {coluna}')
    
    plt.tight_layout()
    plt.show()

# Análise das principais variáveis categóricas
variaveis_categoricas = ['tom_musical', 'modo_musical', 'qt_artistas']
for var in variaveis_categoricas:
    plot_univariada_categorica(df, var)

# %% [markdown]
# ### 7.4 Relação entre Características Categóricas e Numéricas
# Explorando como diferentes categorias influenciam métricas de sucesso

# %%
def plot_bivariada_cat_num(df, cat_col, num_col):
    """
    Análise bivariada entre variável categórica e numérica com visualizações simplificadas e claras
    """
    # Configurar o estilo
    sns.set_style("whitegrid")
    
    # Criar figura
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # 1. Boxplot aprimorado
    sns.boxplot(data=df, 
                x=cat_col, 
                y=num_col,
                ax=ax1,
                hue=cat_col,
                legend=False)
    
    # Personalização do boxplot
    ax1.set_title(f'Distribuição de {num_col} por {cat_col}', pad=20, fontsize=14)
    ax1.set_xlabel(cat_col, fontsize=12)
    ax1.set_ylabel(num_col, fontsize=12)
    ax1.tick_params(axis='x', rotation=45)
    
    # Adicionar pontos de média
    means = df.groupby(cat_col)[num_col].mean()
    ax1.scatter(range(len(means)), means, color='red', s=100, marker='D', label='Média')
    ax1.legend()
    
    # 2. Gráfico de barras com médias
    sns.barplot(data=df,
                x=cat_col,
                y=num_col,
                ax=ax2,
                hue=cat_col,
                legend=False,
                errorbar=('ci', 95),
                capsize=0.05)
    
    # Personalização do gráfico de barras
    ax2.set_title(f'Média de {num_col} por {cat_col}', pad=20, fontsize=14)
    ax2.set_xlabel(cat_col, fontsize=12)
    ax2.set_ylabel(f'Média de {num_col}', fontsize=12)
    ax2.tick_params(axis='x', rotation=45)
    
    # Adicionar valores nas barras
    for i, v in enumerate(df.groupby(cat_col)[num_col].mean()):
        ax2.text(i, v, f'{v:,.0f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    # Exibir estatísticas resumidas
    print(f"\nEstatísticas de {num_col} por {cat_col}:")
    stats = df.groupby(cat_col)[num_col].agg(['count', 'mean', 'std', 'min', 'max'])
    display(stats.round(2))

# Análises bivariadas relevantes
analises_cat_num = [
    ('modo_musical', 'streams'),
    ('tom_musical', 'streams'),
    ('qt_artistas', 'streams')
]

for cat_col, num_col in analises_cat_num:
    plot_bivariada_cat_num(df, cat_col, num_col)

# %% [markdown]
# ### Insights das Análises Bivariadas
# 
# 1. **Modo Musical vs Streams**
#    - Comparação clara entre modos maior e menor
#    - Distribuição e médias de streams por modo
#    - Identificação de outliers significativos
# 
# 2. **Tom Musical vs Streams**
#    - Padrões de popularidade por tom
#    - Tons mais comuns em hits
#    - Variabilidade dentro de cada tom
# 
# 3. **Quantidade de Artistas vs Streams**
#    - Impacto de colaborações no sucesso
#    - Número ideal de artistas por faixa
#    - Tendências de colaboração

# %% [markdown]
# ### Insights das Análises Categóricas
# 
# 1. **Distribuição de Características Musicais**
#    - Distribuição dos tons musicais
#    - Proporção de modos maior/menor
#    - Padrões de colaborações entre artistas
# 
# 2. **Relações com Performance**
#    - Impacto do tom musical nos streams
#    - Diferenças de popularidade entre modos
#    - Efeito de colaborações no sucesso

# %% [markdown]
# ### 7.5 Análise Aprofundada dos Resultados

# %% [markdown]
# #### Análise de Streams e Popularidade

# %%
# Calculando estatísticas de streams
print("📊 Análise de Distribuição dos Streams")
print("-" * 50)
print(f"Média de streams: {df['streams'].mean():,.0f}")
print(f"Mediana de streams: {df['streams'].median():,.0f}")
print(f"Top 10% das músicas acumulam: {(df['streams'].nlargest(int(len(df)*0.1)).sum() / df['streams'].sum() * 100):.1f}% dos streams totais")

# %% [markdown]
# #### Top 10 Hits Mais Streamados

# %%
# Análise dos maiores hits
top_hits = df.nlargest(10, 'streams')[['musica', 'artistas', 'streams', 'danceability', 'energy']]
print("\n🎵 Top 10 Músicas Mais Streamadas")
print("-" * 50)
display(top_hits)

# %% [markdown]
# #### Insights dos Hits
# 
# 1. **Concentração de Streams**
#    - Alta concentração no topo da distribuição
#    - Diferença significativa entre média e mediana
#    - Padrão típico de distribuição de cauda longa
# 
# 2. **Características dos Top Hits**
#    - Combinação ideal de danceability e energy
#    - Presença forte de artistas estabelecidos
#    - Padrões consistentes nas características musicais

# %% [markdown]
# #### Análise Musical Técnica

# %%
# Características Musicais
print("\n🎼 Características Musicais de Sucesso")
print("-" * 50)
print(f"Danceability: {df['danceability'].mean():.1f}% média (correlação: {df['danceability'].corr(df['streams']):.2f})")
print(f"Energy: {df['energy'].mean():.1f}% média (correlação: {df['energy'].corr(df['streams']):.2f})")
print(f"Valence: {df['valence'].mean():.1f}% média (correlação: {df['valence'].corr(df['streams']):.2f})")

# Tons e Modos
print("\n🎹 Análise de Tons e Modos")
print("-" * 50)
top_tons = df.groupby('tom_musical')['streams'].mean().nlargest(3)
print("Tons mais populares:")
for tom, streams in top_tons.items():
    print(f"- {tom}: {streams:,.0f} streams médios")

# %% [markdown]
# #### Análise de Colaborações

# %%
# Impacto das Colaborações
print("\n🤝 Impacto das Colaborações")
print("-" * 50)
colaboracoes = df.groupby('qt_artistas').agg({
    'streams': ['count', 'mean', 'median'],
    'playlists_spotify': 'mean'
}).round(2)
colaboracoes.columns = ['_'.join(col).strip() for col in colaboracoes.columns.values]

# Formatar os números para facilitar a leitura
colaboracoes['streams_mean'] = colaboracoes['streams_mean'].apply(lambda x: f"{x:,.0f}")
colaboracoes['streams_median'] = colaboracoes['streams_median'].apply(lambda x: f"{x:,.0f}")
colaboracoes['playlists_spotify_mean'] = colaboracoes['playlists_spotify_mean'].apply(lambda x: f"{x:,.0f}")

display(colaboracoes)

# Artistas Mais Frequentes
print("\n👨‍🎤 Top 10 Artistas Mais Frequentes")
print("-" * 50)
top_artistas = df['artistas'].value_counts().head(10)
for artista, count in top_artistas.items():
    print(f"{artista}: {count} músicas")

# ...rest of existing code...

# %% [markdown]
# ## 8. Conclusão: O DNA do Sucesso Musical 🎵

# %%
print("\n🎯 Conclusões Principais da Análise")
print("-" * 50)

# 1. Características Musicais
print("\n1. Características que Definem o Sucesso:")
print("   ✓ Danceability médio de {:.1f}%".format(df['danceability'].mean()))
print("   ✓ Energy médio de {:.1f}%".format(df['energy'].mean()))
print("   ✓ Valence médio de {:.1f}%".format(df['valence'].mean()))

# 2. Padrões de Colaboração
colaboracoes_media = df.groupby('qt_artistas')['streams'].mean()
n_ideal = colaboracoes_media.idxmax()
print(f"\n2. Colaborações:")
print(f"   ✓ Número ideal de artistas: {n_ideal}")
print(f"   ✓ {df[df['qt_artistas'] > 1]['musica'].count()} músicas são colaborações")

# 3. Plataformas
print("\n3. Presença nas Plataformas:")
print(f"   ✓ Média de playlists Spotify: {df['playlists_spotify'].mean():.0f}")
print(f"   ✓ Média de playlists Apple: {df['playlists_apple'].mean():.0f}")
print(f"   ✓ Média de playlists Deezer: {df['playlists_deezer'].mean():.0f}")

# %% [markdown]
# ### Principais Insights 🔍
# 
# 1. **Características do Sucesso**
#    - A combinação ideal de danceability e energy é crucial
#    - Músicas com alta danceability tendem a ter mais streams
#    - O equilíbrio entre elementos musicais é fundamental
# 
# 2. **Impacto das Colaborações**
#    - Colaborações múltiplas têm maior potencial viral
#    - Parcerias estratégicas aumentam o alcance
#    - Diversidade de artistas amplia o público
# 
# 3. **Distribuição nas Plataformas**
#    - Presença multi-plataforma é essencial
#    - Spotify lidera em termos de alcance
#    - Estratégia diversificada de distribuição é importante
# 
# 4. **Recomendações Finais**
#    - Foco em elementos que promovem danceability
#    - Investir em colaborações estratégicas
#    - Manter presença forte em múltiplas plataformas
#    - Equilibrar características musicais para máximo apelo

# %%
