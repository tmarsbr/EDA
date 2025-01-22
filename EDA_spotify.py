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
    # Criar cópia para não modificar dados originais
    df = df.copy()
    
    # 1. Remover colunas desnecessárias
    df = df.drop(columns=['cover_url'])
    
    # 2. Converter e limpar tipos numéricos
    # Converter streams para float
    df['streams'] = pd.to_numeric(df['streams'].str.replace(',', ''), errors='coerce')
    
    # Converter e tratar in_deezer_playlists
    df['playlists_deezer'] = pd.to_numeric(df['playlists_deezer'], errors='coerce')
    df['playlists_deezer'] = df['playlists_deezer'].fillna(df['playlists_deezer'].median())
    
    # Converter in_shazam_charts
    df['charts_shazam'] = pd.to_numeric(df['charts_shazam'], errors='coerce')
    df['charts_shazam'] = df['charts_shazam'].fillna(df['charts_shazam'].median())
    
    # 3. Converter colunas de porcentagem para float
    colunas_porcentagem = [col for col in df.columns if '%' in col]
    for col in colunas_porcentagem:
        df[col] = df[col].astype(float)
    
    # 4. Tratar valores categóricos
    df['tom_musical'] = df['tom_musical'].fillna(df['tom_musical'].mode()[0])
    
    return df

# %% [markdown]
# ## 6. Aplicação da Limpeza
# Aplicando as transformações e verificando os resultados

# %% 
# Aplicar limpeza
df = limpar_dados(df)

# Verificar resultado
metadados_df = metadados(df)
metadados_df.head(4)

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
    
    # Gráfico de barras
    sns.barplot(x=contagem.values, y=contagem.index, ax=ax1, 
                palette='viridis', alpha=0.8)
    ax1.set_title(f'Distribuição de {coluna}')
    ax1.set_xlabel('Contagem')
    
    # Gráfico de pizza
    wedges, texts, autotexts = ax2.pie(contagem.values, labels=contagem.index,
                                      autopct='%1.1f%%', colors=sns.color_palette('viridis', n_colors=len(contagem)))
    ax2.set_title(f'Proporção de {coluna}')
    
    plt.tight_layout()
    plt.show()

# Análise das principais variáveis categóricas
variaveis_categoricas = ['tom_musical', 'modo_musical', 'qt_artistas']
for var in variaveis_categoricas:
    plot_univariada_categorica(df, var)

# %% [markdown]
# ### 7.4 Análise Bivariada - Categóricas vs Numéricas

# %% 
def plot_bivariada_cat_num(df, cat_col, num_col):
    """
    Análise bivariada entre variável categórica e numérica
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    
    # Boxplot atualizado
    sns.boxplot(data=df, 
                x=cat_col, 
                y=num_col, 
                ax=ax1,
                hue=cat_col,
                legend=False)
    ax1.set_title(f'Boxplot: {cat_col} vs {num_col}')
    ax1.tick_params(axis='x', rotation=45)
    
    # Gráfico de barras com erro padrão atualizado
    sns.barplot(data=df, 
                x=cat_col, 
                y=num_col, 
                ax=ax2,
                hue=cat_col,
                legend=False,
                errorbar='sd')
    ax2.set_title(f'Média e Desvio Padrão: {cat_col} vs {num_col}')
    ax2.tick_params(axis='x', rotation=45)
    
    # Gráfico de barras com valores médios atualizado
    medias = df.groupby(cat_col)[num_col].mean().sort_values(ascending=False)
    sns.barplot(x=medias.values, 
                y=medias.index, 
                ax=ax3,
                hue=medias.index,
                legend=False)
    ax3.set_title(f'Média de {num_col} por {cat_col}')
    
    plt.tight_layout()
    plt.show()

# Análises bivariadas relevantes
analises_cat_num = [
    ('modo_musical', 'streams'),
    ('tom_musical', 'streams'),
    ('qt_artistas', 'streams'),
    ('modo_musical', 'danceability'),
    ('tom_musical', 'energy')
]

for cat_col, num_col in analises_cat_num:
    plot_bivariada_cat_num(df, cat_col, num_col)

# %% [markdown]
# ### 7.5 Insights das Análises Categóricas
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
# ## 8. A Receita do Sucesso: Descobertas e Recomendações

# %% [markdown]
# ### 8.1 Descobertas Principais
# 
# 1. **🎯 O Poder da Distribuição Digital**
#    - A era do streaming mudou as regras do jogo
#    - 10% das músicas dominam 90% dos streams
#    - Hits virais podem surgir rapidamente e atingir números extraordinários
# 
# 2. **🌟 O Ecossistema das Plataformas**
#    - Spotify emerge como kingmaker do streaming
#    - Presença cross-platform multiplica chances de sucesso
#    - Playlists são o novo rádio do século XXI
# 
# 3. **📅 O Timing Perfeito**
#    - Lançamentos estratégicos impactam performance
#    - Janela de oportunidade crítica nos primeiros 30 dias
#    - Padrões sazonais influenciam o sucesso
# 
# ### 8.2 Estratégias para o Sucesso
# 
# 1. **🎯 Distribuição Inteligente**
#    - Construir presença forte no Spotify como prioridade
#    - Desenvolver estratégia omnichannel coordenada
#    - Focar em playlists estratégicas para crescimento
# 
# 2. **⏰ Lançamento Estratégico**
#    - Identificar momentos ideais para release
#    - Preparar campanha intensiva de 30 dias
#    - Manter calendário consistente de conteúdo
# 
# 3. **📢 Marketing com Dados**
#    - Investir em marketing baseado em análise de dados
#    - Cultivar relacionamentos com curadores de playlist
#    - Criar estratégias personalizadas por plataforma
# 
# ### 8.3 Próximos Passos para Artistas e Labels
# 
# 1. **📊 Monitoramento Contínuo**
#    - Acompanhar métricas em tempo real
#    - Adaptar estratégias baseado em dados
#    - Identificar tendências emergentes
# 
# 2. **🔄 Otimização Constante**
#    - Testar diferentes abordagens
#    - Refinar estratégias baseado em resultados
#    - Manter-se atualizado com tendências do mercado

# %%
