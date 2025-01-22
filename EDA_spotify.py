# %% [markdown]
# # Análise Exploratória de Dados - Spotify Most Streamed Songs
# 
# Este notebook analisa dados das músicas mais tocadas no Spotify.
# O objetivo é entender padrões e características dessas músicas.

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
    df['in_deezer_playlists'] = pd.to_numeric(df['in_deezer_playlists'], errors='coerce')
    df['in_deezer_playlists'] = df['in_deezer_playlists'].fillna(df['in_deezer_playlists'].median())
    
    # Converter in_shazam_charts
    df['in_shazam_charts'] = pd.to_numeric(df['in_shazam_charts'], errors='coerce')
    df['in_shazam_charts'] = df['in_shazam_charts'].fillna(df['in_shazam_charts'].median())
    
    # 3. Converter colunas de porcentagem para float
    colunas_porcentagem = [col for col in df.columns if '%' in col]
    for col in colunas_porcentagem:
        df[col] = df[col].astype(float)
    
    # 4. Tratar valores categóricos
    df['key'] = df['key'].fillna(df['key'].mode()[0])
    
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
# ## 7. Próximos Passos
# - Análise de correlações
# - Visualizações dos dados
# - Identificação de padrões
# - Insights sobre as músicas mais populares
# %%
