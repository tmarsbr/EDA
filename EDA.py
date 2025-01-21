# %%
# EDA spotipy data
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# %%
df = pd.read_csv(r'C:\Users\tiago\OneDrive\Área de Trabalho\meus cursos\EDA\Spotify Most Streamed Songs.csv')
df.head()
# %%
df.shape
# %%
df.info()
# %%
df.describe()
# %%
