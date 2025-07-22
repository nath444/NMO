import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Configuration ---

FEATURES = ['BPM', 'Dance', 'Energy', 'Acoustic', 'Instrumental', 'Happy', 'Speech', 'Live', 'Loud (Db)']
PLAYLISTS = ['playlist1.csv', 'playlist2.csv', 'playlist3.csv', 'playlist4.csv']

PLAYLIST_META = {
    'playlist1.csv': {'name': 'Ambiance coolax', 'color': '#b7aa5b'},
    'playlist2.csv': {'name': 'Fréquence R.A.P', 'color': '#996f4d'},
    'playlist3.csv': {'name': 'Onde nocturne', 'color': '#1a1f2b'},
    'playlist4.csv': {'name': 'Station soirée', 'color': '#683e78'}
}

# --- Fonctions ---

def load_all_playlists():
    dataframes = []
    for file in PLAYLISTS:
        df = pd.read_csv(os.path.join("data", file))
        df['Playlist'] = PLAYLIST_META[file]['name']
        df = df[FEATURES + ['Playlist']].dropna()
        dataframes.append(df)
    return pd.concat(dataframes)

def plot_correlation_matrix(df):
    corr = df[FEATURES].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Matrice de Corrélation entre les caractéristiques musicales")
    plt.tight_layout()
    plt.show()

def plot_feature_distributions(df):
    palette = {PLAYLIST_META[f]['name']: PLAYLIST_META[f]['color'] for f in PLAYLISTS}
    for feature in FEATURES:
        plt.figure(figsize=(8, 4))
        sns.boxplot(x="Playlist", y=feature, data=df, palette=palette)
        plt.title(f"Distribution de « {feature} » par playlist")
        plt.tight_layout()
        plt.show()

def show_mean_profile(df):
    means = df.groupby("Playlist")[FEATURES].mean().T
    names = [PLAYLIST_META[f]['name'] for f in PLAYLISTS]
    colors = [PLAYLIST_META[f]['color'] for f in PLAYLISTS]
    means[names].plot(kind="bar", figsize=(12, 6), color=colors)
    plt.title("Moyenne des caractéristiques par playlist")
    plt.ylabel("Valeur moyenne")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# --- Exécution ---

def main():
    df = load_all_playlists()
    plot_correlation_matrix(df)
    plot_feature_distributions(df)
    show_mean_profile(df)

if __name__ == "__main__":
    main()
