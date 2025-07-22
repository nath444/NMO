import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial.distance import euclidean

# --- Configuration ---

FEATURES = ['BPM', 'Dance', 'Energy', 'Acoustic', 'Instrumental',
            'Happy', 'Speech', 'Live', 'Loud (Db)']

DATA_FOLDER = 'data'
PLAYLIST_INPUT_FOLDER = 'playlists'
THRESHOLD = 2.5

PLAYLIST_META = {
    'playlist1.csv': {'name': 'Ambiance coolax', 'color': '#cfbd48'},
    'playlist2.csv': {'name': 'Fréquence R.A.P', 'color': '#cc9257'},
    'playlist3.csv': {'name': 'Onde nocturne', 'color': '#203259'},
    'playlist4.csv': {'name': 'Station soirée', 'color': '#b353cf'}
}

# --- Fonctions ---

def list_input_files(folder):
    files = [f for f in os.listdir(folder) if f.endswith('.csv')]
    print("\nFichiers disponibles à classifier :")
    for i, f in enumerate(files):
        print(f"{i+1}. {f}")
    idx = int(input("\nEntrez le numéro du fichier à classifier : ")) - 1
    return os.path.join(folder, files[idx])

def load_and_process_csv(file_path):
    df = pd.read_csv(file_path)
    return df[FEATURES].dropna()

def normalize_data(playlists):
    scaler = StandardScaler()
    combined = pd.concat(playlists, ignore_index=True)
    scaler.fit(combined)
    return [pd.DataFrame(scaler.transform(p), columns=FEATURES) for p in playlists], scaler

def classify_song(song_vector, centroids, scaler, threshold=THRESHOLD):
    song_df = pd.DataFrame([song_vector], columns=FEATURES)
    norm_song = scaler.transform(song_df)[0]
    distances = [euclidean(norm_song, c) for c in centroids]
    min_distance = min(distances)
    best_index = np.argmin(distances)
    if min_distance <= threshold:
        return PLAYLIST_META[PLAYLIST_FILES[best_index]]['name'], min_distance, best_index
    else:
        return "NON CLASSÉE", min_distance, best_index

def get_playlist_labels():
    return [PLAYLIST_META[f]['name'] for f in PLAYLIST_FILES]

def plot_distance_bar(title, distances, best_index, i, mode='show'):
    playlist_names = get_playlist_labels()
    plt.figure(figsize=(8, 4))
    bars = plt.bar(playlist_names, distances, color=[PLAYLIST_META[f]['color'] for f in PLAYLIST_FILES])
    bars[best_index].set_edgecolor('black')
    bars[best_index].set_linewidth(2)
    plt.axhline(THRESHOLD, color='red', linestyle='--', label="Seuil de classification")
    plt.title(f"Distances pour « {title} »")
    plt.ylabel("Distance euclidienne")
    plt.legend()
    plt.tight_layout()
    if mode == 'save':
        plt.savefig(f"graph_morceaux/morceau_{i+1}_bar.png")
        plt.close()
    else:
        plt.show()

def plot_pca(playlists_raw, new_songs, scaler, centroids, mode='show'):
    print("Projection PCA des playlists et morceaux à classer...")

    df_all = pd.concat(playlists_raw, ignore_index=True)
    X = scaler.transform(df_all[FEATURES])
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    playlist_labels = []
    for i, p in enumerate(playlists_raw):
        playlist_labels.extend([PLAYLIST_META[PLAYLIST_FILES[i]]['name']] * len(p))

    pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
    pca_df["Playlist"] = playlist_labels

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="Playlist",
                    palette={PLAYLIST_META[f]['name']: PLAYLIST_META[f]['color'] for f in PLAYLIST_FILES})

    new_X = scaler.transform(new_songs[FEATURES])
    new_pca = pca.transform(new_X)

    for i, (x, y) in enumerate(new_pca):
        plt.scatter(x, y, marker="*", color="black", s=100)
        plt.text(x + 0.2, y, new_songs.iloc[i]["Song"], fontsize=8)

    plt.title("Projection PCA des playlists et nouveaux morceaux")
    plt.tight_layout()
    if mode == 'save':
        plt.savefig("graph_morceaux/projection_pca.png")
        plt.close()
    else:
        plt.show()

# --- Script principal ---

def main():
    global PLAYLIST_FILES
    PLAYLIST_FILES = [f for f in PLAYLIST_META.keys()]

    print("Chargement des playlists...")
    playlists_raw = [load_and_process_csv(os.path.join(DATA_FOLDER, f)) for f in PLAYLIST_FILES]
    playlists_norm, scaler = normalize_data(playlists_raw)
    centroids = [p.mean().values for p in playlists_norm]

    print("Sélection du fichier à classifier dans /playlists...")
    file_to_classify = list_input_files(PLAYLIST_INPUT_FOLDER)
    new_songs = pd.read_csv(file_to_classify)
    new_songs_clean = new_songs[FEATURES].dropna()

    print("Classification en cours...")
    results = []
    distances_list = []

    for i, row in new_songs_clean.iterrows():
        title = new_songs.loc[i, 'Song']
        artist = new_songs.loc[i, 'Artist']
        label, distance, best_index = classify_song(row.values, centroids, scaler)
        results.append({
            "Titre": title,
            "Artiste": artist,
            "Playlist assignée": label,
            "Distance": round(distance, 2)
        })
        distances = [euclidean(scaler.transform(pd.DataFrame([row], columns=FEATURES))[0], c) for c in centroids]
        distances_list.append((title, distances, best_index, i))

    results_df = pd.DataFrame(results)

    print("\nQue souhaitez-vous faire ?")
    print("1. Sauvegarder le fichier CSV des résultats")
    print("2. Visualiser les graphiques (fenêtres interactives)")
    print("3. Sauvegarder tous les graphiques dans graph_morceaux/")
    choice = input("Entrez 1, 2, 3 ou plusieurs séparés par une virgule : ").split(',')

    if '1' in choice:
        results_df.to_csv("resultats.csv", index=False)
        print("Fichier resultats.csv sauvegardé.")

    if '2' in choice:
        print("Affichage interactif des graphiques...")
        for title, distances, best_index, i in distances_list:
            plot_distance_bar(title, distances, best_index, i, mode='show')
        plot_pca(playlists_raw, new_songs, scaler, centroids, mode='show')

    if '3' in choice:
        os.makedirs("graph_morceaux", exist_ok=True)
        print("Sauvegarde des graphiques dans graph_morceaux/...")
        for title, distances, best_index, i in distances_list:
            plot_distance_bar(title, distances, best_index, i, mode='save')
        plot_pca(playlists_raw, new_songs, scaler, centroids, mode='save')
        print("Graphiques enregistrés dans le dossier graph_morceaux")

    print("Terminé.")

if __name__ == "__main__":
    main()
