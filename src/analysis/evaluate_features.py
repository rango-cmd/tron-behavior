import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

MATRIX_PATH = "data/features_matrix.csv"
OUTPUT_IMG_DIR = "data/"
OUTPUT_FIG_NAME = "features_cluster.png"

RF_ESTIMATORS = 100
RF_MAX_DEPTH = 8
TEST_SIZE = 0.2
RANDOM_STATE = 42

TSNE_PERPLEXITY = 30
TSNE_MAX_ITER = 1000

def main():
    if not os.path.exists(MATRIX_PATH):
        print(f"Error: {MATRIX_PATH} not found.")
        return

    df = pd.read_csv(MATRIX_PATH)
    X = df.drop(columns=["address", "seed_label"])
    y = df["seed_label"]
    feature_names = X.columns.tolist()

    print(f"Dataset Loaded: {X.shape[0]} samples, {X.shape[1]} features.")
    print(f"Class Distribution:\n{y.value_counts()}\n" + "-" * 50)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    rf = RandomForestClassifier(n_estimators=RF_ESTIMATORS, max_depth=RF_MAX_DEPTH, random_state=RANDOM_STATE)
    rf.fit(X_train, y_train)
    
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]

    print("=== Baseline Model Performance ===")
    print(classification_report(y_test, y_pred, digits=4))
    print(f"ROC AUC Score: {roc_auc_score(y_test, y_prob):.4f}\n" + "-" * 50)

    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]

    print("=== Feature Importance Ranking ===")
    for rank, idx in enumerate(indices):
        print(f"{rank + 1:02d}. {feature_names[idx]:<30} : {importances[idx]:.4f}")
    print("-" * 50)

    print("Running t-SNE Dim-Reduction...")
    X_scaled = StandardScaler().fit_transform(X)
    tsne = TSNE(n_components=2, perplexity=TSNE_PERPLEXITY, max_iter=TSNE_MAX_ITER, random_state=RANDOM_STATE)
    X_tsne = tsne.fit_transform(X_scaled)

    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=X_tsne[:, 0], y=X_tsne[:, 1],
        hue=y,
        palette={0: "#2ecc71", 1: "#e74c3c"},
        alpha=0.7,
        edgecolor=None
    )
    plt.title("Wallet Features t-SNE Visualization", fontsize=14)
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.legend(title="Label", labels=["Normal (0)", "Laundering (1)"])
    plt.grid(True, linestyle="--", alpha=0.5)

    os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
    output_fig = os.path.join(OUTPUT_IMG_DIR, OUTPUT_FIG_NAME)
    plt.savefig(output_fig, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Success: Cluster plot saved to -> {output_fig}")

if __name__ == "__main__":
    main()