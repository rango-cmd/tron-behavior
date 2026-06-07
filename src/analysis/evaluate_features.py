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

def main():
    if not os.path.exists(MATRIX_PATH):
        print(f"Error: {MATRIX_PATH} not found.")
        return

    df = pd.read_csv(MATRIX_PATH)
    X = df.drop(columns=["address", "seed_label"])
    y = df["seed_label"]
    feature_names = X.columns.tolist()

    print(f"Dataset Loaded: {X.shape[0]} samples, {X.shape[1]} features.")
    print(f"Class Distribution:\n{y.value_counts()}")
    print("-" * 50)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=8)
    rf.fit(X_train, y_train)
    
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]

    print("=== Baseline Model Performance ===")
    print(classification_report(y_test, y_pred, digits=4))
    print(f"ROC AUC Score: {roc_auc_score(y_test, y_prob):.4f}")
    print("-" * 50)

    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]

    print("=== Feature Importance Ranking ===")
    for rank, idx in enumerate(indices):
        print(f"{rank + 1:02d}. {feature_names[idx]:<30} : {importances[idx]:.4f}")
    print("-" * 50)

    print("Running t-SNE (May take a few seconds)...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 💡 修正：將 n_iter 修改為相容新版 scikit-learn 的 max_iter
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
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

    output_fig = os.path.join(OUTPUT_IMG_DIR, "features_cluster.png")
    plt.savefig(output_fig, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Success: Cluster plot saved to -> {output_fig}")

if __name__ == "__main__":
    main()