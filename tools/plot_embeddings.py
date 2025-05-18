import argparse, textwrap
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from chromadb import PersistentClient

def main(limit: int = 500):
    col = PersistentClient(path=".chroma").get_collection("doc_chunks")
    data = col.peek(limit=limit)
    X = data["embeddings"]               
    pages = [m["page"] for m in data["metadatas"]]
    snippets = [textwrap.shorten(t, 60) for t in data["documents"]]

    pca = PCA(n_components=2).fit_transform(X)

    plt.figure(figsize=(8, 6))
    sc = plt.scatter(pca[:, 0], pca[:, 1], c=pages, s=15, alpha=0.7, cmap="viridis")
    plt.colorbar(sc, label="Page #")
    for (x, y, s) in zip(pca[:, 0], pca[:, 1], snippets[:50]): 
        plt.text(x, y, s, fontsize=6, alpha=0.7)
    plt.title("APD Manual — Embedding PCA (first %d chunks)" % limit)
    plt.tight_layout()
    plt.savefig("embeddings.png", dpi=300)
    print("Saved embeddings.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=500)
    main(**vars(parser.parse_args()))
