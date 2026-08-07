import pickle
from parse import extract_text
from chunking import chunk_text
from embed import embed_chunks

def save_data(chunks, embeddings, path="store.pkl"):
    """
    Saves chunks and their embeddings to disk as a pickle file.
    """
    bundle = {"chunks": chunks, "embeddings": embeddings}
    with open(path, "wb") as f:
        pickle.dump(bundle, f)

def load_data(path="store.pkl"):
    """
    Loads chunks and embeddings from disk.
    Returns None if the file doesn't exist yet.
    """
    try:
        with open(path, "rb") as f:
            bundle = pickle.load(f)
        return bundle
    except FileNotFoundError:
        return None


if __name__ == "__main__":
    test_path = "/Users/harshaggarwal/Projects_4/sourcerer/sampel.pdf"  # replace with your real path
    text = extract_text(test_path)

    if text:
        chunks = chunk_text(text, chunk_size=300)
        embeddings = embed_chunks(chunks)

        save_data(chunks, embeddings)
        print("Saved to store.pkl")

        loaded = load_data()
        if loaded:
            print(f"Loaded {len(loaded['chunks'])} chunks")
            print(f"Loaded embeddings shape: {loaded['embeddings'].shape}")

            # sanity checks
            assert len(loaded["chunks"]) == len(chunks)
            assert loaded["embeddings"].shape == embeddings.shape
            print("Sanity checks passed.")
        else:
            print("Load failed — file not found.")
    else:
        print("No text extracted, nothing to store.")