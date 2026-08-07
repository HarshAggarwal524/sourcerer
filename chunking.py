from parse import extract_text

def chunk_text(text, chunk_size=300):
    """
    Splits text into a list of chunks, each with roughly chunk_size words.
    """
    if not text:
        return []

    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk_words = words[i:i + chunk_size]
        chunk = " ".join(chunk_words)
        chunks.append(chunk)

    return chunks


if __name__ == "__main__":
    test_path = "/Users/harshaggarwal/Projects_4/sourcerer/sampel.pdf"  # replace with your real path
    text = extract_text(test_path)

    if text:
        chunks = chunk_text(text, chunk_size=300)
        print(f"Total chunks: {len(chunks)}")
        print("\n--- Chunk 1 preview ---")
        print(chunks[0][:200])
        print("\n--- Chunk 2 preview ---")
        if len(chunks) > 1:
            print(chunks[1][:200])
    else:
        print("No text extracted, nothing to chunk.")