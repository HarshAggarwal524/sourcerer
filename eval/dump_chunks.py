import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import load_or_build

pdf_path = "/Users/harshaggarwal/Projects_4/sourcerer/data/testpdf.pdf"  # your actual path
chunks, embeddings = load_or_build(pdf_path)

for i, chunk in enumerate(chunks):
    print(f"--- Chunk {i} ---")
    print(chunk)
    print()