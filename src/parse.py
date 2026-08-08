from pypdf import PdfReader

def extract_text(file_path):
    """
    Extracts all text from a PDF and returns it as a single string.
    Returns None if the file can't be read.
    """
    try:
        reader = PdfReader(file_path)
    except Exception as e:
        print(f"[parse.py] Could not open PDF: {e}")
        return None

    full_text = []
    for page_num, page in enumerate(reader.pages):
        try:
            text = page.extract_text()
            if text:
                full_text.append(text)
            else:
                print(f"[parse.py] No extractable text on page {page_num + 1} (possibly scanned/image-only)")
        except Exception as e:
            print(f"[parse.py] Failed to extract text from page {page_num + 1}: {e}")

    return "\n".join(full_text)


if __name__ == "__main__":
    test_path = "/Users/harshaggarwal/Projects_4/sourcerer/sampel.pdf"  # replace with a real path to test
    result = extract_text(test_path)
    if result:
        print(f"Extracted {len(result)} characters.")
        print(result[:500])
        
        
    else:
        print("No text extracted.")
