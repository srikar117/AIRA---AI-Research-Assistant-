from pdf_processor import extract_text_from_pdf

def chunk_text(text, chunk_size = 1000, overlap = 200):
    chunks = []
    start = 0

    #fixed-size chunking
    #manual chunking to understand logic, see pattern
    # start = 0
    # end = start + chunk_size
    # first_chunk = text[start:end]
    # chunks.append(first_chunk)

    # start = start + (chunk_size - overlap)
    # end = start + chunk_size
    # second_chunk = text[start:end]
    # chunks.append(second_chunk)

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = start + (chunk_size - overlap)

    return chunks

#testing
if __name__ == "__main__":

    print("\n" + "="*50)
    print("TEST : Chunking a real PDF")
    print("="*50)
    
    pdf_path = input("Enter PDF path : ")
    if pdf_path:
        pdf_text = extract_text_from_pdf(pdf_path)
        if pdf_text:
            chunks = chunk_text(pdf_text, chunk_size=500, overlap=100)
            print(f"\nPDF was split into {len(chunks)} chunks")
            print(f"\nFirst chunk preview (200 chars):")
            print(chunks[0][:200])
