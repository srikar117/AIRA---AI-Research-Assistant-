import fitz
import os

def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        print(f"This document has {doc.page_count} pages")
        print(f"Here is the table of contents : {doc.get_toc()}")

        text = ''
        for page in doc:
            text = text + page.get_text()
        doc.close()
        return text
    
    except FileNotFoundError:
        print(f"Error : The file {pdf_path} was not found.")
        return None
    
    except Exception as e:
        print(f"Error : {e}")
        return None
    
#testing the function
if __name__ == "__main__":
    pdf_path = input("Please enter the path to your PDF file: ")
    extracted_text = extract_text_from_pdf(pdf_path)
    
    if extracted_text:
        print(f"\nTotal characters: {len(extracted_text)}")
        print(f"\nFirst 500 characters:\n{extracted_text[:500]}")