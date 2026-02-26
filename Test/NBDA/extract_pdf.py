import sys

pdf_path = r"c:\Users\Sosa\Documents\BI\NBDA orgiinal\Methods Ecol Evol - 2025 - Chimento - STbayes  An R package for creating  fitting and understanding Bayesian models of.pdf"
output_path = r"c:\Users\Sosa\Documents\BI\NBDA orgiinal\extracted_pdf.txt"

def extract_text():
    try:
        import PyPDF2
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)
        print("Successfully extracted using PyPDF2")
        return
    except ImportError:
        pass
    except Exception as e:
        print(f"PyPDF2 failed: {e}")

    try:
        import fitz  # PyMuPDF
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text() + "\n"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)
        print("Successfully extracted using PyMuPDF (fitz)")
        return
    except ImportError:
        print("Neither PyPDF2 nor PyMuPDF is installed.")
    except Exception as e:
        print(f"PyMuPDF failed: {e}")

extract_text()
