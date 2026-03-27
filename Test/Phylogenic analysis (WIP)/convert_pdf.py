import fitz
import sys

def convert_pdf_to_md(pdf_path, md_path):
    doc = fitz.open(pdf_path)
    with open(md_path, 'w', encoding='utf-8') as f:
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text("text")
            f.write(f"## Page {page_num + 1}\n\n")
            f.write(text)
            f.write("\n\n---\n\n")
    doc.close()

if __name__ == "__main__":
    pdf_file = "Estimating Phylogenetic Multilevel Models with brms.pdf"
    md_file = "Estimating Phylogenetic Multilevel Models with brms.md"
    convert_pdf_to_md(pdf_file, md_file)
    print(f"Converted {pdf_file} to {md_file}")
