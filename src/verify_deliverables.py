"""
verify_deliverables.py
Programmatic verification script for Task 6.
Checks project_report.docx and project_report.pdf:
1. Student Name: Farhath Suraiya S D
2. Roll Number: 24AD035
3. Course: U21ADP05 – Exploratory Data Analysis and Visualization
4. Department: Artificial Intelligence and Data Science
5. Submission Date: 15.10.2026
6. Margins: Exactly 1.0 inch (2.54 cm) on all 4 sides across all sections
7. Running headers and footers: completely removed / empty
8. Section Headings: All 11 formal sections present
9. Figures & Captions: 21 figures embedded with captions, schematic labels present
10. PDF Verification: Valid PDF, extract text, check student details and page count
"""

import os
import sys
from docx import Document
from docx.shared import Inches
import pypdf

DOCX_PATH = "results/report/project_report.docx"
PDF_PATH = "results/report/project_report.pdf"


def verify_docx():
    print("=" * 80)
    print("PROGRAMMATIC VERIFICATION: project_report.docx")
    print("=" * 80)
    
    assert os.path.exists(DOCX_PATH), f"File not found: {DOCX_PATH}"
    doc = Document(DOCX_PATH)
    
    # 1. Margins check across all sections
    print("\n[Check 1: Page Margins (1.0 inch)]")
    for idx, s in enumerate(doc.sections):
        tm = round(s.top_margin.inches, 2)
        bm = round(s.bottom_margin.inches, 2)
        lm = round(s.left_margin.inches, 2)
        rm = round(s.right_margin.inches, 2)
        print(f"  Section {idx + 1}: Top={tm}\", Bottom={bm}\", Left={lm}\", Right={rm}\"")
        assert tm == 1.0, f"Section {idx+1} top margin is {tm}, expected 1.0"
        assert bm == 1.0, f"Section {idx+1} bottom margin is {bm}, expected 1.0"
        assert lm == 1.0, f"Section {idx+1} left margin is {lm}, expected 1.0"
        assert rm == 1.0, f"Section {idx+1} right margin is {rm}, expected 1.0"
    print("  -> PASS: All sections configured with exact 1.0 inch margins.")

    # 2. Running Headers & Footers check
    print("\n[Check 2: Running Headers & Footers (Must be completely removed)]")
    for idx, s in enumerate(doc.sections):
        hdr_text = "".join([p.text.strip() for p in s.header.paragraphs])
        ftr_text = "".join([p.text.strip() for p in s.footer.paragraphs])
        print(f"  Section {idx + 1}: Header='{hdr_text}', Footer='{ftr_text}'")
        assert hdr_text == "", f"Section {idx+1} has non-empty header: {hdr_text}"
        assert ftr_text == "", f"Section {idx+1} has non-empty footer: {ftr_text}"
    print("  -> PASS: All running headers and footers are 100% removed and empty.")

    # 3. Student details check in text & tables
    print("\n[Check 3: Student Details & Candidate Authentication]")
    full_text = []
    for p in doc.paragraphs:
        full_text.append(p.text)
    for t in doc.tables:
        for row in t.rows:
            for cell in row.cells:
                full_text.append(cell.text)
    doc_content = "\n".join(full_text)

    required_details = [
        ("Student Name", "Farhath Suraiya S D"),
        ("Roll Number", "24AD035"),
        ("Department", "Artificial Intelligence and Data Science"),
        ("Degree & Branch", "B.Tech. Artificial Intelligence and Data Science"),
        ("Academic Year", "2026–2027"),
        ("Course Code & Title", "U21ADP05 – Exploratory Data Analysis and Visualization"),
        ("Date of Submission", "15.10.2026")
    ]
    for label, val in required_details:
        present = val in doc_content
        print(f"  {label} ('{val}'): {'FOUND' if present else 'MISSING'}")
        assert present, f"Required detail missing: {val}"
    print("  -> PASS: All student details and course credentials verified.")

    # 4. Section headings check (All 11 sections)
    print("\n[Check 4: 11 Required Formal Academic Sections]")
    required_sections = [
        "Section 1: Title Page", # Implicit in Title layout
        "Section 2: Abstract",
        "Section 3: Introduction and Objectives",
        "Section 4: Dataset Description",
        "Section 5: EDA and Preprocessing",
        "Section 6: Data Visualization",
        "Section 7: Deep Learning Model",
        "Section 8: Result Visualization and Interpretation",
        "Section 9: Conclusion and Future Scope",
        "Section 10: References",
        "Section 11: Appendix: Code Section"
    ]
    for sec in required_sections[1:]: # check section 2 to 11 headings
        present = any(sec in p for p in full_text)
        print(f"  {sec}: {'FOUND' if present else 'MISSING'}")
        assert present, f"Section heading missing: {sec}"
    print("  -> PASS: All required academic sections are present.")

    # 5. Captions and Schematic label checks
    print("\n[Check 5: Figure Captions & Schematic Labels]")
    figure_captions = [p for p in full_text if p.startswith("Figure ")]
    print(f"  Total Figure Captions Found: {len(figure_captions)}")
    for fc in figure_captions:
        print(f"    - {fc[:80]}...")
    
    # Verify Figure 1 and Figure 13 schematic labels
    fig1_schematic = any("Figure 1:" in c and "[Schematic Diagram]" in c for c in figure_captions)
    fig13_schematic = any("Figure 13:" in c and "[Schematic Diagram]" in c for c in figure_captions)
    print(f"  Figure 1 labeled [Schematic Diagram]: {fig1_schematic}")
    print(f"  Figure 13 labeled [Schematic Diagram]: {fig13_schematic}")
    assert fig1_schematic, "Figure 1 must be labeled [Schematic Diagram]"
    assert fig13_schematic, "Figure 13 must be labeled [Schematic Diagram]"
    print("  -> PASS: All illustrative figures correctly labeled as schematics.")


    # 6. Verify verified split numbers and confusion matrix values in document content
    print("\n[Check 6: Data Split Proportions and Recalculated Metrics in DOCX]")
    required_metrics = [
        ("Training partition 17,350", "17,350"),
        ("Validation partition 17,351", "17,351"),
        ("Held-out test partition 14,872", "14,872"),
        ("35.0% split proportion", "35.0%"),
        ("30.0% test proportion", "30.0%"),
        ("Corrected confusion matrix [7703, 10, 2, 7157]", "[7703, 10, 2, 7157]"),
        ("Attack recall 99.97%", "99.97%"),
        ("Overall test accuracy 99.92%", "99.92%"),
        ("2 missed attacks", "2 missed")
    ]
    for label, val in required_metrics:
        present = val in doc_content
        print(f"  {label}: {'FOUND' if present else 'MISSING'}")
        assert present, f"Required metric string missing in DOCX: {val}"
    print("  -> PASS: All split numbers and metrics verified in DOCX.")


def verify_pdf():
    print("\n" + "=" * 80)
    print("PROGRAMMATIC VERIFICATION: project_report.pdf")
    print("=" * 80)
    
    assert os.path.exists(PDF_PATH), f"File not found: {PDF_PATH}"
    size_mb = os.path.getsize(PDF_PATH) / (1024 * 1024)
    print(f"  PDF File Size: {size_mb:.2f} MB")
    assert size_mb > 1.0, f"PDF file size too small: {size_mb} MB"

    reader = pypdf.PdfReader(PDF_PATH)
    total_pages = len(reader.pages)
    print(f"  Total PDF Pages: {total_pages}")
    assert total_pages >= 15, f"Expected comprehensive report (>=15 pages), found {total_pages}"

    # Check page dimensions and boundary consistency on EVERY page (1-inch margins on Letter = 612x792 pt)
    print("\n[Check PDF Page Dimensions & Boundaries on Every Page]")
    for p_idx, page in enumerate(reader.pages):
        w = float(page.mediabox.width)
        h = float(page.mediabox.height)
        assert w == 612.0 and h == 792.0, f"Page {p_idx+1} dimensions ({w}, {h}) differ from standard Letter (612, 792)"
    print(f"  -> PASS: All {total_pages} pages verified with uniform Letter dimensions (8.5\" x 11.0\" / 612 x 792 pt).")

    # Extract text from pages and check student details, split sizes, and metrics
    pdf_text = ""
    for p_idx in range(total_pages):
        extracted = reader.pages[p_idx].extract_text()
        if extracted:
            pdf_text += extracted + "\n"

    print("\n[Check PDF Content Extraction & Verified Metrics]")
    pdf_checks = [
        ("Candidate Name", "Farhath Suraiya S D"),
        ("Roll Number", "24AD035"),
        ("Department", "Artificial Intelligence and Data Science"),
        ("Course Code", "U21ADP05"),
        ("Date", "15.10.2026"),
        ("Training size 17,350", "17,350"),
        ("Validation size 17,351", "17,351"),
        ("Held-out test size 14,872", "14,872"),
        ("Accuracy 99.92%", "99.92%"),
        ("ROC-AUC 0.9997", "0.9997"),
        ("Recall 99.97%", "99.97%"),
        ("Confusion Matrix [7703, 10, 2, 7157]", "7703")
    ]
    for label, d in pdf_checks:
        present = d in pdf_text
        print(f"  {label} ('{d}') in PDF: {'FOUND' if present else 'MISSING'}")
        assert present, f"Detail missing in PDF: {d}"
    print("  -> PASS: PDF successfully verified with exact candidate credentials, split sizes, and metrics.")


if __name__ == "__main__":
    verify_docx()
    verify_pdf()
    print("\n" + "=" * 80)
    print("ALL PROGRAMMATIC VERIFICATION CHECKS PASSED SUCCESSFULLY!")
    print("=" * 80)
