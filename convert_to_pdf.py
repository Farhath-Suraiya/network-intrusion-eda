"""
convert_to_pdf.py
Converts project_report.docx to project_report.pdf with DisplayAlerts disabled.
"""

import os
import sys

def convert_docx_to_pdf(docx_path, pdf_path):
    abs_docx = os.path.abspath(docx_path)
    abs_pdf = os.path.abspath(pdf_path)

    print(f"Converting:\n  Input:  {abs_docx}\n  Output: {abs_pdf}")
    
    import win32com.client
    word = win32com.client.Dispatch("Word.Application")
    word.Visible = False
    word.DisplayAlerts = 0  # wdAlertsNone = 0

    try:
        # Open in read-only mode to prevent any locking issues
        doc = word.Documents.Open(abs_docx, ReadOnly=True, ConfirmConversions=False)
        # wdExportFormatPDF = 17
        doc.ExportAsFixedFormat(
            OutputFileName=abs_pdf,
            ExportFormat=17,
            OpenAfterExport=False,
            OptimizeFor=0, # wdExportOptimizeForPrint
            CreateBookmarks=1 # wdExportCreateHeadingBookmarks
        )
        doc.Close(SaveChanges=False)
        print("Export completed successfully!")
    finally:
        word.Quit()

if __name__ == "__main__":
    docx = "results/report/project_report.docx"
    pdf = "results/report/project_report.pdf"
    convert_docx_to_pdf(docx, pdf)
