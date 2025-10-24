import os
from pdfminer.high_level import extract_text
from pathlib import Path
import re

doc_path = Path("/home/andreas/Documents/Python/local_llm/documents")


class DocumentProcessor:
    def __init__(self):
        pass

    def load_and_textualize(self, path):
        file = Path(path)
        if file.is_file():
            try:
                file_type = file.suffix.lower()
                text = extract_text(file)
                file_name = file.name.replace(" ", "_")
                file_name = file_name.replace(file_type, "")

                cleaned_text = self.clean_text(text)

                with open(f"{doc_path}/{file_name}.txt", "w") as f:
                    f.write(cleaned_text)
                print(f"{doc_path}{file_name}.txt created")

                return True

            except:
                print("!")
                return False

    def clean_text(self, text):
        text = re.sub(r"\s+", " ", text)  # collapse whitespace
        text = re.sub(r"-\s+", "", text)  # fix hyphenated line breaks
        text = re.sub(r"\f", "", text)  # remove form-feed chars

        return text.strip()


doc = DocumentProcessor()
doc.load_and_textualize(
    "/home/andreas/Documents/Python/local_llm/documents/Attention is all you need.pdf"
)
