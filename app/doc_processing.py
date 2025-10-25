import os
from pdfminer.high_level import extract_text
from pathlib import Path
import re
import fitz

doc_path = Path("/home/andreas/Documents/Python/local_llm/documents")


class DocumentProcessor:
    def __init__(self):
        self.doc_path = Path("/home/andreas/Documents/Python/local_llm/documents")
        self.archive = Path(
            "/home/andreas/Documents/Python/local_llm/documents/processed_archive"
        )

    def pipeline(self, path):
        src_file = Path(path)
        file_name = src_file.name.replace(" ", "_")
        file_type = src_file.suffix.lower()
        file_name = file_name.replace(file_type, "")

        raw = self.load_and_textualize(src_file)
        print("Raw length:", len(raw))
        print(raw[:1000])

        cleaned_text = self.clean_text(raw)
        # print(f"\n{cleaned_text}")
        cleaned_text = self.remove_headers_and_footers(cleaned_text)
        # print(f"\n{cleaned_text}")

        with open(f"{self.doc_path}/processed/{file_name}.txt", "w") as f:
            f.write(cleaned_text)

    def load_and_textualize(self, src_file):
        if src_file.is_file():
            raw_text = ""
            with fitz.open(src_file) as doc:
                for page in doc:
                    raw_text += page.get_text("text") + "\n"

            return raw_text
        else:
            raise FileNotFoundError(f"File not found: {src_file}")

    def clean_text(self, text):
        text = re.sub(r"\s+", " ", text)  # collapse whitespace
        text = re.sub(r"-\s+", "", text)  # fix hyphenated line breaks
        text = re.sub(r"\f", "", text)  # remove form-feed chars
        text = re.sub(r"\[\s?\d+([,\-–]\s?\d+)*\s?\]", "", text)
        text = re.sub(r"-\n", "", text)

        return text.strip()

    def remove_references(self, text):
        pattern = re.compile(
            r"\n?\s*(References|Bibliography|Literature Cited)[\s\:\-]*\n",
            re.IGNORECASE,
        )
        match = pattern.search(text)
        if match:
            return text[: match.start()]
        return text

    def remove_tables(self, text):
        lines = text.split("\n")
        clean_lines = []
        for line in lines:
            # Skip lines that look like tables
            if re.match(r"^\s*[\d\.\|\t\s]+\s*$", line):
                continue
            if len(re.findall(r"\s{2,}", line)) > 3:  # too many space columns
                continue
            clean_lines.append(line)
        return "\n".join(clean_lines)

    def remove_headers_and_footers(self, text):
        lines = text.split("\n")
        cleaned = []
        for line in lines:
            # skip page numbers or running headers
            if re.match(r"^\s*\d+\s*$", line):
                continue
            if "©" in line or "Page" in line:
                continue
            cleaned.append(line)
        return "\n".join(cleaned)


doc = DocumentProcessor()
doc.pipeline(
    "/home/andreas/Documents/Python/local_llm/documents/Attention is all you need.pdf"
)
