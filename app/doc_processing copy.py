import os
import re
from pathlib import Path
import fitz


from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import EasyOcrOptions, PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

artifacts_path = "/local/path/to/models"

pipeline_options = PdfPipelineOptions(artifacts_path=artifacts_path)
doc_converter = DocumentConverter(
    format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
)


class DocumentProcessor:
    def __init__(self):
        self.doc_path = Path("/home/andreas/Documents/Python/local_llm/documents")
        self.archive = self.doc_path / "processed_archive"
        self.output_dir = self.doc_path / "processed"
        self.output_dir.mkdir(exist_ok=True)

    def pipeline(self, path):
        src_file = Path(path)
        file_name = src_file.stem.replace(" ", "_")

        raw = self.load_and_textualize(src_file)
        print(f"\nProcessing {file_name} | Raw length: {len(raw)}")

        cleaned = self.clean_text(raw)
        cleaned = self.remove_headers_and_footers(cleaned)
        cleaned = self.remove_references(cleaned)
        cleaned = self.remove_tables(cleaned)

        out_path = self.output_dir / f"{file_name}.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(cleaned)

        print(f"Saved cleaned text to {out_path}")

    def load_and_textualize(self, src_file: Path) -> str:
        if not src_file.is_file():
            raise FileNotFoundError(f"File not found: {src_file}")

        raw_text = ""
        with fitz.open(src_file) as doc:
            for page in doc:
                page_text = page.get_text("text")
                raw_text += page_text + "\n\n"
        return raw_text

    def clean_text(self, text: str) -> str:
        text = text.replace("\f", " ")  # remove form-feed
        text = re.sub(r"-\s*\n", "", text)  # fix hyphenated line breaks
        text = re.sub(r"\n{3,}", "\n\n", text)  # limit multiple blank lines
        text = re.sub(r"[ \t]+", " ", text)  # collapse excessive spaces
        text = re.sub(r"\[\s?\d+(?:[,\-–]\s?\d+)*\s?\]", "", text)  # remove [1,2] etc
        return text.strip()

    def remove_references(self, text: str) -> str:
        pattern = re.compile(
            r"\n\s*(References|Bibliography|Literature Cited)\s*[:\-]?\s*\n",
            re.IGNORECASE,
        )
        match = pattern.search(text)
        if match:
            return text[: match.start()].strip()
        return text

    def remove_tables(self, text: str) -> str:
        lines = text.splitlines()
        clean_lines = []
        for line in lines:
            if re.match(r"^\s*[\d\.\|\t\s]+\s*$", line):
                continue
            if len(re.findall(r"\s{3,}", line)) > 4:  # too many space columns
                continue
            clean_lines.append(line)
        return "\n".join(clean_lines)

    def remove_headers_and_footers(self, text: str) -> str:
        lines = text.splitlines()
        cleaned = []
        for line in lines:
            if re.match(r"^\s*\d+\s*$", line):  # page number
                continue
            if re.search(r"(Page\s*\d+|©|\bCopyright\b)", line, re.IGNORECASE):
                continue
            cleaned.append(line)
        return "\n".join(cleaned).strip()


if __name__ == "__main__":
    doc = DocumentProcessor()
    base_path = Path("/home/andreas/Documents/Python/local_llm/documents")

    for filename in os.listdir(base_path):
        file_path = base_path / filename
        if file_path.is_file() and file_path.suffix.lower() == ".pdf":
            doc.pipeline(file_path)
