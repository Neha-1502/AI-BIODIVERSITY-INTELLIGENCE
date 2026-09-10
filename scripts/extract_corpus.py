"""
extract_corpus.py

Converts real downloaded PDF reports (FAO / IPCC / IPBES / etc.) into
corpus-ready .txt files for the RAG pipeline in knowledge_base/corpus/.

USAGE:
  1. Download report PDFs into a folder, e.g. raw_pdfs/
  2. Edit the SOURCES list below to describe each file (title, org, topics, url)
  3. Run:
       pip install pdfplumber --break-system-packages
       python scripts/extract_corpus.py

This will:
  - Extract raw text from each PDF
  - Strip common boilerplate (page numbers, running headers/footers)
  - Split into paragraph-level sections and re-chunk to a reasonable size
  - Write one .txt file per PDF into knowledge_base/corpus/, with the
    SOURCE: / TOPIC: header format that build_vector_db.py expects

IMPORTANT -- copyright / attribution:
  - Only extract from reports you have legitimately downloaded and are
    permitted to use for this purpose (most FAO/IPCC/IPBES reports are
    published for open policy/research use, but check each report's own
    license page before wide redistribution).
  - Keep the full original PDF alongside the extracted text and always
    preserve the source URL/citation in the header -- you'll need it when
    the LLM cites sources in its recommendations.
  - This pipeline is for internal retrieval (your chatbot looking up
    evidence), not for republishing the reports themselves.
"""

import os
import re
import pdfplumber

RAW_PDF_DIR = os.path.join(os.path.dirname(__file__), "..", "raw_pdfs")
CORPUS_DIR = os.path.join(os.path.dirname(__file__), "..", "knowledge_base", "corpus")

# ---------------------------------------------------------------------------
# EDIT THIS: describe each PDF you've downloaded into raw_pdfs/
# filename must match exactly what's in raw_pdfs/
# ---------------------------------------------------------------------------
SOURCES = [
    {
        "filename": "Soil Organic Carbon - the hidden potential.pdf",
        "source_label": "FAO - Soil Organic Carbon: the hidden potential",
        "topics": "soil_organic_carbon, soil_mapping, land_degradation",
        "url": "https://www.fao.org/global-soil-partnership/resources/",
    },
    {
        "filename": "IPCC_AR6_WGII_FullReport.pdf",
        "source_label": "IPCC AR6 WG2 Chapter 5 - Food, Fibre and Other Ecosystem Products",
        "topics": "climate, land_use, food_systems, agroforestry",
        "url": "https://www.ipcc.ch/report/ar6/wg2/",
        "page_range": (713, 999),
    },
    {
        "filename": "ipbes_global_assessment_report_summary_for_policymakers_en.pdf",
        "source_label": "IPBES Global Assessment Report - Summary for Policymakers",
        "topics": "biodiversity, habitat_fragmentation, species_richness, human_impact",
        "url": "https://www.ipbes.net/global-assessment",
    },
]

MIN_PARAGRAPH_WORDS = 40   # skip tiny fragments (headers, page numbers, table labels)
CHUNK_TARGET_WORDS = 250   # roughly matches build_vector_db.py's chunk size


def clean_text(raw: str) -> str:
    """Removes common PDF extraction noise: repeated whitespace, page numbers,
    isolated short lines that are likely headers/footers."""
    lines = raw.split("\n")
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if re.fullmatch(r"\d{1,4}", stripped):          # bare page numbers
            continue
        if len(stripped.split()) <= 3 and stripped.isupper():  # short ALL-CAPS headers
            continue
        cleaned_lines.append(stripped)
    text = " ".join(cleaned_lines)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def paragraph_chunks(text: str, target_words: int = CHUNK_TARGET_WORDS):
    """Groups cleaned text into ~target_words chunks, breaking on sentence
    boundaries so chunks stay coherent."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    current = []
    current_len = 0
    for sentence in sentences:
        words = sentence.split()
        if current_len + len(words) > target_words and current:
            chunks.append(" ".join(current))
            current = []
            current_len = 0
        current.append(sentence)
        current_len += len(words)
    if current:
        chunks.append(" ".join(current))
    return [c for c in chunks if len(c.split()) >= MIN_PARAGRAPH_WORDS]

def is_likely_garbled(text: str, threshold: float = 0.03) -> bool:
    """Flags chunks that are likely extraction artifacts (rotated/mirrored figure
    text, reversed strings from complex PDF layouts) by checking whether common
    English function words appear at a normal rate. Real prose typically has
    5-15% of words as these; garbled/reversed text has almost none."""
    common_words = {
        "the", "and", "of", "in", "to", "a", "is", "that", "for", "are",
        "with", "as", "on", "by", "this", "it", "be", "was", "were",
        "has", "have", "which", "from", "their",
    }
    words = re.findall(r"[a-zA-Z']+", text.lower())
    if not words:
        return True
    common_count = sum(1 for w in words if w in common_words)
    return (common_count / len(words)) < threshold

def extract_pdf(filepath: str, page_range: tuple | None = None) -> str:
    """page_range: optional (start_page, end_page), 1-indexed, inclusive.
    Use this for large reports where you only want a specific chapter --
    e.g. (713, 999) for IPCC AR6 WGII's Chapter 5 within the full report PDF."""
    all_text = []
    with pdfplumber.open(filepath) as pdf:
        pages = pdf.pages
        if page_range:
            start, end = page_range
            pages = pages[start - 1:end]
        for page in pages:
            page_text = page.extract_text() or ""
            all_text.append(page_text)
    return "\n".join(all_text)


def process_source(source: dict):
    pdf_path = os.path.join(RAW_PDF_DIR, source["filename"])
    if not os.path.exists(pdf_path):
        print(f"[skip] {source['filename']} not found in {RAW_PDF_DIR}/")
        return

    print(f"[processing] {source['filename']} ...")
    raw_text = extract_pdf(pdf_path, page_range=source.get("page_range"))
    cleaned = clean_text(raw_text)
    chunks = paragraph_chunks(cleaned)
    chunks_before = len(chunks)
    chunks = [c for c in chunks if not is_likely_garbled(c)]
    if chunks_before != len(chunks):
        print(f"  (filtered out {chunks_before - len(chunks)} likely-garbled chunks)")

    out_name = os.path.splitext(source["filename"])[0] + "_extracted.txt"
    out_path = os.path.join(CORPUS_DIR, out_name)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"SOURCE: {source['source_label']} ({source['url']})\n")
        f.write(f"TOPIC: {source['topics']}\n\n")
        for chunk in chunks:
            f.write(chunk + "\n\n")

    print(f"  -> wrote {len(chunks)} paragraph-chunks to {out_path}")


if __name__ == "__main__":
    os.makedirs(CORPUS_DIR, exist_ok=True)
    if not os.path.isdir(RAW_PDF_DIR):
        print(f"Create {RAW_PDF_DIR}/ and put your downloaded report PDFs there first.")
    else:
        for source in SOURCES:
            process_source(source)
        print("\nDone. Now run: python scripts/build_vector_db.py to re-index.")
