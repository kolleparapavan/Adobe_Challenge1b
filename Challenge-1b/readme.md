# 🧠 Intelligent Document Processing Suite

This repository contains two powerful tools for analyzing PDF documents:

1. *PDF Outline Extractor* – Automatically extracts the *title* and *headings (H1-H3)* from PDFs.
2. *Intelligent Document Analyst System* – Prioritizes and summarizes document sections based on a given *persona* and *job-to-be-done*.

---
---

## 🧩 1. PDF Outline Extractor (Headings & Title)

### 🔍 What It Does

- Scans all PDF files in input/
- Extracts:
  - *Document title* from large, centered text on the first page
  - *Headings* (H1, H2, H3) based on font size, boldness, and layout
- Outputs: a flat list of heading entries and the title into output/filename.json

### ⚙ How It Works

- *Font Style Analysis*: Determines body vs. heading fonts
- *Metadata Filtering*: Removes headers/footers (e.g., page numbers)
- *Heading Detection*: Classifies lines using:
  - Typographic style
  - Patterns like 1., 1.1, Appendix A
- *Title Detection*: Uses large, top-center text blocks
- *Multi-page Processing*: Supports full PDFs with repeating metadata

### 🐳 Docker Execution

```bash
# Build the Docker image
docker build -t pdf-extractor ./outline_extractor

# Run the container
docker run --rm -v "$(pwd)/outline_extractor/input:/app/input" \
                 -v "$(pwd)/outline_extractor/output:/app/output" pdf-extractor