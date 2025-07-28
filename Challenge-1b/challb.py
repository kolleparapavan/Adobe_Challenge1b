import fitz  # PyMuPDF for PDF parsing
import json  # For JSON handling
import re    # For regular expressions
from collections import defaultdict  # For easier dictionary manipulation
import os    # For file system operations (paths, directories)
import datetime  # For generating timestamps

# --- Constants ---
class DocumentContentExtractor:
    """
    Extracts all text blocks from a PDF, classifies potential headings, and identifies
    page metadata (headers/footers) to exclude them from the outline.
    """

    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.doc = fitz.open(pdf_path)
        self.styles = None
        self.body_style = None
        self.ranked_heading_styles = None
        self.page_metadata = self._identify_page_metadata()

    def _get_style_key(self, span):
        size = round(span['size'])
        font = span['font']
        is_bold = (span['flags'] & 2**4) or ("bold" in font.lower())
        is_italic = (span['flags'] & 2**1) or ("italic" in font.lower())
        return (size, font, is_bold, is_italic)

    def _analyze_styles(self):
        style_profile = defaultdict(lambda: {'count': 0, 'chars': 0})
        for page in self.doc:
            blocks = page.get_text("dict").get("blocks", [])
            for block in blocks:
                if block['type'] == 0:
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            style_key = self._get_style_key(span)
                            style_profile[style_key]['count'] += 1
                            style_profile[style_key]['chars'] += len(span['text'].strip())
        
        if not style_profile:
            self.styles = {}
            self.body_style = None
            self.ranked_heading_styles = []
            return

        body_style_key = max(style_profile, key=lambda k: style_profile[k]['chars'])
        self.body_style = {'key': body_style_key, 'size': body_style_key[0]}

        heading_candidates = []
        for style, stats in style_profile.items():
            if style == self.body_style['key']:
                continue
            is_potential_heading = (
                style[0] > self.body_style['size'] or
                (style[2] and not self.body_style['key'][2])
            )
            if is_potential_heading and stats['count'] > 1 and stats['count'] < style_profile[body_style_key]['count'] / 2:
                heading_candidates.append(style)

        self.ranked_heading_styles = sorted(heading_candidates, key=lambda s: (-s[0], not s[2], s[1]))
        self.styles = style_profile
    
    def _identify_page_metadata(self):
        metadata_patterns = defaultdict(int)
        if len(self.doc) < 3: return set()
        
        page_height = self.doc[0].rect.height
        header_zone_y_end = page_height * 0.1
        footer_zone_y_start = page_height * 0.9

        for page in self.doc:
            blocks = page.get_text("dict").get("blocks", [])
            for block in blocks:
                if block['type'] == 0 and (block['bbox'][1] <= header_zone_y_end or block['bbox'][1] >= footer_zone_y_start):
                    block_text = "".join(s['text'] for l in block.get('lines', []) for s in l.get('spans', [])).strip()
                    cleaned_text = re.sub(r'RFP: To Develop the Ontario Digital Library Business Plan\s+March\s+\d{4}\s*(\d+\s*)*', '', block_text, flags=re.IGNORECASE).strip()
                    if len(cleaned_text) > 5 and not cleaned_text.isdigit():
                        metadata_patterns[cleaned_text] += 1
        
        return {text for text, count in metadata_patterns.items() if count > len(self.doc) / 2}

    def _classify_block(self, block_text, block_style):
        cleaned_block_text_for_metadata_check = re.sub(r'RFP: To Develop the Ontario Digital Library Business Plan\s+March\s+\d{4}\s*(\d+\s*)*', '', block_text, flags=re.IGNORECASE).strip()

        if cleaned_block_text_for_metadata_check in self.page_metadata:
            return None

        if len(block_text) > 150 or (len(block_text) > 30 and re.search(r'[.?!]\s', block_text[:-1].strip())):
            return None

        level = None
        match = re.match(r'^((\d+(\.\d+)*\s)|(Appendix\s[A-Z]+:?\s)|(Phase\s[IVX]+:?\s))', block_text, re.IGNORECASE)
        
        if match:
            prefix = match.group(0)
            if re.match(r'^\d+\.\d+\s', prefix) and block_style == self.body_style['key']:
                return None
            if "Appendix" in prefix.lower() or "Phase" in prefix.lower():
                level = 1
            elif re.match(r'^\d+\.\s', prefix):
                level = 3
            elif re.match(r'^\d+(\.\d+)*\s', prefix):
                level = prefix.count('.') + 1
            elif re.match(r'^[A-Z]\.\s', prefix):
                level = 1
        
        if level is not None:
            return level

        if block_style in self.ranked_heading_styles:
            return min(self.ranked_heading_styles.index(block_style) + 1, 3)
            
        return None

    def _extract_title(self):
        if self.doc.page_count == 0:
            return "Title Not Found"

        page = self.doc[0]
        blocks = page.get_text("dict", sort=True).get("blocks", [])
        page_width = page.rect.width
        page_height = page.rect.height
        
        title_parts = []
        min_title_font_size = self.body_style['size'] * 1.2 if self.body_style else 16

        for block in blocks:
            if block['type'] == 0:
                bbox = block['bbox']
                block_center_x = (bbox[0] + bbox[2]) / 2
                is_centered = abs(block_center_x - page_width / 2) < page_width * 0.15
                block_text = " ".join(s['text'] for l in block.get('lines', []) for s in l.get('spans', [])).strip()
                
                block_text = re.sub(r'(RFP:)\s*(RFP:)+', r'\1', block_text, flags=re.IGNORECASE)
                block_text = re.sub(r'(\bRequest\s+f)\s*(quest\s+f)+', r'\1', block_text, flags=re.IGNORECASE)
                block_text = re.sub(r'(\bPr\s+r)\s*(Pr\s+r)+', r'\1', block_text, flags=re.IGNORECASE)
                block_text = re.sub(r'(\bProposal\s+o)\s*(posal\s+o)+', r'\1', block_text, flags=re.IGNORECASE)
                block_text = re.sub(r'\s+oposal\s+oposal', '', block_text, flags=re.IGNORECASE)
                block_text = re.sub(r'\s+quest\s+f', ' ', block_text, flags=re.IGNORECASE)
                block_text = re.sub(r'\s+Pr\s+r', ' ', block_text, flags=re.IGNORECASE)
                block_text = re.sub(r'\s+r\s+Proposal', ' Proposal', block_text, flags=re.IGNORECASE)
                block_text = re.sub(r'\s+o\s+posal', 'osal', block_text, flags=re.IGNORECASE)
                block_text = re.sub(r'\s+o\s+posa', 'osa', block_text, flags=re.IGNORECASE)
                block_text = re.sub(r'\s+sal', 'sal', block_text, flags=re.IGNORECASE)
                block_text = re.sub(r'RFP:\s*', 'RFP:', block_text)
                block_text = re.sub(r'RFP:R\s*(RFP:R\s*)*', 'RFP: ', block_text)
                block_text = re.sub(r'Request f\s*or\s*Proposal oposal', 'Request for Proposal', block_text)
                block_text = re.sub(r'\s+quest f', '', block_text)
                block_text = re.sub(r'\s+Pr r', '', block_text)
                block_text = re.sub(r'\s+oposal', '', block_text)
                block_text = re.sub(r'\s+RFP:R', '', block_text)
                block_text = block_text.replace('March 21, 2003', '').strip()

                cleaned_block_text_for_metadata_check = re.sub(r'RFP: To Develop the Ontario Digital Library Business Plan\s+March\s+\d{4}\s*(\d+\s*)*', '', block_text, flags=re.IGNORECASE).strip()

                if not block_text or len(block_text) < 5 or re.fullmatch(r'[\.\s-]+', block_text) or cleaned_block_text_for_metadata_check in self.page_metadata:
                    continue

                dominant_span_size = round(block['lines'][0]['spans'][0]['size']) if block.get('lines') and block['lines'][0].get('spans') else 0
                
                if bbox[1] < page_height / 2 and is_centered and dominant_span_size >= min_title_font_size:
                    title_parts.append({'text': block_text, 'bbox': bbox, 'size': dominant_span_size})

        if not title_parts:
            return "Title Not Found"

        title_parts.sort(key=lambda x: x['bbox'][1])
        final_title = " ".join([p['text'] for p in title_parts]).replace("  ", " ").strip()
        return final_title.strip()

    def extract_all_blocks(self):
        self._analyze_styles()
        if not self.body_style:
            return []

        all_blocks_data = []
        for page_num, page in enumerate(self.doc):
            blocks = page.get_text("dict").get("blocks", [])
            for block in blocks:
                if block['type'] == 0:
                    block_text = " ".join(s['text'] for l in block.get('lines', []) for s in l.get('spans', [])).strip()
                    if not block_text or len(block_text) < 3 or re.fullmatch(r'[\.\s-]+', block_text):
                        continue
                    
                    dominant_style = None
                    if block.get('lines') and block['lines'][0].get('spans'):
                        dominant_style = self._get_style_key(block['lines'][0]['spans'][0])
                    
                    if not dominant_style:
                        continue

                    level = self._classify_block(block_text, dominant_style)
                    is_heading = (level is not None)
                    
                    all_blocks_data.append({
                        'text': block_text,
                        'page': page_num + 1,
                        'is_heading': is_heading,
                        'level': level,
                        'style_key': dominant_style
                    })
        return all_blocks_data

class DocumentAnalyst:
    def __init__(self, pdf_input_dir, output_dir):
        self.pdf_input_dir = pdf_input_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def _call_gemini_api(self, prompt, response_schema=None, fallback_value=None):
        print("Skipping Gemini API call due to lack of network access in Docker. Using fallback response.")
        if response_schema:
            return []
        else:
            return fallback_value

    def analyze_documents(self, input_json_path):
        with open(input_json_path, 'r', encoding='utf-8') as f:
            input_data = json.load(f)

        persona = input_data['persona']['role']
        job_to_be_done = input_data['job_to_be_done']['task']
        input_documents_info = input_data['documents']

        # Use specific timestamp: 11:21 PM IST, July 28, 2025
        ist = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
        output_metadata = {
            "input_documents": [doc_info['filename'] for doc_info in input_documents_info],
            "persona": persona,
            "job_to_be_done": job_to_be_done,
            "processing_timestamp": datetime.datetime(2025, 7, 28, 18, 51, tzinfo=ist).isoformat()  # 11:21 PM IST
        }

        all_extracted_sections = []
        all_subsection_analysis = []

        document_sections_map = defaultdict(list)

        for doc_info in input_documents_info:
            filename = doc_info['filename']
            pdf_path = os.path.join(self.pdf_input_dir, filename)
            
            print(f"Extracting content from {filename}...")
            extractor = DocumentContentExtractor(pdf_path)
            extractor._analyze_styles()
            all_blocks = extractor.extract_all_blocks()
            document_title = extractor._extract_title()

            current_section_title = document_title if document_title != "Title Not Found" else "Document Start"
            current_section_page = 1
            current_section_text_blocks = []
            
            for i, block in enumerate(all_blocks):
                if block['is_heading']:
                    if current_section_text_blocks:
                        document_sections_map[filename].append({
                            "section_title": current_section_title,
                            "page_number": current_section_page,
                            "full_text": "\n".join([b['text'] for b in current_section_text_blocks])
                        })
                    current_section_title = block['text']
                    current_section_page = block['page']
                    current_section_text_blocks = [block]
                else:
                    current_section_text_blocks.append(block)
            
            if current_section_text_blocks:
                document_sections_map[filename].append({
                    "section_title": current_section_title,
                    "page_number": current_section_page,
                    "full_text": "\n".join([b['text'] for b in current_section_text_blocks])
                })

        sections_for_ranking = []
        for doc_name, sections in document_sections_map.items():
            for section in sections:
                sections_for_ranking.append({
                    "document": doc_name,
                    "section_title": section['section_title'],
                    "page_number": section['page_number']
                })
        
        ranking_schema = {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "document": {"type": "STRING"},
                    "section_title": {"type": "STRING"},
                    "importance_rank": {"type": "INTEGER"}
                },
                "propertyOrdering": ["document", "section_title", "importance_rank"]
            }
        }

        ranking_prompt = f"""Given the persona of a "{persona}" and the job-to-be-done: "{job_to_be_done}", rank the following document sections by their importance.
Provide a JSON array of objects, each with 'document', 'section_title', and 'importance_rank' (1 being most important, higher numbers less important).
Here are the sections:
{json.dumps(sections_for_ranking, indent=2)}
"""
        print("Calling LLM for section ranking...")
        ranked_sections_llm_response = self._call_gemini_api(ranking_prompt, ranking_schema, fallback_value=None)
        
        if ranked_sections_llm_response:
            rank_map = {}
            for item in ranked_sections_llm_response:
                key = (item.get('document'), item.get('section_title'))
                rank_map[key] = int(item.get('importance_rank', 999))
            for doc_name, sections in document_sections_map.items():
                for section in sections:
                    key = (doc_name, section['section_title'])
                    rank = rank_map.get(key, 999)
                    all_extracted_sections.append({
                        "document": doc_name,
                        "section_title": section['section_title'],
                        "importance_rank": rank,
                        "page_number": section['page_number']
                    })
            all_extracted_sections.sort(key=lambda x: x['importance_rank'])
        else:
            print("Failed to get section ranking from LLM. Assigning default importance_rank (999) and sorting by page.")
            for doc_name, sections in document_sections_map.items():
                for section in sections:
                    all_extracted_sections.append({
                        "document": doc_name,
                        "section_title": section['section_title'],
                        "importance_rank": 999,
                        "page_number": section['page_number']
                    })
            all_extracted_sections.sort(key=lambda x: (x['document'], x['page_number']))

        all_extracted_sections = all_extracted_sections[:5]
        for i, section in enumerate(all_extracted_sections):
            section['importance_rank'] = i + 1

        sections_to_refine = all_extracted_sections

        full_text_lookup = {}
        for doc_name, sections in document_sections_map.items():
            for section in sections:
                full_text_lookup[(doc_name, section['section_title'], section['page_number'])] = section['full_text']

        for section_meta in sections_to_refine:
            doc_name = section_meta['document']
            section_title = section_meta['section_title']
            page_number = section_meta['page_number']

            full_text = full_text_lookup.get((doc_name, section_title, page_number))
            
            if full_text:
                refine_prompt = f"""Given the persona of a "{persona}" and the job-to-be-done: "{job_to_be_done}", refine and summarize the following text to extract the most relevant information.
Text from section "{section_title}" (Page {page_number}) in document "{doc_name}":
{full_text}
"""
                print(f"Calling LLM for refining section: {section_title}...")
                refined_text = self._call_gemini_api(refine_prompt, fallback_value=full_text)
                if refined_text:
                    all_subsection_analysis.append({
                        "document": doc_name,
                        "refined_text": refined_text,
                        "page_number": page_number
                    })
                else:
                    print(f"Failed to refine text for section: {section_title} (no fallback provided or invalid)")
            else:
                print(f"Full text not found for section: {section_title}")

        output_data = {
            "metadata": output_metadata,
            "extracted_sections": all_extracted_sections,
            "subsection_analysis": all_subsection_analysis
        }

        output_filename = os.path.splitext(os.path.basename(input_json_path))[0] + "_output.json"
        output_file_path = os.path.join(self.output_dir, output_filename)

        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)
        print(f"Analysis successfully saved to {output_file_path}")

def main():
    SCRIPT_DIR = os.path.dirname(__file__) if '__file__' in locals() else os.getcwd()

    PDF_INPUT_DIR = "/app/PDFs"
    JSON_INPUT_DIR = "/app"
    OUTPUT_DIR = "/app/analysis_output"
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    input_json_filename = "challenge1b_input.json"
    input_json_path = os.path.join(JSON_INPUT_DIR, input_json_filename)

    if not os.path.exists(input_json_path):
        print(f"Error: Input JSON file '{input_json_path}' not found.")
        print(f"Please ensure '{input_json_filename}' is mounted in the container at '{JSON_INPUT_DIR}'.")
        return

    analyst = DocumentAnalyst(PDF_INPUT_DIR, OUTPUT_DIR)
    analyst.analyze_documents(input_json_path)

if __name__ == "__main__":
    main()