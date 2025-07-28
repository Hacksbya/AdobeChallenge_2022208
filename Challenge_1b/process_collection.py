#!/usr/bin/env python3
"""
process_collection.py

Reads a challenge1b_input.json describing documents, persona, and job,
extracts and ranks relevant sections from PDFs, and writes a challenge1b_output.json.
"""
import os
import json
import argparse
from pathlib import Path
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer, util
from collections import defaultdict, Counter

def load_input(config_path: Path):
    data = json.loads(config_path.read_text(encoding='utf-8'))
    challenge_id = data['challenge_info']['challenge_id']
    persona = data['persona']['role']
    job = data['job_to_be_done']['task']
    # documents: list of dicts with filename and title
    docs = data['documents']
    # resolve PDFs folder
    base = config_path.parent
    pdf_folder = base / 'PDFs'
    doc_paths = [(base / 'PDFs' / d['filename'], d.get('title', d['filename'])) for d in docs]
    output_path = base / 'challenge1b_output.json'
    return challenge_id, persona, job, doc_paths, output_path


def extract_sections(pdf_path: Path):
    """
    Extract sections: list of (section_title, page_number, content_lines)
    using heading detection similar to Round 1A.
    """
    doc = fitz.open(pdf_path)
    spans = []
    # gather spans: (text, size, page, y)
    for page in doc:
        for block in page.get_text('dict')['blocks']:
            for line in block.get('lines', []):
                text = ''.join(span['text'] for span in line['spans']).strip()
                if not text:
                    continue
                size = max(span['size'] for span in line['spans'])
                y = line['spans'][0]['bbox'][1]
                spans.append({'text': text, 'size': size, 'page': page.number+1, 'y': y})
    if not spans:
        return []
    # detect body size
    sizes = [s['size'] for s in spans]
    body_size = Counter(sizes).most_common(1)[0][0]
    # heuristics: headings if size > body_size + 1
    headings = [s for s in spans if s['size'] > body_size + 1]
    # unify into sections: for each heading, collect content until next heading
    # sort spans and headings
    spans_sorted = sorted(spans, key=lambda s: (s['page'], s['y']))
    headings_sorted = sorted(headings, key=lambda s: (s['page'], s['y']))
    sections = []
    for i, h in enumerate(headings_sorted):
        title = h['text']
        page = h['page']
        start_index = spans_sorted.index(h)
        if i+1 < len(headings_sorted):
            next_h = headings_sorted[i+1]
            end_index = spans_sorted.index(next_h)
        else:
            end_index = len(spans_sorted)
        content = [s['text'] for s in spans_sorted[start_index+1:end_index]]
        sections.append({'title': title, 'page': page, 'content': ' '.join(content)})
    return sections


def rank_sections(sections, query, model, top_k=5):
    """
    Rank sections by semantic similarity to query.
    Returns top_k sections.
    """
    # embed query
    q_emb = model.encode(query, convert_to_tensor=True)
    texts = [sec['title'] + ' ' + sec['content'] for sec in sections]
    emb = model.encode(texts, convert_to_tensor=True)
    sims = util.cos_sim(q_emb, emb)[0]
    top = sims.topk(k=min(top_k, len(sections)))
    top_idxs = top.indices.tolist()
    ranked = []
    for rank, idx in enumerate(top_idxs, start=1):
        sec = sections[idx]
        ranked.append({'rank': rank, 'section': sec})
    return ranked


def extract_subsections(model, query, sections, top_k=3):
    """
    For each section, find top_k sentences relevant to query.
    """
    out = []
    q_emb = model.encode(query, convert_to_tensor=True)
    for sec in sections:
        sentences = sec['content'].split('.')
        sent_emb = model.encode(sentences, convert_to_tensor=True)
        sims = util.cos_sim(q_emb, sent_emb)[0]
        best_idx = int(sims.argmax())
        refined = sentences[best_idx].strip()
        out.append({'document': sec['document'], 'refined_text': refined, 'page_number': sec['page']})
    return out


def main():
    import datetime
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help='Path to challenge1b_input.json')
    args = parser.parse_args()
    cid, persona, job, doc_paths, out_path = load_input(Path(args.config))

    # Load embedding model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Extract sections from all documents
    all_sections = []
    for pdf_path, title in doc_paths:
        secs = extract_sections(pdf_path)
        for s in secs:
            s['document'] = pdf_path.name
            all_sections.append(s)

    # Build query
    query = f"{persona}. Task: {job}"

    # Rank all sections, take top 5
    ranked = rank_sections(all_sections, query, model, top_k=5)
    extracted_sections = [
        {
            'document': item['section']['document'],
            'section_title': item['section']['title'],
            'importance_rank': item['rank'],
            'page_number': item['section']['page']
        }
        for item in ranked
    ]

    # Subsection analysis for those top 5
    subsections = extract_subsections(model, query, [item['section'] for item in ranked])

    # Assemble output
    output = {
        'metadata': {
            'input_documents': [p.name for p,_ in doc_paths],
            'persona': persona,
            'job_to_be_done': job,
            'processing_timestamp': datetime.datetime.now().isoformat()
        },
        'extracted_sections': extracted_sections,
        'subsection_analysis': subsections
    }

    # Write JSON
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"Wrote output to {out_path}")
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help='Path to challenge1b_input.json')
    args = parser.parse_args()
    cid, persona, job, doc_paths, out_path = load_input(Path(args.config))

    # load embedding model (<200MB)
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # gather all sections across docs
    all_sections = []
    for pdf_path, title in doc_paths:
        sections = extract_sections(pdf_path)
        for sec in sections:
            sec['document'] = pdf_path.name
            all_sections.append(sec)

    # combine persona+job into query
    query = persona + '. Task: ' + job

    ranked = rank_sections(all_sections, query, model, top_k= len(all_sections))
    extracted_sections = []
    for item in ranked:
        sec = item['section']
        extracted_sections.append({
            'document': sec['document'],
            'section_title': sec['title'],
            'importance_rank': item['rank'],
            'page_number': sec['page']
        })

    # take top N for subanalysis
    top_secs = [item['section'] for item in ranked[:5]]
    subs = extract_subsections(model, query, top_secs)

    output = {
        'metadata': {
            'input_documents': [p.name for p,_ in doc_paths],
            'persona': persona,
            'job_to_be_done': job
        },
        'extracted_sections': extracted_sections,
        'subsection_analysis': subs
    }

    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Wrote output to {out_path}')
