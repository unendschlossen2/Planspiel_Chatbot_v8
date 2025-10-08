import re
from typing import List, Dict, Any

DEFAULT_MAX_CHARS_PER_CHUNK = 2000
DEFAULT_MIN_CHARS_PER_CHUNK = 50
CHUNK_HEADER_OVERVIEW_SUFFIX = " (Übersicht)"
MERGED_HEADER_SEPARATOR = " & "

# Basic print function for feedback, replaces logger
def _print_feedback(message: str, level: str = "INFO"):
    print(f"Chunking-Feedback [{level}]: {message}")

def _create_chunk_dict(
        header_text: str,
        header_level: int,
        content: str,
        source_filename: str,
        original_base_header: str
) -> Dict[str, Any]:
    return {
        "header_text": header_text,
        "header_level": header_level,
        "content": content.strip(),
        "source_filename": source_filename,
        "original_base_header": original_base_header
    }

def filter_missing_content_lines(text: str) -> str:
    lines = text.splitlines()
    filtered_lines = [
        line for line in lines
        if not (line.strip().startswith(">") and "fehlt" in line.lower())
    ]
    return "\n".join(filtered_lines)

def _find_all_headers_in_text(markdown_text: str) -> List[Dict[str, Any]]:
    headers = []
    header_regex = re.compile(r"^(#{1,6})\s+(.*)", re.MULTILINE)
    for match in header_regex.finditer(markdown_text):
        level = len(match.group(1))
        title = match.group(2).strip()
        header_line_start_idx = match.start()
        header_line_end_idx = match.end()

        content_start_idx = header_line_end_idx
        if header_line_end_idx < len(markdown_text) and markdown_text[header_line_end_idx] == '\n':
            content_start_idx = header_line_end_idx + 1
        elif header_line_end_idx + 1 < len(markdown_text) and markdown_text[header_line_end_idx:header_line_end_idx+2] == '\r\n':
            content_start_idx = header_line_end_idx + 2

        headers.append({
            "level": level,
            "title": title,
            "header_start_idx": header_line_start_idx,
            "content_start_idx": content_start_idx
        })
    return headers

def _split_by_found_headers(
        content_to_split: str, sub_headers: List[Dict[str, Any]],
        parent_chunk_dict: Dict[str, Any]
) -> List[Dict[str, Any]]:
    newly_created_chunks = []
    parent_header_text = parent_chunk_dict.get("header_text", "")
    parent_header_level = parent_chunk_dict.get("header_level", 0)
    original_base_header = parent_chunk_dict.get("original_base_header", "")
    source_filename = parent_chunk_dict.get("source_filename", "unbekannt")

    current_pos = 0

    if sub_headers:
        first_sub_header_starts_at = sub_headers[0]['header_start_idx']
        if first_sub_header_starts_at > 0:
            intro_content_raw = content_to_split[current_pos:first_sub_header_starts_at]
            intro_content_filtered = filter_missing_content_lines(intro_content_raw)
            if intro_content_filtered.strip():
                newly_created_chunks.append(_create_chunk_dict(
                    header_text=f"{parent_header_text}{CHUNK_HEADER_OVERVIEW_SUFFIX}",
                    header_level=parent_header_level,
                    content=intro_content_filtered,
                    source_filename=source_filename,
                    original_base_header=original_base_header
                ))

    for i, header_data in enumerate(sub_headers):
        section_content_start_in_parent = header_data['content_start_idx']

        if (i + 1) < len(sub_headers):
            section_content_end_in_parent = sub_headers[i+1]['header_start_idx']
        else:
            section_content_end_in_parent = len(content_to_split)

        raw_section_content = content_to_split[section_content_start_in_parent:section_content_end_in_parent]
        filtered_section_content = filter_missing_content_lines(raw_section_content)

        if filtered_section_content.strip() or header_data['title']:
            newly_created_chunks.append(_create_chunk_dict(
                header_text=header_data['title'],
                header_level=header_data['level'],
                content=filtered_section_content,
                source_filename=source_filename,
                original_base_header=original_base_header
            ))
    return newly_created_chunks


def _handle_oversized_chunk(
        oversized_chunk: Dict[str, Any]
) -> List[Dict[str, Any]]:
    content = oversized_chunk["content"]
    parent_level = oversized_chunk["header_level"]

    headers_in_content = _find_all_headers_in_text(content)

    sub_headers_to_split_by = []
    start_level_for_sub_split = parent_level + 1 if parent_level > 0 else 1
    for level_to_try in range(start_level_for_sub_split, 7):
        found_headers_at_level = [h for h in headers_in_content if h['level'] == level_to_try]
        if found_headers_at_level:
            sub_headers_to_split_by = found_headers_at_level
            break

    if sub_headers_to_split_by:
        split_chunks = _split_by_found_headers(content, sub_headers_to_split_by, oversized_chunk)
        if split_chunks and not (len(split_chunks) == 1 and split_chunks[0]["content"].strip() == content.strip()):
            return split_chunks
        else:
            return [oversized_chunk]
    else:
        return [oversized_chunk]

def _initial_split(
        markdown_text: str, source_filename: str, split_level: int
) -> List[Dict[str, Any]]:
    _print_feedback(f"Initiale Teilung für '{source_filename}' nach H{split_level} Überschriften.")

    all_doc_headers = _find_all_headers_in_text(markdown_text)
    initial_chunks = []

    target_level_headers = [h for h in all_doc_headers if h['level'] == split_level]

    if not target_level_headers:
        _print_feedback(f"Keine H{split_level} Überschriften in '{source_filename}' gefunden. Behandle gesamtes Dokument als einen initialen Chunk.", "WARNING")
        cleaned_full_content = filter_missing_content_lines(markdown_text)
        if cleaned_full_content.strip() or (not markdown_text.strip() and cleaned_full_content == ""):
            initial_chunks.append(_create_chunk_dict(
                header_text=f"{source_filename} (Gesamtes Dokument)",
                header_level=0,
                content=cleaned_full_content,
                source_filename=source_filename,
                original_base_header=f"{source_filename} (Gesamtes Dokument)"
            ))
        return initial_chunks

    for i, header_data in enumerate(target_level_headers):
        chunk_title = header_data['title']
        chunk_level = header_data['level']
        content_start = header_data['content_start_idx']
        content_end = len(markdown_text)

        for next_h_data in all_doc_headers:
            if next_h_data['header_start_idx'] > header_data['header_start_idx'] and next_h_data['level'] <= split_level:
                content_end = next_h_data['header_start_idx']
                break

        raw_chunk_content = markdown_text[content_start:content_end]
        cleaned_content = filter_missing_content_lines(raw_chunk_content)

        if cleaned_content.strip() or chunk_title:
            initial_chunks.append(_create_chunk_dict(
                header_text=chunk_title,
                header_level=chunk_level,
                content=cleaned_content,
                source_filename=source_filename,
                original_base_header=chunk_title
            ))

    _print_feedback(f"{len(initial_chunks)} initiale Abschnitte aus H{split_level}-Teilung gefunden.")
    return initial_chunks


def split_markdown_by_headers(
        markdown_text: str,
        source_filename: str = "unbekannt",
        split_level: int = 3,
        max_chars_per_chunk: int = DEFAULT_MAX_CHARS_PER_CHUNK,
        min_chars_per_chunk: int = DEFAULT_MIN_CHARS_PER_CHUNK,
) -> List[Dict[str, Any]]:

    _print_feedback(f"Starte Markdown-Chunking für '{source_filename}' (max_Zeichen: {max_chars_per_chunk}, min_Zeichen: {min_chars_per_chunk}, Teilungsebene: H{split_level}). Modus: Nur Überschriftenteilung.")

    base_chunks = _initial_split(markdown_text, source_filename, split_level)

    processing_queue = list(base_chunks)
    fully_processed_chunks = []

    MAX_ITERATIONS_OVERSIZED = len(base_chunks) + 100
    iter_count = 0

    while processing_queue and iter_count < MAX_ITERATIONS_OVERSIZED :
        iter_count += 1
        chunk = processing_queue.pop(0)
        current_content_len = len(chunk["content"])

        if current_content_len > max_chars_per_chunk:
            if not chunk["content"].strip() and not chunk["header_text"]:
                continue

            sub_chunks = _handle_oversized_chunk(chunk)

            if len(sub_chunks) == 1 and sub_chunks[0] is chunk:
                _print_feedback(
                    f"Chunk '{chunk['header_text']}' (Länge {current_content_len}) "
                    f"überschreitet max_chars_per_chunk ({max_chars_per_chunk}) und "
                    f"konnte nicht weiter durch verfügbare Überschriften geteilt werden. Dieser Chunk wird ÜBERSPRUNGEN.", "ERROR"
                )
            else:
                processing_queue = sub_chunks + processing_queue

        elif chunk["content"].strip() or chunk["header_text"]:
            fully_processed_chunks.append(chunk)

    if iter_count >= MAX_ITERATIONS_OVERSIZED and processing_queue:
        _print_feedback(f"Maximale Verarbeitungsiterationen ({MAX_ITERATIONS_OVERSIZED}) für übergroße Chunks erreicht. {len(processing_queue)} Chunks verbleiben in der Warteschlange und werden ÜBERSPRUNGEN, falls immer noch übergroß.", "WARNING")

    if not fully_processed_chunks:
        _print_feedback("Keine Chunks zum Zusammenführen vorhanden.")
        return []

    merged_chunks: List[Dict[str, Any]] = []
    i = 0
    while i < len(fully_processed_chunks):
        current_chunk = fully_processed_chunks[i]

        if not current_chunk["content"].strip() and not current_chunk["header_text"].strip():
            i += 1
            continue

        current_content_len = len(current_chunk["content"])

        if current_content_len < min_chars_per_chunk and (i + 1) < len(fully_processed_chunks):
            next_chunk = fully_processed_chunks[i+1]

            if not next_chunk["content"].strip() and not next_chunk["header_text"].strip():
                merged_chunks.append(current_chunk)
                i += 2
                continue

            combined_content_len = current_content_len + len(next_chunk["content"]) + (2 if current_chunk["content"].strip() and next_chunk["content"].strip() else 0)

            if combined_content_len <= max_chars_per_chunk:
                new_content_parts = []
                if current_chunk["content"].strip(): new_content_parts.append(current_chunk["content"].strip())
                if next_chunk["content"].strip(): new_content_parts.append(next_chunk["content"].strip())
                new_content = "\n\n".join(new_content_parts)

                base_cur = current_chunk["original_base_header"]
                base_next = next_chunk["original_base_header"]

                new_header_text = current_chunk["header_text"]
                if base_cur == base_next and base_cur :
                    new_header_text = base_cur
                elif current_chunk["header_text"] != next_chunk["header_text"]:
                    if current_chunk["header_text"].startswith(next_chunk["header_text"] + CHUNK_HEADER_OVERVIEW_SUFFIX):
                        new_header_text = next_chunk["header_text"]
                    elif next_chunk["header_text"].startswith(current_chunk["header_text"] + CHUNK_HEADER_OVERVIEW_SUFFIX):
                        new_header_text = current_chunk["header_text"]
                    else:
                        new_header_text = f"{current_chunk['header_text']}{MERGED_HEADER_SEPARATOR}{next_chunk['header_text']}"

                merged_chunks.append(_create_chunk_dict(
                    header_text=new_header_text,
                    header_level=current_chunk["header_level"],
                    content=new_content,
                    source_filename=current_chunk["source_filename"],
                    original_base_header=base_cur
                ))
                i += 2
            else:
                merged_chunks.append(current_chunk)
                i += 1
        else:
            merged_chunks.append(current_chunk)
            i += 1

    final_valid_chunks = [
        chunk for chunk in merged_chunks if chunk["content"].strip() or chunk["header_text"].strip()
    ]
    _print_feedback(f"Markdown-Chunking-Prozess abgeschlossen. Gebe {len(final_valid_chunks)} Chunks zurück.")
    return final_valid_chunks