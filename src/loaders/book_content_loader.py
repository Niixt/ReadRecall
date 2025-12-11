import requests
from bs4 import BeautifulSoup
import random
import re
import unicodedata
from collections import Counter
from typing import List


def search_books(query: str) -> dict | None:
    """
    Search for books using the Open Library API.
    Args:
        query (str): The search query string.
    Returns:
        dict: The JSON response from the API containing search results.
    """
    url = f"https://openlibrary.org/search.json?q={query}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return None
    

def get_book_cover(value: str, key_type='olid', size='M') -> str:
    """
    Fetch the book cover from the Open Library Covers API.
    Args:
        value (str): The identifier value (e.g., OLID, ISBN).
        key_type (str): The type of identifier ('olid', 'isbn', 'oclc', 'lccn').
        size (str): The size of the cover ('S', 'M', 'L').
    Returns:
        str: The URL of the book cover image.
    """
    url = f"https://covers.openlibrary.org/b/{key_type}/{value}-{size}.jpg"
    return url

def get_book_archive_url(identifier: str) -> str:
    """
    Get the Internet Archive URL for a book given its identifier.
    Args:
        identifier (str): The Internet Archive identifier for the book.
    Returns:
        str: The URL of the book on Internet Archive.
    """
    base_url_archive = "https://archive.org/details/"
    return base_url_archive + identifier

def get_book_archive_page(json_result: dict) -> str | None:
    """
    Get the Internet Archive URL for the first book in the search results.
    Args:
        json_result (dict): The JSON response from the Open Library search API.
    Returns:
        str or None: The URL of the book on Internet Archive, or None if not found.
    """
    base_url_archive = "https://archive.org/details/"

    # Get the first book ia identifier 
    # Take the first result 
    if 'docs' not in json_result or len(json_result['docs']) == 0:
        return None
    
    # Find the first book with an Internet Archive identifier
    for doc in json_result['docs']:
        ia_list = doc.get('ia', [])
        if ia_list:
            return base_url_archive + ia_list[0]
            
    return None

def get_book_candidates(json_result: dict) -> List[dict]:
    """
    Extract available books with Internet Archive IDs from search results.
    Args:
        json_result (dict): The JSON response from the Open Library search API.
    Returns:
        List[dict]: A list of dictionaries containing book metadata and IA ID.
    """
    candidates = []
    if 'docs' not in json_result:
        return candidates
    
    for doc in json_result['docs']:
        ia_list = doc.get('ia', [])
        if ia_list:
            # Use the first IA ID
            ia_id = ia_list[0]
            if is_full_text_available(f"https://archive.org/details/{ia_id}") is None:
                continue  # skip if no text found
            title = doc.get('title', 'Unknown Title')
            author_name = doc.get('author_name', ['Unknown Author'])
            author = author_name[0] if author_name else 'Unknown Author'
            year = doc.get('first_publish_year', 'N/A')
            
            candidates.append({
                'label': f"{title} ({year}) by {author}",
                'ia_id': ia_id,
                'title': title,
                'value': ia_id # For Gradio Dropdown
            })
    return candidates

def is_full_text_available(url_archive: str) -> None | str:
    """
    Check if full text is available for a book on its Internet Archive page.
    Args:
        url_archive (str): The URL of the book on Internet Archive.
    Returns:
        None or str: The URL of the full text if available, else None.
    """
    response_archive_page = requests.get(url_archive)
    response_archive_page.raise_for_status()  # Check for download errors

    soup_archive_page = BeautifulSoup(response_archive_page.content, 'html.parser')
    # Find the FULL TEXT (txt) link
    content_div = soup_archive_page.find_all('a', class_='format-summary download-pill')

    for link in content_div:
        if 'FULL TEXT' in link.text:
            return link.get('href')
    return None

def fetch_book_text(url_archive: str, debug_print: bool = False) -> str | None:
    """
    Fetch the full text of a book from its Internet Archive page.
    Args:
        url_archive (str): The URL of the book on Internet Archive.
    Returns:
        str or None: The full text of the book, or None if not found.
    """
    url_full_text = is_full_text_available(url_archive)
    if url_full_text is None:
        return None
    
    if not url_full_text.startswith('http'):
        url_full_text = "https://archive.org" + url_full_text
    
    # print("Full text URL:", url_full_text)
    response_full_text = requests.get(url_full_text)
    response_full_text.raise_for_status()  # Check for download errors

    soup_full_text = BeautifulSoup(response_full_text.content, 'html.parser')
    # Find the main content container
    content_div = soup_full_text.find('pre')
    book_text = None
    if content_div:
        book_text = content_div.get_text()
        if debug_print:
            print(f'Book found with length: {len(book_text)} characters')
            # Print a random excerpt from the book
            start_index = random.randint(0, len(book_text) - 200)
            print(book_text[start_index:start_index + 200])
    
    if not book_text:
        print("No book text found.")
        return None
    return book_text


def write_book_to_file(book_text: str, search_query: str, debug_print: bool = False) -> str:
    """
    Write the book text to a file.
    Args:
        book_text (str): The full text of the book.
        search_query (str): The search query used to find the book.
    Returns:
        str: The path to the file where the book text is written.
    """
    filename = search_query.replace(" ", "_")
    path = f"documents/{filename}.txt"
    with open(path, "w", encoding='utf-8') as f:
        f.write(book_text)

    if debug_print:
        print(f"Book text written to {path}")
    return path

def normalize_unicode(text: str) -> str:
    LIGATURES = {
        "ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl", "ﬅ": "ft", "ﬆ": "st"
    }
    text = unicodedata.normalize("NFKC", text)
    for lig, repl in LIGATURES.items():
        text = text.replace(lig, repl)
    # normalize different dashes and quotes
    text = text.replace("\u2013", "-").replace("\u2014", " - ").replace("\u00AC", "-")
    text = text.replace("\u2018", "'").replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    # remove weird non-breaking spaces
    text = text.replace("\u00A0", " ")
    return text

def remove_uc_only_lines(text: str) -> str:
    # heuristics: many header/footer lines are ALL CAPS short words (publisher, title)
    lines = text.splitlines()
    out = []
    for ln in lines:
        ln_stripped = ln.strip()
        if not ln_stripped:
            out.append(ln)
            continue
        # if line is short and mostly uppercase and not a sentence, drop it
        if len(ln_stripped) < 60:
            alpha_chars = re.sub(r'[^A-Za-z]', '', ln_stripped)
            if alpha_chars and alpha_chars.upper() == alpha_chars and len(alpha_chars) > 3:
                # avoid deleting lines that look like sentences (end with . ? !)
                if not re.search(r'[.?!]\s*$', ln_stripped) and 'chapter' not in ln_stripped.lower():
                    # skip likely header/footer
                    continue
        out.append(ln)
    return "\n".join(out)

def remove_page_numbers(text: str) -> str:
    # Remove lines that are only numbers or numbers with small decorations
    lines = text.splitlines()
    newlines = []
    for ln in lines:
        if re.fullmatch(r'\s*\d+\s*', ln):
            continue
        # also remove lines like "Page 12" (common)
        if re.fullmatch(r'\s*(page|pg|p\.)\s*\d+\s*', ln, flags=re.IGNORECASE):
            continue
        newlines.append(ln)
    return "\n".join(newlines)

def fix_hyphenation(text: str) -> str:
    # merge words split with hyphen at EOL:
    # pattern: 'hy- \nphenated' or 'hy-\nphenated' -> 'hyphenated'
    text = re.sub(r'([A-Za-z])-\n([A-Za-z])', r'\1\2', text)
    # also handle hyphen + spaces + newline
    text = re.sub(r'([A-Za-z])-\s*\n\s*([A-Za-z])', r'\1\2', text)
    return text

def reflow_paragraphs(text: str) -> str:
    # Reflow lines within paragraphs: paragraphs separated by empty lines.
    parts = re.split(r'\n{2,}', text)
    reflowed = []
    for p in parts:
        # strip leading/trailing spaces per paragraph
        lines = [ln.strip() for ln in p.splitlines() if ln.strip()]
        if not lines:
            reflowed.append("")
            continue
        # join with single space
        joined = " ".join(lines)
        # collapse multiple spaces
        joined = re.sub(r'\s+', ' ', joined).strip()
        reflowed.append(joined)
    return "\n".join(reflowed)

def dedupe_repeated_header_footer(text: str, page_break_token: str = None) -> str:
    # If you have a page break token (like '\f') use it. Otherwise guess by rough pages.
    if page_break_token and page_break_token in text:
        pages = text.split(page_break_token)
    else:
        # attempt naive page split if the source used form feed markers, else split by approx page length
        approx_chars = 3000
        pages = [text[i:i+approx_chars] for i in range(0, len(text), approx_chars)]
    header_cands = Counter()
    footer_cands = Counter()
    first_lines = []
    last_lines = []
    for p in pages:
        lines = [ln.strip() for ln in p.splitlines() if ln.strip()]
        if not lines:
            continue
        first_lines.append(lines[0][:120])
        last_lines.append(lines[-1][:120])
    # find frequent first/last lines
    for ln in first_lines:
        header_cands[ln] += 1
    for ln in last_lines:
        footer_cands[ln] += 1
    # choose candidates that appear on many pages (threshold)
    n_pages = max(1, len(pages))
    headers = {ln for ln, c in header_cands.items() if c > max(1, n_pages*0.4)}
    footers = {ln for ln, c in footer_cands.items() if c > max(1, n_pages*0.4)}
    # remove these exact lines from the text
    if headers or footers:
        def drop_headers_footers_line(ln):
            s = ln.strip()
            if s in headers or s in footers:
                return False
            return True
        out_lines = [ln for ln in text.splitlines() if drop_headers_footers_line(ln)]
        return "\n".join(out_lines)
    return text

def collapse_whitespace(text: str) -> str:
    text = re.sub(r'[ \t]+', ' ', text)
    # normalize repeated blank lines to two newlines for paragraph separation
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def clean_book_text(raw: str, page_break_token: str = None) -> str:
    t = raw
    t = normalize_unicode(t)
    t = dedupe_repeated_header_footer(t, page_break_token=page_break_token)
    t = remove_page_numbers(t)
    t = remove_uc_only_lines(t)
    t = fix_hyphenation(t)
    t = reflow_paragraphs(t)
    t = collapse_whitespace(t)
    return t