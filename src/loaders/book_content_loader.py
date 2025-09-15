import requests
from bs4 import BeautifulSoup
import random

def search_books(query):
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
    

def get_book_cover(value, key_type='olid', size='M'):
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

def get_book_archive_page(json_result):
    """
    Get the Internet Archive URL for the first book in the search results.
    Args:
        json_result (dict): The JSON response from the Open Library search API.
    Returns:
        str or None: The URL of the book on Internet Archive, or None if not found.
    """
    base_url_archive = "https://archive.org/details/"

    # Get the first book ia identifier 
    # TODO handle multiple results, let user choose 
    ia_list = json_result['docs'][0].get('ia', [])
    if ia_list == []:
        return None
    return base_url_archive + ia_list[0]

def fetch_book_text(url_archive):
    """
    Fetch the full text of a book from its Internet Archive page.
    Args:
        url_archive (str): The URL of the book on Internet Archive.
    Returns:
        str or None: The full text of the book, or None if not found.
    """
    response_archive_page = requests.get(url_archive)
    response_archive_page.raise_for_status()  # Check for download errors

    soup_archive_page = BeautifulSoup(response_archive_page.content, 'html.parser')
    # Find the FULL TEXT (txt) link
    content_div = soup_archive_page.find_all('a', class_='format-summary download-pill')

    url_full_text = None
    for link in content_div:
        if 'FULL TEXT' in link.text:
            url_full_text = link.get('href')
            break
    if not url_full_text:
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
        print(f'Book found with length: {len(book_text)} characters')
        # Print a random excerpt from the book
        start_index = random.randint(0, len(book_text) - 200)
        print(book_text[start_index:start_index + 200])
    
    if not book_text:
        print("No book text found.")
        return None
    return book_text


def write_book_to_file(book_text, search_query):
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
    print(f"Book text written to {path}")
    return path