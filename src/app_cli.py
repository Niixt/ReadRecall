import os
import sys
import traceback
import time
from tqdm import tqdm


os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add the current directory to sys.path to ensure imports work correctly
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

import loaders.book_content_loader as bcl
import retrieval.rag_retriever as rr
from utils.config_loader import load_config

# Global variables to store the RAG system
rag_system = None
config = load_config()

def search_and_load_book(book_name: str) -> tuple[str, bool]:
    global rag_system
    global config
    
    if not book_name:
        return "Please enter a book name.", False

    with tqdm(total=6, desc="Starting...") as pbar:
        pbar.set_description("Searching for book...")
        try:
            res_search = bcl.search_books(book_name)
            if not res_search or 'docs' not in res_search or not res_search['docs']:
                return f"Book '{book_name}' not found.", False
        except Exception as e:
            return f"Error searching for book: {e}", False
        pbar.update(1)

        pbar.set_description("Getting archive link...")
        url_archive = bcl.get_book_archive_page(res_search)
        if not url_archive:
            return "No Internet Archive link found for this book.", False
        pbar.update(1)

        pbar.set_description("Fetching book text...")
        try:
            book_text = bcl.fetch_book_text(url_archive)
            if not book_text:
                return "Could not fetch book text.", False
        except Exception as e:
            return f"Error fetching book text: {e}", False
        pbar.update(1)

        pbar.set_description("Cleaning text...")
        cleaned_text = bcl.clean_book_text(book_text, page_break_token='\f')
        
        # Save to file
        documents_dir = config['paths']['documents']
        os.makedirs(documents_dir, exist_ok=True)
        filename = f"{book_name.replace(' ', '_')}_clean.txt"
        file_path = os.path.join(documents_dir, filename)
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(cleaned_text)
        pbar.update(1)

        pbar.set_description("Initializing RAG System (Loading Model)...")
        
        # Path to HF token
        token_path = config['paths']['hf_token']
        
        try:
            # Initialize RAG
            rag_system = rr.LocalRAGSystem(
                path_documents=file_path,
                path_token_hf=token_path,
                path_custom_prompt=config['paths']['custom_prompt'],
                model_name=config['models']['llm'],
                model_name_embeddings=config['models']['embeddings'],
                model_name_reranker=config['models']['reranker'],
                debug_print=False
            )
        except Exception as e:
            return f"Error initializing RAG system: {e}", False 
        pbar.update(1)

        pbar.set_description("Analyzing chapters...")
        
        # Determine max chapters
        max_chapter = 0
        if rag_system.docs:
            for doc in rag_system.docs:
                ch_idx = doc.metadata.get("chapter_index", 0)
                if ch_idx > max_chapter:
                    max_chapter = ch_idx
        
        max_chapter = max(1, max_chapter)
        pbar.update(1)
        pbar.set_description("Ready!")
    
    status_msg = f"Loaded '{book_name}'. Found {max_chapter} chapters."
    
    return status_msg, True

def chat_response(message : str, history : list, chapter_limit : int | None = None):
    global rag_system
    if not rag_system:
        return "System not initialized. Please load a book first."
    
    try:
        if rag_system.debug_print:
            print(f"Received message: {message} of type: {type(message)}")

        result = rag_system.query(message, chapter_max=chapter_limit)

        return result['result']
    except Exception as e:
        print(f"Error in chat_response: {e}")
        traceback.print_exc()
        return f"Error during query: {e}"

    
if __name__ == "__main__":
    print("Welcome to the ReadRecall CLI!")
    print("Type 'exit' or 'quit' to leave the application.\n")
    book_name = input("Search and load a book: ")
    status = search_and_load_book(book_name)
    while not status[1]:
        print("Please try again.")
        book_name = input("Search and load a book: ")
        if book_name.lower() in ['exit', 'quit']:
            sys.exit(0)
        status = search_and_load_book(book_name)
    print(status[0])
    chapter_limit = input("Set chapter limit (number) leave empty for no limit: ")
    chapter_limit = int(chapter_limit) if chapter_limit != '' else None
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            break
        
        start_time = time.time()
        response = chat_response(user_input, [], chapter_limit)
        end_time = time.time()
        generation_time = end_time - start_time
        print(f"Bot: {response}\n\n(Generated in {generation_time:.2f} seconds)")

# Takes about 7s to generate answer