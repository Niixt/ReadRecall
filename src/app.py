import gradio as gr
import os
import sys
import traceback

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add the src directory to sys.path to ensure imports work correctly
current_dir = os.path.dirname(os.path.abspath(__file__))
# src_dir = os.path.join(current_dir, "src")
# sys.path.append(src_dir)
sys.path.append(current_dir)

import loaders.book_content_loader as bcl
import retrieval.rag_retriever as rr
from utils.config_loader import load_config

# Global variables to store the RAG system
rag_system = None
config = load_config()

def search_book_step(book_name: str) -> tuple[str, gr.update, gr.update, dict]:
    if not book_name:
        return "Please enter a book name.", gr.update(visible=False, choices=[]), gr.update(visible=False), {}
    
    # progress(0.1, desc="Searching for book...")
    try:
        res_search = bcl.search_books(book_name)
        candidates = bcl.get_book_candidates(res_search)
        
        if not candidates:
             return f"No books found for '{book_name}' with available text.", gr.update(visible=False, choices=[]), gr.update(visible=False), {}
        
        # Prepare choices for dropdown: list of (label, value) tuples
        choices = [(c['label'], c['ia_id']) for c in candidates]
        books_map = {c['ia_id']: c['label'] for c in candidates}

        if len(candidates) == 1:
            output_message = f"Found 1 book. Make sure it is the one you searched for."
        else:
            output_message = f"Found {len(candidates)} books. Please select one."

        return output_message, gr.update(visible=True, choices=choices, value=choices[0][1]), gr.update(visible=True), books_map

    except Exception as e:
        return f"Error searching for book: {e}", gr.update(visible=False, choices=[]), gr.update(visible=False), {}


def load_book_step(ia_id: str, books_map: dict, oauth_token: gr.OAuthToken | None, progress: gr.Progress = gr.Progress()) -> tuple[str, dict, int, gr.update]:
    global rag_system
    global config
    
    # Default slider state (hidden)
    default_slider = {"visible": False, "maximum": 100, "value": 1}
    
    if not ia_id:
        return "Please select a book.", default_slider, 1, gr.update(visible=False)

    # Get token from OAuth
    token_str = None
    if oauth_token and oauth_token.token:
        token_str = oauth_token.token
    
    if not token_str and not config['running_mode']['local']:
        print("Warning: No OAuth token found. Ensure you are logged in if using gated models.")

    progress(0.2, desc="Getting archive link...")
    
    url_archive = bcl.get_book_archive_url(ia_id)
    
    progress(0.3, desc="Fetching book text...")
    try:
        book_text = bcl.fetch_book_text(url_archive)
        if not book_text:
            return "Could not fetch book text.", default_slider, 1, gr.update(visible=False)
    except Exception as e:
        return f"Error fetching book text: {e}", default_slider, 1, gr.update(visible=False)

    progress(0.5, desc="Cleaning text...")
    cleaned_text = bcl.clean_book_text(book_text, page_break_token='\f')
    
    # Save to file
    documents_dir = config['paths']['documents']
    os.makedirs(documents_dir, exist_ok=True)
    filename = f"{ia_id}_clean.txt"
    file_path = os.path.join(documents_dir, filename)
    
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(cleaned_text)

    progress(0.6, desc="Initializing RAG System (Loading Model)...")
    
    try:
        # Initialize RAG
        rag_system = rr.LocalRAGSystem(
            path_documents=file_path,
            path_token_hf=config['paths']['hf_token'],

            path_custom_prompt=config['paths']['custom_prompt'],
            model_name=config['models']['llm'],
            model_name_embeddings=config['models']['embeddings'],
            model_name_reranker=config['models']['reranker'],
            hf_token_str=token_str,
            run_local=config['running_mode']['local'],
            debug_print=False
        )
    except Exception as e:
        traceback.print_exc()
        return f"Error initializing RAG system: {e}", default_slider, 1, gr.update(visible=False)

    progress(0.9, desc="Analyzing chapters...")
    
    # Determine max chapters
    max_chapter = 0
    if rag_system.docs:
        for doc in rag_system.docs:
            ch_idx = doc.metadata.get("chapter_index", 0)
            if ch_idx > max_chapter:
                max_chapter = ch_idx
    
    max_chapter = max(1, max_chapter)

    progress(1.0, desc="Ready!")

    book_label = books_map.get(ia_id, ia_id) if books_map else ia_id
    status_msg = f"Loaded book '{book_label}'. Found {max_chapter} chapters."
    
    # Update slider and show chat
    new_slider_config = {"maximum": max_chapter, "value": 1, "visible": True}
    return status_msg, new_slider_config, 1, gr.update(visible=True)

def chat_response(message: str | list, chapter_limit: int | None) -> tuple[str, list]:
    global rag_system
    if not rag_system:
        return "System not initialized. Please load a book first.", []
    
    try:
        # Handle input being a string or a list of messages
        query_text = message
        if isinstance(message, list):
            last_msg = message[-1]
            if isinstance(last_msg, dict):
                query_text = last_msg.get('content') or last_msg.get('text')
            else:
                query_text = last_msg
        
        result = rag_system.query(query_text, chapter_max=chapter_limit)
        return result['result'], result['source_documents']
    except Exception as e:
        print(f"Error in chat_response: {e}")
        traceback.print_exc()
        return f"Error during query: {e}", []

def chat_interface_fn(message, history, chapter_limit):
    bot_message, sources = chat_response(message, chapter_limit)
    
    if sources:
        bot_message += "\n\n<details><summary><b>Source Chunks</b></summary>\n\n"
        for i, source in enumerate(sources):
            # Clean up newlines for better markdown rendering in blockquotes
            clean_source = source.replace('\n', ' ')
            bot_message += f"**Chunk {i+1}:**\n> {clean_source}\n\n"
        bot_message += "</details>"
        
    return bot_message

# Gradio Interface
with gr.Blocks(title="ReadRecall") as demo:
    gr.Markdown("# ReadRecall")
    login_msg = gr.Markdown("You will need to login to use the application.")
    if "SPACE_HOST" in os.environ:
        space_host = os.environ.get("SPACE_HOST")
        direct_url = f"https://{space_host}"
        gr.Markdown(f"**Note:** If you are experiencing login loops (especially on mobile), please use the [direct link]({direct_url}) (open in a new tab).")

    with gr.Row():
        gr.LoginButton()

    with gr.Group(visible=False) as main_layout:
        with gr.Row():
            with gr.Column(scale=1):
                book_input = gr.Textbox(label="Search book", placeholder="Enter book title (e.g., Martin Eden) or author", info=f"Note that only books with full text available on [Internet Archive](https://archive.org/) can be loaded.")
                search_button = gr.Button("Search", variant="primary")
                
                book_dropdown = gr.Dropdown(label="Select Book", visible=True, interactive=True)
                load_button = gr.Button("Load Selected Book", variant="secondary", visible=False)
                
                status_output = gr.Textbox(label="Status", interactive=False, lines=3, value="System not initialized.")
                
                # Store slider config in state to trigger re-render
                slider_state = gr.State({"maximum": 100, "value": 1, "visible": False})
                slider_value = gr.State(1)
                
                @gr.render(inputs=[slider_state])
                def render_slider(config):
                    s = gr.Slider(
                        minimum=1,
                        maximum=config["maximum"],
                        value=config["value"],
                        step=1,
                        label="Chapter Limit",
                        visible=config["visible"],
                        info="Limit the context to the first N chapters.",
                        elem_id="chapter_slider",
                        interactive=True
                    )
                    s.change(lambda x: x, s, slider_value)
                
                search_state = gr.State()
                books_state = gr.State({})

            with gr.Column(scale=2, visible=False) as chat_column:
                gr.ChatInterface(
                    fn=chat_interface_fn,
                    additional_inputs=[slider_value],
                    chatbot=gr.Chatbot(height=600, label="Chat with Book"),
                    autofocus=True
                )

    search_button.click(
        fn=search_book_step,
        inputs=[book_input],
        outputs=[status_output, book_dropdown, load_button, books_state]
    )

    load_button.click(
        fn=load_book_step,
        inputs=[book_dropdown, books_state],
        outputs=[status_output, slider_state, slider_value, chat_column]
    )
    
    def check_login(token: gr.OAuthToken | None):
        if token and token.token:
            return gr.update(visible=True), gr.update(visible=False)
        return gr.update(visible=False), gr.update(visible=True)

    demo.load(fn=check_login, inputs=None, outputs=[main_layout, login_msg])

if __name__ == "__main__":
    demo.launch(ssr_mode=False)

# Takes about 8min (467s) to generate answer with gradio
# Fix with fast api