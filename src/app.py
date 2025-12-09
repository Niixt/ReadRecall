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

def search_and_load_book(book_name : str,
                         oauth_token: gr.OAuthToken | None,
                         progress : gr.Progress = gr.Progress()) -> tuple[str, gr.update, gr.update]:
    global rag_system
    global config
        
    if not book_name:
        return "Please enter a book name.", gr.update(visible=False), gr.update(visible=False)

    # Get token from OAuth
    token_str = None
    if oauth_token and oauth_token.token:
        token_str = oauth_token.token
    
    if not token_str and not config['running_mode']['local']:
        print("Warning: No OAuth token found. Ensure you are logged in if using gated models.")

    progress(0.1, desc="Searching for book...")
    try:
        res_search = bcl.search_books(book_name)
        if not res_search or 'docs' not in res_search or not res_search['docs']:
            return f"Book '{book_name}' not found.", gr.update(visible=False), gr.update(visible=False)
    except Exception as e:
        return f"Error searching for book: {e}", gr.update(visible=False), gr.update(visible=False)

    progress(0.2, desc="Getting archive link...")
    url_archive = bcl.get_book_archive_page(res_search)
    if not url_archive:
        return "No Internet Archive link found for this book.", gr.update(visible=False), gr.update(visible=False)

    progress(0.3, desc="Fetching book text...")
    try:
        book_text = bcl.fetch_book_text(url_archive)
        if not book_text:
            return "Could not fetch book text.", gr.update(visible=False), gr.update(visible=False)
    except Exception as e:
        return f"Error fetching book text: {e}", gr.update(visible=False), gr.update(visible=False)

    progress(0.5, desc="Cleaning text...")
    cleaned_text = bcl.clean_book_text(book_text, page_break_token='\f')
    
    # Save to file
    documents_dir = config['paths']['documents']
    os.makedirs(documents_dir, exist_ok=True)
    filename = f"{book_name.replace(' ', '_')}_clean.txt"
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
        return f"Error initializing RAG system: {e}", gr.update(visible=False), gr.update(visible=False)

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
    
    status_msg = f"Loaded '{book_name}'. Found {max_chapter} chapters."
    
    # Update slider and show chat
    new_slider_config = {"maximum": max_chapter, "value": 1, "visible": True}
    return status_msg, new_slider_config, 1, gr.update(visible=True)

def chat_response(message: list, chapter_limit: int | None) -> str:
    global rag_system
    if not rag_system:
        return "System not initialized. Please load a book first."
    
    try:
        result = rag_system.query(message[-1]['text'] if isinstance(message[-1], dict) else message[-1], chapter_max=chapter_limit)
        return result['result']
    except Exception as e:
        print(f"Error in chat_response: {e}")
        traceback.print_exc()
        return f"Error during query: {e}"

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
                book_input = gr.Textbox(label="Book Name", placeholder="Enter book title (e.g., Martin Eden)")
                # make the search button click twice when pressed
                search_button = gr.Button("Search & Load", variant="primary")
                status_output = gr.Textbox(label="Status", interactive=False)
                
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

            with gr.Column(scale=2, visible=False) as chat_column:
                chatbot = gr.Chatbot(height=600,
                                     label="Chat with Book",
                                    #  type="messages"
                                    )
                msg = gr.Textbox(label="Ask a question", placeholder="Type your question here...")
                clear = gr.Button("Clear Chat")

    def user(user_message : str, history : list) -> tuple[str, list]:
        return "", history + [{"role": "user", "content": user_message}]

    def bot(history: list, chapter_limit: int) -> list:
        user_message = history[-1]["content"]
        bot_message = chat_response(user_message, chapter_limit)
        history.append({"role": "assistant", "content": bot_message})
        return history

    search_button.click(
        fn=search_and_load_book,
        inputs=[book_input],
        outputs=[status_output, slider_state, slider_value, chat_column]
    )
    
    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, [chatbot, slider_value], chatbot
    )
    
    clear.click(lambda: None, None, chatbot, queue=False)

    def check_login(token: gr.OAuthToken | None):
        if token and token.token:
            return gr.update(visible=True), gr.update(visible=False)
        return gr.update(visible=False), gr.update(visible=True)

    demo.load(fn=check_login, inputs=None, outputs=[main_layout, login_msg])

if __name__ == "__main__":
    # Debugging helper
    if "SPACE_HOST" in os.environ:
        print(f"SPACE_HOST: {os.environ['SPACE_HOST']}")
        
    demo.launch(ssr_mode=False)

# Takes about 8min (467s) to generate answer with gradio
# Fix with fast api