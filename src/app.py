import gradio as gr
import os
import sys
import traceback


os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add the current directory to sys.path to ensure imports work correctly
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

import loaders.book_content_loader as bcl
import retrieval.rag_retriever as rr

# Global variables to store the RAG system
rag_system = None

def search_and_load_book(book_name : str,
                         progress : gr.Progress = gr.Progress()) -> tuple[str, gr.update, gr.update]:
    global rag_system
    
    if not book_name:
        return "Please enter a book name.", gr.update(visible=False), gr.update(visible=False)

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
    documents_dir = os.path.join(current_dir, "documents")
    os.makedirs(documents_dir, exist_ok=True)
    filename = f"{book_name.replace(' ', '_')}_clean.txt"
    file_path = os.path.join(documents_dir, filename)
    
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(cleaned_text)

    progress(0.6, desc="Initializing RAG System (Loading Model)...")
    
    # Path to HF token - assuming it's in the parent directory of src
    token_path = os.path.join(os.path.dirname(current_dir), "HF_TOKEN")
    
    try:
        # Initialize RAG
        rag_system = rr.LocalRAGSystem(
            documents_path=file_path,
            path_token_hf=token_path,
            model_name="meta-llama/Meta-Llama-3-8B-Instruct"
        )
    except Exception as e:
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
    return status_msg, gr.update(maximum=max_chapter, value=max_chapter, visible=True), gr.update(visible=True)

def chat_response(message: list, chapter_limit: int | None, debug_print : bool = False) -> str:
    global rag_system
    if not rag_system:
        return "System not initialized. Please load a book first."
    
    try:
        if type(message) is str:
            result = rag_system.query(message, chapter_max=chapter_limit)
        else:
            result = rag_system.query(message[-1]['text'], chapter_max=chapter_limit, debug_print=debug_print)
        return result['result']
    except Exception as e:
        print(f"Error in chat_response: {e}")
        traceback.print_exc()
        return f"Error during query: {e}"

# Gradio Interface
with gr.Blocks(title="ReadRecall") as demo:
    gr.Markdown("# ReadRecall Application")
    
    with gr.Row():
        with gr.Column(scale=1):
            book_input = gr.Textbox(label="Book Name", placeholder="Enter book title (e.g., Martin Eden)")
            search_button = gr.Button("Search & Load", variant="primary")
            status_output = gr.Textbox(label="Status", interactive=False)
            
            # Slider initially hidden
            chapter_slider = gr.Slider(
                minimum=1, 
                maximum=100, 
                value=1, 
                step=1, 
                label="Chapter Limit", 
                visible=False,
                info="Limit the context to the first N chapters."
            )

        with gr.Column(scale=2, visible=False) as chat_column:
            chatbot = gr.Chatbot(height=600, label="Chat with Book")
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
        outputs=[status_output, chapter_slider, chat_column]
    )
    
    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, [chatbot, chapter_slider], chatbot
    )
    
    clear.click(lambda: None, None, chatbot, queue=False)

if __name__ == "__main__":
    demo.launch(
        share=True
    )


# Takes about 8min (467s) to generate answer with gradio