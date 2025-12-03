import os
import sys
import subprocess
import traceback
from typing import List, Dict
import torch
import re
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder

# Fix for deadlock in threaded environments (Gradio)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Monkeypatch subprocess.Popen to handle encoding errors gracefully on Windows
# This fixes UnicodeDecodeError when libraries (like bitsandbytes) capture output containing non-UTF-8 characters
_original_Popen = subprocess.Popen

class SafePopen(_original_Popen):
    def __init__(self, *args, **kwargs):
        # If running in text mode, ensure we handle encoding errors
        if kwargs.get('text') or kwargs.get('universal_newlines'):
            if 'errors' not in kwargs:
                kwargs['errors'] = 'replace'
        super().__init__(*args, **kwargs)

subprocess.Popen = SafePopen

os.environ["BITSANDBYTES_NOWELCOME"] = "1"

from langchain_community.document_loaders import TextLoader, DirectoryLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM, BitsAndBytesConfig, logging as transformers_logging
from huggingface_hub import InferenceClient

class LocalRAGSystem:
    def __init__(self,
                 path_documents: str,
                 path_custom_prompt: str,
                 model_name: str,
                 model_name_embeddings: str,
                 model_name_reranker: str,
                 path_token_hf: str = None,
                 hf_token_str: str = None,
                 run_local: bool = False,
                 debug_print: bool = False):
        """
        Initialize the RAG system with a local LLM.
        
        Args:
            path_documents: Path to documents for the knowledge base
            path_custom_prompt: Path to the custom prompt template file
            model_name: Name of the HuggingFace model to use
            model_name_embeddings: Name of the sentence transformer model for embeddings
            model_name_reranker: Name of the cross-encoder model for re-ranking
            path_token_hf: Path to the file containing the HuggingFace token (optional)
            hf_token_str: The HuggingFace token as a string (optional, takes precedence over path)
            run_local: Whether to run the model locally or use InferenceClient
            debug_print: Whether to print debug information
        """
        
        self.path_documents = path_documents
        self.model_name = model_name
        self.index = None
        self.chunks = []
        self.chunk_metadatas = []
        self.embeddings = None
        self.qa_pipeline = None
        self.run_local = run_local
        
        embed_model = model_name_embeddings
        rerank_model = model_name_reranker
        
        self.embedder = SentenceTransformer(embed_model)
        self.reranker = CrossEncoder(rerank_model)
        self.use_gpu = torch.cuda.is_available()
        
        self.path_token_hf = path_token_hf
        if hf_token_str:
            self.hf_token = hf_token_str
        elif self.path_token_hf:
            self.hf_token = self.load_hf_token()
        else:
            self.hf_token = None
            
        self.pad_token_id = None
        self.debug_print = debug_print

        self.MAX_NEW_TOKENS = 512
        self.CHUNK_SIZE = 1000
        self.CHUNK_OVERLAP = 200
        self.TOP_K_CHUNKS = 10
        self.CANDIDATE_K_CHUNKS = 50 # Fetch more chunks for re-ranking
        
        prompt_path = path_custom_prompt
        
        try:
            with open(prompt_path, "r", encoding="utf-8") as f:
                self.custom_prompt_template = f.read()
        except UnicodeDecodeError:
            # Fallback for Windows-1252 encoded files
            with open(prompt_path, "r", encoding="cp1252") as f:
                self.custom_prompt_template = f.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"{prompt_path} file not found. Please create the file with the desired prompt template.")
        
        self.initialize()

    def load_hf_token(self) -> str:
        """
        Load HuggingFace token from a file.
        
        Args:
            path_token_hf: Path to the file containing the HF token
        Returns:
            The HuggingFace token as a string
        """
        try:
            # Try UTF-8 first
            with open(self.path_token_hf, "r", encoding="utf-8") as file:
                hf_token = file.read().strip()
            return hf_token
        except UnicodeDecodeError:
            print("UTF-8 decode failed for token file, trying cp1252...")
            with open(self.path_token_hf, "r", encoding="cp1252") as file:
                hf_token = file.read().strip()
            return hf_token
        except Exception as e:
            print(f"Error reading HF_TOKEN: {e}")
            return None

    def parse_chapters(self, text: str) -> List[Document]:
        """
        Splits the full text into chapter-level Documents.
        Handles: "Chapter 1", "CHAPTER IV", "IV", "1", "Part One", etc.
        """
        pattern = r'(?m)^\s*((?:CHAPTER|Chapter|chapter|PART|Part|part|BOOK|Book|book)\s+(?:[IVXLCDM]+|\d+|[A-Za-z]+)|(?:[IVXLCDM]+)|(?:\d+))\s*$'
        
        matches = list(re.finditer(pattern, text))
        
        documents = []
        
        # Handle text before the first chapter (Preface, Title Page, etc.)
        if not matches:
            return [Document(page_content=text, metadata={"chapter": "Full Text"})]
            
        if matches[0].start() > 0:
            preamble = text[:matches[0].start()].strip()
            if preamble:
                documents.append(Document(page_content=preamble,
                                          metadata={"chapter": "Preamble", "chapter_index": 0}))
        
        for i in range(len(matches)):
            # Start of this chapter
            start = matches[i].start()
            # End is start of next chapter, or end of text
            end = matches[i+1].start() if i + 1 < len(matches) else len(text)
            
            # Extract title and content
            title = matches[i].group(1).strip()
            content = text[start:end].strip()
            
            documents.append(Document(page_content=content,
                                      metadata={"chapter": title, "chapter_index": i + 1}))
            
        return documents

    def load_documents(self) -> List:
        """
        Load documents, split by chapter, then chunk recursively.
        """
        # 1. Load the raw text content
        full_text = ""
        if os.path.isfile(self.path_documents):
            try:
                with open(self.path_documents, 'r', encoding='utf-8') as f:
                    full_text = f.read()
            except UnicodeDecodeError:
                print(f"Warning: UTF-8 decoding failed for {self.path_documents}. Retrying with cp1252...")
                # Fallback to cp1252 (Windows default) and replace any remaining unknown characters
                with open(self.path_documents, 'r', encoding='cp1252', errors='replace') as f:
                    full_text = f.read()
        else:
            # For directories, you might need to loop through files
            loader = DirectoryLoader(self.path_documents, glob="**/*.txt")
            raw_docs = loader.load()
            full_text = "\n\n".join([d.page_content for d in raw_docs])
        
        # 2. Split into Chapter Documents
        chapter_docs = self.parse_chapters(full_text)
        
        # 3. Chunk within chapters (so chunks don't cross chapter boundaries)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.CHUNK_SIZE,
            chunk_overlap=self.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
            keep_separator=False
        )
        
        # split_documents preserves the metadata (chapter title) from chapter_docs
        final_chunks = text_splitter.split_documents(chapter_docs)
        if self.debug_print:
            print(f"Created {len(final_chunks)} chunks from {len(chapter_docs)} chapters.")
        return final_chunks
    
    def setup_vectorstore(self, texts: List[Document]):
        """
        Set up the vector store with document embeddings using FAISS.
        
        Args:
            texts: list of langchain Document objects (already split)
        """
        if self.debug_print:
            print("Creating embeddings and vector store...")
        # Extract text content from Document objects
        self.chunks = [doc.page_content for doc in texts]
        self.chunk_metadatas = [doc.metadata for doc in texts] # Store metadata

        if not self.chunks:
            print("No documents to index.")
            return

        # Create embeddings
        embeddings = self.embedder.encode(self.chunks)
        self.embeddings = embeddings # Store embeddings for filtered search

        # Build FAISS index
        d = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(d)
        self.index.add(embeddings)
        if self.debug_print:
            print(f"Vector store created with {len(self.chunks)} chunks.")
    
    def load_transformers_pipeline(self):
        """
        Load the local language model using HuggingFace Transformers.
        
        Returns:
            The loaded language model pipeline
        """
        if self.debug_print:
            print(f"Loading model locally: {self.model_name}")
        try:
            # Attempt to load in 4-bit to save VRAM (fits easily in 16GB)
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4"
            )
            if self.debug_print:
                print("Attempting to load model with 4-bit quantization...")
            tokenizer = AutoTokenizer.from_pretrained(self.model_name,
                                                      token=self.hf_token)
            if self.debug_print:
                print("Tokenizer loaded successfully.")
            # Initialize pipeline
            qa_pipeline = pipeline(
                "text-generation",
                model=self.model_name,
                model_kwargs={
                    "quantization_config": quantization_config,
                    "device_map": "auto"
                },
                tokenizer=tokenizer,
                token=self.hf_token,
                max_new_tokens=self.MAX_NEW_TOKENS,
                do_sample=True,
                temperature=0.6,
                top_p=0.9
            )
            self.pad_token_id = tokenizer.pad_token_id
            if self.debug_print:
                print("Model loaded successfully with 4-bit quantization!")
        except Exception as e:
            if self.debug_print:
                print(f"Failed to load 4-bit model: {e}")
                print("Falling back to standard loading (might require more VRAM)...")
            # Fallback to fp16 if 4-bit fails (might OOM on 16GB depending on context)
            qa_pipeline = pipeline(
                "text-generation",
                model=self.model_name,
                model_kwargs={
                    "torch_dtype": torch.float16,
                    "device_map": "auto"
                },
                token=self.hf_token,
                max_new_tokens=self.MAX_NEW_TOKENS,
                do_sample=True,
                temperature=0.6
            )
            if self.debug_print:
                print("Model loaded successfully (fp16)!")

        return qa_pipeline

    def load_inference_client(self):
        """
        Load the Inference Client for HuggingFace.
        
        Returns:
            The loaded inference client
        """
        if self.debug_print:
            print(f"Loading Inference Client for model: {self.model_name}")
        
        try:
            client = InferenceClient(token=self.hf_token, model=self.model_name)
            if self.debug_print:
                print("Inference Client initialized successfully.")
            return client
        except Exception as e:
            print(f"Error initializing Inference Client: {e}")
            raise e
    
    def initialize(self):
        """
        Initialize the complete RAG pipeline.
        """
        texts = self.load_documents()
        self.docs = texts 
        self.setup_vectorstore(texts)
        
        if self.run_local:
            self.qa_pipeline = self.load_transformers_pipeline()
        else:
            self.qa_pipeline = self.load_inference_client()
            
        if self.debug_print:
            print("RAG system initialized successfully!")
        
    def query(self, question: str, chapter_max: int = None) -> Dict:
        """
        Query the RAG system.
        
        Args:
            question: The question to ask
            
        Returns:
            Dict containing the answer and source documents
        """
        if not self.qa_pipeline or not self.index:
            raise ValueError("RAG system not initialized. Call initialize() first.")
            
        # Search
        if self.debug_print:
            print(f"Querying for question: {question} with chapter_max={chapter_max}")
        q_emb = self.embedder.encode([question])


        # Determine search space
        if chapter_max is not None:
            # Find the cutoff index for the allowed chapters
            # Since chunks are sequential, we find the first chunk that exceeds the limit
            cutoff_index = len(self.chunks)
            for i, meta in enumerate(self.chunk_metadatas):
                if meta.get('chapter_index', 0) > chapter_max:
                    cutoff_index = i
                    break
            
            if cutoff_index == 0:
                return {"result": "I can't answer that yet, as there is no book content available up to this chapter.", "source_documents": []}

            # Slice the embeddings and create a temporary index
            # This ensures we ONLY search within the allowed scope
            subset_embeddings = self.embeddings[:cutoff_index]
            d = subset_embeddings.shape[1]
            search_index = faiss.IndexFlatL2(d)
            search_index.add(subset_embeddings)
            k = min(self.CANDIDATE_K_CHUNKS, cutoff_index)
            D, I = search_index.search(q_emb, k)
            # Retrieve chunks from the subset (indices match self.chunks[:cutoff_index])
            retrieved_chunks = [self.chunks[i] for i in I[0]]
        else:
            # Full search
            k = min(self.CANDIDATE_K_CHUNKS, len(self.chunks))
            D, I = self.index.search(q_emb, k)
            retrieved_chunks = [self.chunks[i] for i in I[0]]
        if self.debug_print:
            print(f"Retrieved indices: {I}")
        
        
        # Deduplicate
        unique_chunks = []
        seen = set()
        for chunk in retrieved_chunks:
            if chunk not in seen:
                unique_chunks.append(chunk)
                seen.add(chunk)
        # Re-ranking
        if unique_chunks:
            pairs = [[question, chunk] for chunk in unique_chunks]
            scores = self.reranker.predict(pairs)
            
            # Zip, sort by score, and unzip
            scored_chunks = sorted(zip(unique_chunks, scores), key=lambda x: x[1], reverse=True)
            
            # Take top K
            final_chunks = [chunk for chunk, score in scored_chunks[:self.TOP_K_CHUNKS]]
            if self.debug_print:
                print("Re-ranking scores:")
                for chunk, score in scored_chunks[:self.TOP_K_CHUNKS]:
                    print(f"Score: {score:.4f} | Chunk: {chunk[:50]}...")
        else:
            final_chunks = []
        
        relevant_context = "\n\n...\n\n".join(final_chunks)
        if self.debug_print:
            print(f"Relevant context for question:\n{relevant_context}")
        # Construct prompt using Llama 3 chat template forma
        messages = [
            {"role": "system", "content": self.custom_prompt_template},
            {"role": "user", "content": f"Context:\n{relevant_context}\n\nQuestion: {question}"}
        ]
        
        # Generate
        try:
            if self.run_local:
                # Use transformers pipeline
                outputs = self.qa_pipeline(
                    messages,
                    pad_token_id=self.pad_token_id if self.pad_token_id is not None else 0,
                    max_new_tokens=self.MAX_NEW_TOKENS,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.9
                )
                if self.debug_print:
                    print(f"Raw model output: {outputs}")
                try:
                    generated_text = outputs[0]['generated_text'][-1]['content']
                except (IndexError, KeyError, TypeError) as e:
                    print(f"Error extracting response from model output: {e}")
                    generated_text = "I apologize, but I encountered an error generating the response."
            else:
                # Use InferenceClient
                response = ""
                for message in self.qa_pipeline.chat_completion(
                    messages,
                    max_tokens=self.MAX_NEW_TOKENS,
                    stream=True,
                    temperature=0.6,
                    top_p=0.9,
                ):
                    choices = message.choices
                    token = ""
                    if len(choices) and choices[0].delta.content:
                        token = choices[0].delta.content
                    response += token
                
                generated_text = response
            
        except Exception as e:
            print(f"Error generating response: {e}")
            traceback.print_exc()
            generated_text = "I apologize, but I encountered an error generating the response."
        
        return {"result": generated_text, "source_documents": final_chunks}