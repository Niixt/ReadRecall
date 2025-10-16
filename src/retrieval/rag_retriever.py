import os
import traceback
from typing import List, Dict
import torch
import traceback

# from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma
from chromadb.config import Settings
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM


class LocalRAGSystem:
    def __init__(self, documents_path: str, model_name: str = "HuggingFaceTB/SmolLM2-135M", model_name_embeddings: str = "all-MiniLM-L6-v2"):
        """
        Initialize the RAG system with a local LLM.
        
        Args:
            documents_path: Path to documents for the knowledge base
            model_name: Name of the HuggingFace model to use
        """
        self.documents_path = documents_path
        self.model_name = model_name
        self.model_name_embeddings = model_name_embeddings
        self.vectorstore = None
        self.qa_chain = None
        self.use_gpu = torch.cuda.is_available()
        
        try:
            with open("utils/custom_prompt.txt", "r") as f:
                self.custom_prompt_template = f.read()
        except FileNotFoundError:
            raise FileNotFoundError("utils/custom_prompt.txt file not found. Please create the file with the desired prompt template.")
        self.initialize()

    def load_documents(self) -> List:
        """
        Load documents from the specified path.
        
        Returns:
            List of document chunks
        """
        if os.path.isfile(self.documents_path):
            loader = TextLoader(self.documents_path)
            documents = loader.load()
        else:
            loader = DirectoryLoader(self.documents_path, glob="**/*.txt")
            documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150
        )
        
        texts = text_splitter.split_documents(documents)
        print(f"Loaded {len(texts)} document chunks")
        return texts
    
    def setup_vectorstore(self, texts, model_name="all-MiniLM-L6-v2", persist_directory="chroma_db"):
        """
        Set up the vector store with document embeddings.
        
        Args:
            texts: list of langchain Document objects (already split)
            model_name: embedding model name
            persist_directory: where to persist the chroma DB
        """
        embeddings = HuggingFaceEmbeddings(model_name=model_name)

        # If a persisted DB exists and you want to reuse it, you can construct Chroma with the same persist_directory.
        # Using from_documents will overwrite/add to the persisted store.
        self.vectorstore = Chroma(
            collection_name="documents",
            embedding_function=embeddings
        )

        # add documents to the vector store
        ids = [f"doc_{i}" for i in range(len(texts))]
        self.vectorstore.add_documents(texts, ids=ids)

        # Persist to disk so future runs can load it
        try:
            self.vectorstore.persist()
        except Exception:
            # persist() may not be required/available depending on your langchain/chromadb version
            pass

        # Debug/logging (Chroma objects don't expose `.index.ntotal`)
        try:
            # number of embeddings is available via .count() depending on langchain version
            n_vectors = self.vectorstore._collection.count() if hasattr(self.vectorstore, "_collection") else "unknown"
        except Exception:
            n_vectors = "unknown"

        print(f"Vector store created (persist_directory={persist_directory}), vectors={n_vectors}")
    
    def load_local_llm(self, use_gpu=False):
        """
        Load the local language model using HuggingFace.
        
        Args:
            use_gpu: Whether to use GPU if available
        Returns:
            The loaded language model
        """
        print(f"Loading model: {self.model_name}")
        # Load the model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_name,
                                                  use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            token=None,  # Add your HF token here if needed
            trust_remote_code=True,
            revision="main"
        )
        model.generation_config.pad_token_id = tokenizer.eos_token_id

        if use_gpu:
            model = model.to("cuda")
        
        # Create a text generation pipeline
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.1,
            top_p=0.8, # 0.9,
            top_k=20, # 50
            min_p=0,
            repetition_penalty=1.3,
            no_repeat_ngram_size=4,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )
        
        # Create a LangChain wrapper around the pipeline
        llm = HuggingFacePipeline(pipeline=pipe)
        return llm
    
    def setup_qa_chain(self):
        """
        Set up the question-answering chain with custom prompt template.
        """
        llm = self.load_local_llm(self.use_gpu)
        
        # Create the prompt template
        prompt = PromptTemplate(
            template=self.custom_prompt_template,
            input_variables=["context", "question"]
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_kwargs={"k": 3}
            ),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )
        
    def initialize(self):
        """
        Initialize the complete RAG pipeline.
        """
        texts = self.load_documents()
        self.docs = texts 
        return
        self.setup_vectorstore(texts, self.model_name_embeddings)
        self.setup_qa_chain()
        print("RAG system initialized successfully!")
        
    def query(self, question: str) -> Dict:
        """
        Query the RAG system.
        
        Args:
            question: The question to ask
            
        Returns:
            Dict containing the answer and source documents
        """
        if not self.qa_chain:
            raise ValueError("RAG system not initialized. Call initialize() first.")
            
        result = self.qa_chain.invoke({"query": question})
        return result