import os
import traceback
from typing import List, Dict
import torch
import traceback

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, HuggingFacePipeline
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
    
    def setup_vectorstore(self, texts, model_name="all-MiniLM-L6-v2"):
        """
        Set up the vector store with document embeddings.
        
        Args:
            model_name: Name of the embedding model to use
        """
        embeddings = HuggingFaceEmbeddings(model_name=model_name)

        self.vectorstore = FAISS.from_documents(texts, embeddings)
        print("Vector store created successfully")
    
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
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            token=None,  # Add your HF token here if needed
            trust_remote_code=True,
            revision="main"
        )

        if use_gpu:
            model = model.to("cuda")
        
        # Create a text generation pipeline
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=2048,
            temperature=0.05,
            top_p=5, # 0.1,
            repetition_penalty=1.1
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
            return_source_documents=False,
            chain_type_kwargs={"prompt": prompt}
        )
        
    def initialize(self):
        """
        Initialize the complete RAG pipeline.
        """
        texts = self.load_documents()
        print(1.1)
        self.setup_vectorstore(texts, self.model_name_embeddings)
        print(1.2)
        self.setup_qa_chain()
        print(1.3)
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
            
        result = self.qa_chain({"query": question})
        return result