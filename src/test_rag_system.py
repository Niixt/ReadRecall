import os
from typing import List, Dict

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEndpoint, HuggingFacePipeline
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
import torch

class LocalRAGSystem:
    def __init__(self, documents_path: str, model_name: str = "HuggingFaceTB/SmolLM2-135M"):
        """
        Initialize the RAG system with a local LLM.
        
        Args:
            documents_path: Path to documents for the knowledge base
            model_name: Name of the HuggingFace model to use
        """
        self.documents_path = documents_path
        self.model_name = model_name
        self.vectorstore = None
        self.qa_chain = None
        self.use_gpu = torch.cuda.is_available()
        
    def load_documents(self) -> List:
        """Load documents from the specified path."""
        if os.path.isfile(self.documents_path):
            loader = TextLoader(self.documents_path)
            documents = loader.load()
        else:
            loader = DirectoryLoader(self.documents_path, glob="**/*.txt")
            documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        texts = text_splitter.split_documents(documents)
        print(f"Loaded {len(texts)} document chunks")
        return texts
    
    def setup_vectorstore(self, texts, model_name="all-MiniLM-L6-v2"):
        """Set up the vector store with document embeddings."""
        # Use a smaller embedding model that can run locally
        embeddings = HuggingFaceEmbeddings(model_name=model_name)

        self.vectorstore = FAISS.from_documents(texts, embeddings)
        print("Vector store created successfully")
    
    def load_local_llm(self, use_gpu=False):
        """Load the local language model using HuggingFace."""
        print(f"Loading model: {self.model_name}")
        
        # Load the model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
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
            max_length=512,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1
        )
        
        # Create a LangChain wrapper around the pipeline
        llm = HuggingFacePipeline(pipeline=pipe)
        return llm
    
    def setup_qa_chain(self):
        """Set up the question-answering chain."""
        llm = self.load_local_llm(self.use_gpu)
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_kwargs={"k": 3}
            ),
            return_source_documents=True
        )
        
    def initialize(self):
        """Initialize the complete RAG pipeline."""
        texts = self.load_documents()
        self.setup_vectorstore(texts)
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
            
        result = self.qa_chain({"query": question})
        return result

if __name__ == "__main__":
    # Example usage
    documents_path = "documents"  # Path to your documents folder
    
    rag = LocalRAGSystem(documents_path)
    rag.initialize()
    
    while True:
        question = input("\nEnter your question (or 'quit' to exit): ")
        if question.lower() == "quit":
            break
            
        result = rag.query(question)
        print("\nAnswer:", result["result"])
        print("\nSources:")
        for i, doc in enumerate(result["source_documents"]):
            print(f"Source {i+1}:", doc.page_content[:150], "...\n")