import os
from typing import Dict, Any
# Swapped OneDriveLoader for local directory and PDF parsing utilities
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Ensure API keys are set
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class LocalKnowledgeBot:
    def __init__(self, local_dir_path: str, db_path: str = "./chroma_db"):
        self.local_dir_path = local_dir_path
        self.db_path = db_path
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.vector_store = None
        self.rag_chain = None

    def ingest_local_ebooks(self):
        """Loads local PDFs file-by-file, safely skipping corrupted files."""
        print(f"🔄 Scanning local directory: {self.local_dir_path}...")
        
        if not os.path.exists(self.local_dir_path):
            raise FileNotFoundError(f"The directory {self.local_dir_path} does not exist.")

        import glob
        # Manually find all PDF files
        pdf_files = glob.glob(os.path.join(self.local_dir_path, "**/*.pdf"), recursive=True)
        print(f"📂 Found {len(pdf_files)} PDF files. Parsing items individually...")

        raw_documents = []
        for file_path in pdf_files:
            try:
                # Initialize loader for a single file
                file_loader = PyMuPDFLoader(file_path)
                pages = file_loader.load()
                raw_documents.extend(pages)
                print(f"  ✅ Successfully parsed: {os.path.basename(file_path)} ({len(pages)} pages)")
            except Exception as e:
                # This catches the ByteStringObject error and keeps going!
                print(f"  ❌ Skipping corrupted file {os.path.basename(file_path)} due to error: {e}")

        print(f"\n✅ Total parsed pages from healthy documents: {len(raw_documents)}")

        if not raw_documents:
            print("⚠️ No documents were successfully parsed. Ingestion halted.")
            return

        # Chunking: Using Recursive Text Splitter to preserve paragraphs
        print("✂️ Chunking documents...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            length_function=len
        )
        chunked_docs = text_splitter.split_documents(raw_documents)
        print(f"📦 Created {len(chunked_docs)} semantic chunks.")

        # Vector Storage
        print("💾 Indexing vectors into persistent storage...")
        self.vector_store = Chroma.from_documents(
            documents=chunked_docs,
            embedding=self.embeddings,
            persist_directory=self.db_path
        )
        print(f"✨ Vector store initialized and saved to '{self.db_path}'")

    def build_lcel_pipeline(self):
        """Constructs the RAG pipeline using LangChain Expression Language (LCEL)."""
        if not self.vector_store:
            # Load the existing local vector store if ingestion was already completed
            self.vector_store = Chroma(
                persist_directory=self.db_path, 
                embedding_function=self.embeddings
            )

        # Set up a retriever
        retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )

        # Explicit Prompt Template
        prompt = ChatPromptTemplate.from_template("""
        You are an expert Knowledge Bot helping users preview eBooks stored in our digital library.
        Users ask questions about the book's content to evaluate if they want to download it.
        
        Use the following extracted context passages to answer the user's question accurately.
        If you do not know the answer based on the context, state clearly that the information is not in the preview text.
        Always cite the book title or source file name (found in the context metadata) in your answer.

        Context:
        {context}

        Question: 
        {question}

        Answer:
        """)

        # Helper function to extract document source metadata for dynamic citations
        def format_docs(docs):
            formatted = []
            for doc in docs:
                # Local loaders automatically store the file path/name in the 'source' key
                full_path = doc.metadata.get("source", "Unknown eBook")
                file_name = os.path.basename(full_path) # Extract just the file name from path
                formatted.append(f"--- Document Source: {file_name} ---\n{doc.page_content}")
            return "\n\n".join(formatted)

        # Constructing the LCEL Chain
        self.rag_chain = (
            {
                "context": retriever | format_docs, 
                "question": RunnablePassthrough()
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
        print("🚀 Local RAG LCEL Chain compiled successfully.")

    def ask(self, user_question: str) -> str:
        """Invokes the running LCEL pipeline."""
        if not self.rag_chain:
            raise ValueError("The RAG chain has not been initialized. Call build_lcel_pipeline() first.")
        return self.rag_chain.invoke(user_question)

# --- EXECUTION DEMO ---
if __name__ == "__main__":
    # Create target folder paths for demo purposes
    LOCAL_EBOOKS_DIR = "./ebooks"
    if not os.path.exists(LOCAL_EBOOKS_DIR):
        os.makedirs(LOCAL_EBOOKS_DIR)
        print(f"📁 Created empty '{LOCAL_EBOOKS_DIR}' directory. Drop your PDF ebooks here!")

    # 1. Initialize Bot targeting local folder
    bot = LocalKnowledgeBot(local_dir_path=LOCAL_EBOOKS_DIR)
    
    # 2. Run Ingestion (Uncomment this line when you have added PDFs to your folder)
    bot.ingest_local_ebooks() 
    
    # 3. Build Retrieval Pipeline
    bot.build_lcel_pipeline()
    
    # 4. Ask questions
    sample_query = "How many marks for Cyber safety?"
    print(f"\nUser Question: {sample_query}")
    
    response = bot.ask(sample_query)
    print(f"\nKnowledgeBot Response:\n{response}")