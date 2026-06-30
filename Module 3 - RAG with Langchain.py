import os
import msal
import requests
from typing import Dict, Any
from langchain_community.document_loaders import OneDriveLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Ensure API keys are set
#os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

# Paste your credentials from Microsoft Entra admin center

CLIENT_ID = "bbbce114-dd92-480e-96a6-8397d5f437a5"
TENANT_ID = "xoriant"  # Use "common" for personal & work accounts
AUTHORITY = f"https://login.microsoftonline.com/xoriota.onmicrosoft.com"

CLIENT_SECRET = "d798Q~0WTI6381e3bfoGWdtGmzvL~xcav4szobpf"
# Use your tenant domain or the 36-character Tenant ID GUID
# Application permissions require the '.default' scope
SCOPES = ["https://graph.microsoft.com/.default"]

def get_access_token():
    # Initialize ConfidentialClientApplication to accept the secret
    app = msal.ConfidentialClientApplication(
        CLIENT_ID,
        authority=AUTHORITY,
        client_credential=CLIENT_SECRET
    )
    
    # Check MSAL cache first
    accounts = app.get_accounts()
    if accounts:
        result = app.acquire_token_silent(SCOPES, account=accounts[0])
        if result:
            return result['access_token']
             

    # Authenticate directly using the Client Secret (Client Credentials Flow)
    result = app.acquire_token_for_client(scopes=SCOPES)

    return result['access_token']

class OneDriveKnowledgeBot:
    def __init__(self, folder_path: str, db_path: str = "./chroma_db"):
        self.folder_path = folder_path
        self.db_path = db_path
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small", 
            api_key=OPENAI_API_KEY
        )
        self.llm = ChatOpenAI(
            model="gpt-4o-mini", 
            temperature=0,
            api_key=OPENAI_API_KEY
        )
        self.vector_store = None
        self.rag_chain = None

    
    def ingest_onedrive_ebooks(self):
        """Connects to OneDrive via Microsoft Graph API, loads documents, and chunks them."""
        print(f"🔄 Connecting to OneDrive and loading files from: {self.folder_path}...")
        
        # 1. Fetch the token via your MSAL application setup
        token = get_access_token()
        if not token:
            raise Exception("❌ Failed to retrieve MSAL Access Token.")

        # 2. Set up the Microsoft Graph API endpoint
        TARGET_USER = "test@xoriota.onmicrosoft.com"
        endpoint = f"https://graph.microsoft.com/v1.0/users/{TARGET_USER}/drive/root:{self.folder_path}:/children"
        headers = {"Authorization": f"Bearer {token}"}
        
        # 3. Request the file list inside the target folder
        response = requests.get(endpoint, headers=headers)
        if response.status_code != 200:
            raise Exception(f"❌ Graph API Error: {response.status_code} - {response.text}")
            
        items = response.json().get('value', [])
        
        # Filter down to actual files only (ignores subfolders)
        files_to_process = [item for item in items if 'file' in item]
        print(f"📂 Found {len(files_to_process)} target files in '{self.folder_path}'.")

        raw_documents = []
        from langchain_core.documents import Document
        
        # 4. Stream down each file directly using its pre-authenticated download URL
        for item in files_to_process:
            file_name = item['name']
            download_url = item.get('@microsoft.graph.downloadUrl')
            
            if not download_url:
                print(f"⚠️ Skipping {file_name}: No download URL available.")
                continue
                
            print(f"📥 Downloading content for: {file_name}...")
            file_response = requests.get(download_url)
            
            if file_response.status_code == 200:
                # Wrap the raw text content into a LangChain Document structure
                doc = Document(
                    page_content=file_response.text,
                    metadata={"source": file_name, "onedrive_id": item['id']}
                )
                raw_documents.append(doc)
            else:
                print(f"⚠️ Failed to download content for {file_name}: Status {file_response.status_code}")

        print(f"✅ Loaded {len(raw_documents)} raw files from OneDrive.")

        # 5. Chunking: Split documents using the Recursive Text Splitter
        print("✂️ Chunking documents...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            length_function=len
        )
        chunked_docs = text_splitter.split_documents(raw_documents)
        print(f"📦 Created {len(chunked_docs)} semantic chunks.")

        # 6. Vector Storage: Persist the chunks into Chroma DB
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
            # Load existing vector store if ingestion was already completed
            self.vector_store = Chroma(
                persist_directory=self.db_path, 
                embedding_function=self.embeddings
            )

        # Set up a retriever with metadata filtering capabilities
        retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )

        # Structure the explicit Prompt Template
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

        # Helper function to format retrieved documents for the prompt context block
        def format_docs(docs):
            formatted = []
            for doc in docs:
                source = doc.metadata.get("source", "Unknown eBook")
                formatted.append(f"--- Document Source: {source} ---\n{doc.page_content}")
            return "\n\n".join(formatted)

        # Constructing the LCEL Chain
        # The pipeline streams data sequentially: Retrieval -> Context Formatting -> Prompt Assignment -> LLM Inference -> Output Parsing
        self.rag_chain = (
            {
                "context": retriever | format_docs, 
                "question": RunnablePassthrough()
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
        print("🚀 RAG LCEL Chain compiled successfully.")

    def ask(self, user_question: str) -> str:
        """Invokes the running LCEL pipeline."""
        if not self.rag_chain:
            raise ValueError("The RAG chain has not been initialized. Call build_lcel_pipeline() first.")
        return self.rag_chain.invoke(user_question)

# --- EXECUTION DEMO ---
if __name__ == "__main__":
    # 1. Initialize Bot
    bot = OneDriveKnowledgeBot(folder_path="/Public/ebooks")  # Adjust the folder path as needed
    
    # 2. Run Ingestion (Typically executed as a background/cron task)
    bot.ingest_onedrive_ebooks() 
    
    # 3. Build Retrieval Pipeline
    bot.build_lcel_pipeline()
    
    # 4. Simulate a prospective user asking questions before downloading
    sample_query = "How many marks are for Cyber Safety?"
    print(f"\nUser Question: {sample_query}")
    
    response = bot.ask(sample_query)
    print(f"\nKnowledgeBot Response:\n{response}")