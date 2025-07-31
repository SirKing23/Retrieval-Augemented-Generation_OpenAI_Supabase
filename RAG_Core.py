from openai import OpenAI                       # for AI

import os                                        # for supabase
from supabase import create_client, Client       # for supabase  

from PyPDF2 import PdfReader                    # for process documents
import docx                                     # for process documents
import json                                     # for process documents
from tqdm import tqdm                           # for process documents / for progress bar while uploading files to supabase
import tiktoken                                 # for chunking
import requests                                 # for embedding using offline ollama server 


from langchain.text_splitter import RecursiveCharacterTextSplitter                          # for process documents using LangChain   
from langchain.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader              # for process documents using LangChain  
from langchain.schema import Document                                                       # for process documents using LangChain  


class RAGSystem:

    def __init__(self, AI_API_Key: str, Supabase_API_Key: str, Supabase_URL: str, 
                 Main_Model: str = "gpt-4o-mini", Embedding_Model: str = "text-embedding-3-small", 
                 max_token_response: int = 700, Vector_Search_Threshold: float = 0.5, 
                 Vector_Search_Match_Count: int = 3, MAX_CHUNK_TOKENS: int = 600, 
                 OVERLAP_CHUNK_TOKENS: int = 300): 
        
        # Input validation
        if not AI_API_Key or not isinstance(AI_API_Key, str):
            raise ValueError("AI_API_Key must be a non-empty string")
        if not Supabase_API_Key or not isinstance(Supabase_API_Key, str):
            raise ValueError("Supabase_API_Key must be a non-empty string")
        if not Supabase_URL or not isinstance(Supabase_URL, str):
            raise ValueError("Supabase_URL must be a non-empty string")
        if max_token_response <= 0:
            raise ValueError("max_token_response must be positive")
        if not (0.0 <= Vector_Search_Threshold <= 1.0):
            raise ValueError("Vector_Search_Threshold must be between 0.0 and 1.0")
        if Vector_Search_Match_Count <= 0:
            raise ValueError("Vector_Search_Match_Count must be positive")
        if MAX_CHUNK_TOKENS <= OVERLAP_CHUNK_TOKENS:
            raise ValueError("MAX_CHUNK_TOKENS must be greater than OVERLAP_CHUNK_TOKENS")
            
        # AI parameters
        self.ai_api_key = AI_API_Key
        self.ai_main_model = Main_Model
        self.ai_embedding_model = Embedding_Model
        self.ai_max_token_response = max_token_response
        self.ai_instance = OpenAI(api_key=AI_API_Key)
        self.ai_default_no_response = "I couldn't find any relevant information in the knowledge base regarding your question."
        self.ai_system_role_prompt = "You are a helpful assistant with access to relevant documents and chat history. If the relevant documents and chat history has nothing to do with the user query, respond politely that the query is out of context."
        self.max_chunk_tokens = MAX_CHUNK_TOKENS
        self.overlap_chunk_tokens = OVERLAP_CHUNK_TOKENS

        # Supabase parameters
        self.supabase_key = Supabase_API_Key
        self.supabase_url = Supabase_URL
        try:
            self.supabase: Client = create_client(self.supabase_url, self.supabase_key)
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Supabase: {e}")
        self.vector_search_threshold = Vector_Search_Threshold
        self.vector_search_match_count = Vector_Search_Match_Count

        # Files processing parameters
        self.processed_files_list = "processed_files.txt"

        # Chat History parameters
        self.chat_history = []  # Initialize an empty list to store chat history
        self.is_initial_session = False

    def generate_embedding(self,text: str):
        try:        
            response = self.ai_instance.embeddings.create(
            input=text,
            model=self.ai_embedding_model)
            embedding = response.data[0].embedding           
            return embedding
        except Exception as e:  
            print(f"Embedding error: {e}")          
            return None
    
    def generate_embedding2(self, text: str): # use this method if you want to use the offline ollama server for embedding - 768 token limit
        try:
            response = requests.post(
                "http://localhost:11434/api/embeddings",
                json={
                    "model": "nomic-embed-text",  # e.g., "nomic-embed-text"
                    "prompt": text
                }
            )
            response.raise_for_status()
            embedding = response.json()["embedding"]
            return embedding
        except Exception as e:
            print(f"Embedding error: {e}")
            return None

    def answer_this(self, question: str):
        # Step 1: Convert query to embedding
        query_embedding = self.generate_embedding(question)

        # Step 2: Vector search in Supabase
        response = self.supabase.rpc(
            "match_documents", 
            {
                "query_embedding": query_embedding,
                "match_threshold": self.vector_search_threshold,
                "match_count": self.vector_search_match_count
            }
        ).execute()

        documents = response.data if response.data else []

        if not documents:
            return {"response": self.ai_default_no_response}

        # Step 3: Extract content from matched docs
        context = "\n".join([doc["content"] for doc in documents])

        # Step 4: Optional: Truncate history to prevent token overflow
        if len(self.chat_history) > 10:
            self.chat_history = self.chat_history[-10:]

        # Step 5: Construct full message history
        messages = []

        # We only add the AI prompt in the first chat
        if not self.is_initial_session:
            messages = [{"role": "system", "content": self.ai_system_role_prompt}]
            self.is_initial_session = True
               
        if self.chat_history:
            messages.append({"role": "system", "content": "Chat history follows:"})
            messages += self.chat_history
            
        # Add the current user query with retrieved context
        combined_input = f"[Reference Documents]\n{context}\n\n{question}"
        messages.append({"role": "user", "content": combined_input})

        # Step 6: Call OpenAI
        chat_response = self.ai_instance.chat.completions.create(
            model=self.ai_main_model,
            messages=messages,
            max_tokens=self.ai_max_token_response
        )

        # Step 7: Get AI response and append to history
        answer = chat_response.choices[0].message.content

        # Update chat history for memory
        self.chat_history.append({"role": "user", "content": question})
        self.chat_history.append({"role": "assistant", "content": answer})

        return {"response": answer}


    def initialize_files(self, file_directory: str):
        """Process all files in the directory, skipping already processed files."""
        processed_files = self.load_processed_files()
    
        for filename in os.listdir(file_directory):
            file_path = os.path.join(file_directory, filename)
            
            # Skip directories and already processed files
            if os.path.isdir(file_path) or filename in processed_files: # print(f"Skipping already processed file: {filename}")
                continue
        
            # Process the file
            try:
                print(f"Processing file: {filename}")
                text = self.process_file(file_directory,filename)
                # Save the processed file to the list
                self.save_processed_file(filename)
            except Exception as e:
                print(f"Error processing file {filename}: {e}")

    def load_processed_files(self): #Load the list of already processed files.
        """Load the list of already processed files."""
        if os.path.exists(self.processed_files_list):
            with open(self.processed_files_list, "r", encoding="utf-8") as file:
                return set(file.read().splitlines())
        return set()
     
     #process the documents manually - here we use the chunk_text method to chunk the text into smaller parts
   
    def process_file1(self, directory: str, filename: str): #process the documents using manual chunking - we use chunk_text method
      

        file_extension = os.path.splitext(filename)[1].lower()
        text = ""

        if file_extension == ".txt":
            with open(directory, "r", encoding="utf-8") as file:
                text = file.read()

        elif file_extension == ".pdf":
            reader = PdfReader(directory)
            for page in reader.pages:
                text += page.extract_text()

        elif file_extension == ".docx":
            doc = docx.Document(directory)
            for para in doc.paragraphs:
                text += para.text + "\n"

        else:
            raise ValueError(f"Unsupported file type: {file_extension}")

        chunks = self.chunk_text(text)

        print(f"Processing '{filename}' with {len(chunks)} chunks...\n")

        for i, chunk in tqdm(enumerate(chunks), total=len(chunks), desc="Uploading chunks"):
            embedding = self.generate_embedding(chunk)
            if embedding is None:
                continue

            metadata = {"filename": filename, "chunk_index": i}
            data = {
                "content": chunk,
                "embedding": embedding.tolist(),  # Don't stringify if Supabase handles vectors
                "metadata": {"filename": filename, "chunk_index": i},
                "file_path": directory,
                "chunk_index": i
            }


            try:
                self.supabase.table("document_embeddings").insert(data).execute()
            except Exception as e:
                print(f"\n❌ ERROR at chunk {i}: {e}")
  
    def process_file(self, directory: str, filename: str):   #process the documents using langchain - here we dont use the chunk_text method, because langchain does it for us

        filename = filename.strip()  # Remove leading/trailing spaces
        file_path = os.path.normpath(os.path.join(directory, filename))

        file_path = os.path.join(directory, filename)
        file_extension = os.path.splitext(filename)[1].lower()

        # Use LangChain loaders based on file extension
        if file_extension == ".txt":
            loader = TextLoader(file_path, encoding="utf-8")
        elif file_extension == ".pdf":
            loader = PyPDFLoader(file_path)
        elif file_extension == ".docx":
            loader = Docx2txtLoader(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")

        # Load the document (LangChain returns a list of Document objects)
        documents = loader.load()

        # Optional: Add metadata to each document (e.g., filename)
        for doc in documents:
            doc.metadata["filename"] = filename

        # Use RecursiveCharacterTextSplitter for intelligent chunking
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.max_chunk_tokens,
            chunk_overlap=self.overlap_chunk_tokens,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        split_docs = splitter.split_documents(documents)

        print(f"Processing '{filename}' with {len(split_docs)} chunks...\n")

        for i, doc in tqdm(enumerate(split_docs), total=len(split_docs), desc="Uploading chunks"):
            embedding = self.generate_embedding(doc.page_content)
            if embedding is None:
                continue

            metadata = doc.metadata.copy()
            metadata["chunk_index"] = i

            data = {
                "content": doc.page_content,
               "embedding": embedding,  # Don't stringify if Supabase handles vectors
                "metadata": json.dumps(metadata),
                "file_path": file_path,
                "chunk_index": i
            }

            try:
                self.supabase.table("document_embeddings").insert(data).execute()
            except Exception as e:
                print(f"\n❌ ERROR at chunk {i}: {e}")

    def save_processed_file(self,filename: str): #Save the filename to the list of processed files.
        """Add a file to the list of processed files."""
        with open(self.processed_files_list, "a", encoding="utf-8") as file:
            file.write(filename + "\n")
   
    def chunk_text(self,text: str): # Chunk the text into smaller parts for processing
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(text)
            
        chunks = []
        for i in range(0, len(tokens), self.max_chunk_tokens - self.overlap_chunk_tokens):
            chunk_tokens = tokens[i:i + self.max_chunk_tokens]
            chunk_text = encoding.decode(chunk_tokens)
            chunks.append(chunk_text)
        
        return chunks
        
    def upload_folder (self, file_directory :str): #Only necessary if we want to upload all the files from a folder
        self.file_directory=file_directory
    
        for filename in os.listdir(file_directory):
            file_path = os.path.join(file_directory, filename)
            
        # Process the file
        try:
            self.process_file(file_path,filename)
        except Exception as e:
            print(f"Error processing file {filename}: {e}")


    