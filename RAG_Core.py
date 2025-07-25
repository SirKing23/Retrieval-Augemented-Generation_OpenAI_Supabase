from openai import OpenAI                       # for AI

import os                                        # for supabase
from supabase import create_client, Client       # for supabase  

from PyPDF2 import PdfReader                    # for process documents
import docx                                     # for process documents
import json                                     # for process documents
from tqdm import tqdm                           # for process documents / for progress bar while uploading files to supabase
import tiktoken                                 # for chunking
import requests                                 # for embedding using offline ollama server 


from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain.schema import Document


  
class AI:
    # the embedding model for now is text-embedding-ada-002, but we will change it to text-embedding-3-small in the future
    # if we change it, we will need to erase all the embeddings in the supabase database and re-upload them! sad!
    def __init__(self, AI_API_Key :str, Supabase_API_Key:str, Supabase_URL:str, Main_Model :str = "gpt-4o-mini", Embedding_Model :str = "text-embedding-3-small", max_token_response: int = 700, Vector_Search_Threshold :float = 0.5, Vector_Search_Match_Count : int=3, MAX_CHUNK_TOKENS:int = 600, OVERLAP_CHUNK_TOKENS:int = 300): 
       # AI parameters
        self.AI_API_Key = AI_API_Key
        self.AI_Main_Model = Main_Model
        self.AI_Embedding_Model = Embedding_Model
        self.AI_Max_Token_Response = max_token_response
        self.AI_Instance = OpenAI(api_key=AI_API_Key)
        self.AI_Default_No_Response = "Hmmm... I couldn't find any relevant information in the knowledge base regarding your question."
        self.AI_System_Role_Prompt = "You are a helpful assistant with access to relevant documents and chat history. If the relevant documents and chat history has nothing to do with the user query, respond politely that the query is out of context."
        self.MAX_CHUNK_TOKEN = MAX_CHUNK_TOKENS
        self.OVERLAP_CHUNK_TOKENS = OVERLAP_CHUNK_TOKENS

      #Supabase parameters
        self.Supabase_key = Supabase_API_Key
        self.Supabase_Url = Supabase_URL
        self.supabase: Client = create_client(self.Supabase_Url, self.Supabase_key)
        self.Supabase_Vector_Search_Threshold=Vector_Search_Threshold
        self.Supabase_Vector_Search_Match_Count=Vector_Search_Match_Count

        #Files processing parameters
        self.PROCESSED_FILES_LIST = "processed_files.txt"

        #Chat History parameters
        self.chat_history = []  # Initialize an empty list to store chat history

    def generate_embedding(self,text: str):
        try:        
            response = self.AI_Instance.embeddings.create(
            input=text,
            model=self.AI_Embedding_Model)
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
                "match_threshold": self.Supabase_Vector_Search_Threshold,
                "match_count": self.Supabase_Vector_Search_Match_Count
            }
        ).execute()

        documents = response.data if response.data else []

        if not documents:
            return {"response": self.AI_Default_No_Response}

        # Step 3: Extract content from matched docs
        context = "\n".join([doc["content"] for doc in documents])

        # Step 4: Append user message to history
        self.chat_history.append({"role": "user", "content": question})

        # Optional: Truncate history to prevent token overflow
        if len(self.chat_history) > 10:
            self.chat_history = self.chat_history[-10:]

        # Step 5: Construct full message history
        messages = [{"role": "system", "content": self.AI_System_Role_Prompt}]
        messages += self.chat_history
        messages.append({
            "role": "assistant",
            "content": f"[Reference Documents]\n{context}"
        })

        # Step 6: Call OpenAI
        chat_response = self.AI_Instance.chat.completions.create(
            model=self.AI_Main_Model,
            messages=messages,
            max_tokens=self.AI_Max_Token_Response
        )

        # Step 7: Get AI response and append to history
        answer = chat_response.choices[0].message.content
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
        if os.path.exists(self.PROCESSED_FILES_LIST):
            with open(self.PROCESSED_FILES_LIST, "r", encoding="utf-8") as file:
                return set(file.read().splitlines())
        return set()
     
     #process the documents manually - here we use the chunk_text method to chunk the text into smaller parts
    def process_file1(self, directory: str, filename: str):
      

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

    #process the documents using langchain - here we dont use the chunk_text method, because langchain does it for us
    def process_file(self, directory: str, filename: str):

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
            chunk_size=self.MAX_CHUNK_TOKEN,
            chunk_overlap=self.OVERLAP_CHUNK_TOKENS,
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
        with open(self.PROCESSED_FILES_LIST, "a", encoding="utf-8") as file:
            file.write(filename + "\n")
   
    def chunk_text(self,text: str): # Chunk the text into smaller parts for processing
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(text)
            
        chunks = []
        for i in range(0, len(tokens), self.MAX_CHUNK_TOKEN - self.OVERLAP_CHUNK_TOKENS):
            chunk_tokens = tokens[i:i + self.MAX_CHUNK_TOKEN]
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


    