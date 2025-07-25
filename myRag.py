
from PyPDF2 import PdfReader
import docx
import openai
import tiktoken
from supabase import create_client, Client
from dotenv import load_dotenv
import os
import json
from openai import OpenAI
import time
# Load environment variables
load_dotenv()


# Configure Supabase
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Configure OpenAI
OPEN_AI_KEY = os.getenv('OPEN_AI_API_KEY')
openai.api_key = OPEN_AI_KEY
client = OpenAI(api_key=OPEN_AI_KEY)
OPENAI_MODEL="gpt-4o-mini"

#OpenAI max response token
MAX_RESPONSE_TOKEN=700

#Embeddings to Supabase Parameters
MAX_TOKENS = 600
OVERLAP_TOKENS = 300

#Vector Search Parameters
MATCH_THRESHOLD= 0.5
MATCH_COUNT= 3

PROCESSED_FILES_LIST = "processed_files.txt"
chat_history = [] # for temporary storage of chat history

# 1. Upload file from a predefined directory
def upload_file(directory: str, filename: str):
    file_path = os.path.join(directory, filename)
    if os.path.exists(file_path):
        print(f"1. SUCCESS: File '{filename}' found in directory '{directory}'.")
        return file_path
    else:
        print(f"1. ERROR: File '{filename}' not found in directory '{directory}'.")
        return None

# 2. Chunk the file
def chunk_text(text: str):
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    
    chunks = []
    for i in range(0, len(tokens), MAX_TOKENS - OVERLAP_TOKENS):
        chunk_tokens = tokens[i:i + MAX_TOKENS]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)
    
    print("2. SUCCESS: Text chunked into", len(chunks), "segments.")
    return chunks

# 3. Generate embeddings
def generate_embedding(text: str):
    try:
       
        response = client.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
)
        embedding = response.data[0].embedding
        print("3. SUCCESS: Embedding generated.")
        return embedding
    except Exception as e:
        print(f"3. ERROR: Failed to generate embedding. {e}")
        return None

# 4. Process file and store in Supabase
def process_file(directory: str, filename: str):
    #file_path = upload_file(directory, filename)
    #if not file_path:
    #    return
    
    # Get the file extension
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
    
    chunks = chunk_text(text)
    
    for i, chunk in enumerate(chunks):
        embedding = generate_embedding(chunk)
        if embedding is None:
            continue
        
        metadata = {"filename": filename, "chunk_index": i}
        data = {
            "content": chunk,
            "embedding": json.dumps(embedding),
            "metadata": json.dumps(metadata),
            "file_path": directory,
            "chunk_index": i
        }
        
        # 5. Store in Supabase
        try:
            response = supabase.table("document_embeddings").insert(data).execute()
            #print(f"5. SUCCESS: Data stored for chunk {i}.")
            loading_display(len(chunks), i)
        except Exception as e:
            print(f"5. ERROR: Failed to store data for chunk {i}. {e}")

def loading_display(chunk_length: int, chunk_index: int):
    """Display a loading message for the chunk processing."""
    total_chunks = chunk_length
    current_chunk = chunk_index + 1
    percentage = (current_chunk / total_chunks) * 100
  
    #for i in range(101):  # Loop from 0% to 100%
    print(f"Processing chunk {current_chunk}/{total_chunks} ({percentage:.2f}%)...", end="", flush=True)  # Overwrites the same line
        
def load_processed_files():
    """Load the list of already processed files."""
    if os.path.exists(PROCESSED_FILES_LIST):
        with open(PROCESSED_FILES_LIST, "r", encoding="utf-8") as file:
            return set(file.read().splitlines())
    return set()

def save_processed_file(filename: str):
    """Add a file to the list of processed files."""
    with open(PROCESSED_FILES_LIST, "a", encoding="utf-8") as file:
        file.write(filename + "\n")

def process_directory(directory: str):
    """Process all files in the directory, skipping already processed files."""
    processed_files = load_processed_files()
    
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        
        # Skip directories and already processed files
        if os.path.isdir(file_path) or filename in processed_files: # print(f"Skipping already processed file: {filename}")
            continue
        
        # Process the file
        try:
            print(f"Processing file: {filename}")
            text = process_file(file_path,filename)
            # Save the processed file to the list
            save_processed_file(filename)
        except Exception as e:
            print(f"Error processing file {filename}: {e}")

def add_message_history(role, content):
    chat_history.append({
        "role": role,
        "content": content,
        "timestamp": time.time()  # Optional for sorting later
    })
# Function to retrieve a message by index
def get_message_history(index):
    if 0 <= index < len(chat_history):
        return chat_history[index]
    else:
        return None  # Handle out-of-bounds access


def query_rag(user_query: str):
    """Retrieve relevant documents from Supabase and generate a response."""
    add_message_history("user", user_query)

    query_embedding = generate_embedding(user_query) #convert text query to embeddings

    
    # Step 2: Perform vector search in Supabase
    response = supabase.rpc(
        "match_documents", 
        {"query_embedding": query_embedding, "match_threshold": MATCH_THRESHOLD, "match_count": MATCH_COUNT}
    ).execute()

  
    # Step 3: Extract relevant documents
    documents = response.data if response.data else []
    
    # If no documents are found, return a default response
    if not documents:
        return {"response": "Hmmm... I couldn't find any relevant information in the knowledge base regarding your question."}
        add_message_history("system", "I couldn't find any relevant information in the knowledge base regarding your question.")
        
    # Step 4: Format the retrieved content
    context = "\n".join([doc["content"] for doc in documents])

   
    # Step 5: Generate AI response using retrieved context
    num_items = len(chat_history) -2

    chat_response = client.chat.completions.create(
    model=OPENAI_MODEL,
    messages=[
        {"role": "system", "content": "You are a helpful assistant with access to relevant documents and chat history. If the relevant documents and chat history has nothing to do with the user query, respond politely that the query is out of context."},
        {"role": "user", "content": f"Here are some relevant documents:\n{context}\n\nUser Query: {user_query}\n\nChat History: {get_message_history(num_items)} "} #get_message_history is -2 means get the 2nd to the last message which is the latest AI response
        ], 
    max_tokens=MAX_RESPONSE_TOKEN 
    )
   
    add_message_history("system", chat_response.choices[0].message.content)
    return {"response": chat_response.choices[0].message.content}


if __name__ == "__main__":
    directory = "./uploaded_docs"
 
    process_directory(directory)
    while True:
        user_query = input("You: ")  # Get user input

        if user_query.lower() == "bye bye":
            print("AI: Goodbye! Have a great day! 😊")
            break  # Exit the loop

        ai_response = query_rag(user_query)  # Call your RAG function
        print(f"AI: {ai_response['response']}")  # Print AI response