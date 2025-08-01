import os
import dotenv
from dotenv import load_dotenv
load_dotenv()
import RAG_Core

if __name__ == "__main__":
  MyClient= RAG_Core.AI(os.getenv('OPEN_AI_API_KEY'),os.getenv('SUPABASE_KEY'),os.getenv('SUPABASE_URL'))
  print ("Please wait while we initialize the files in the directory...")
  MyClient.initialize_files("C:\\Users\\YourKing\\Desktop\\RAG_File_Upload")
  print ("Files initialized successfully.")


  print("Welcome! How can I help you today?")
  while True:
      question = input("You: ")
      if question.strip().lower() in ["exit", "quit"]:
          break
      elif question.strip().lower() == "reset":
          MyClient.reset_chat_history()
          print("History reset.")
          continue
      answer = MyClient.answer_this(question)
      print("AI:", answer["response"])
  

