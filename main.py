import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

def init_embeddings():
    # 1. Load the variables from the .env file into system environment
    load_dotenv()
    
    # 2. Verify the key exists before running your app logic
    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError("Critical Error: GOOGLE_API_KEY is missing from your .env file!")
        
    print("✅ Environment variables loaded successfully.")
    
    # 3. Initialize and return the object
    return GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

if __name__ == "__main__":
    # Run the integration
    embeddings = init_embeddings()
    
    # Quick test to confirm it works
    test_vector = embeddings.embed_query("Hello world")
    print(f"✅ Integration complete. Vector dimensions generated: {len(test_vector)}")
