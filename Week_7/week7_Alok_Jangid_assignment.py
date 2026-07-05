import os
import chromadb
from pypdf import PdfReader
from google import genai
import dotenv

dotenv.load_dotenv()


# Set the GEMINI_API_KEY environment variable
os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")

# Initialize the Gemini client
client = genai.Client()

def get_embedding(text: str) -> list[float]:
    print(f"\n      - Generating embedding for text snippet: '{text[:70].replace('\n', ' ')}...'")
    response = client.models.embed_content(
        model="gemini-embedding-001",  
        contents=text
    )
    embedding_values = response.embeddings[0].values

    return embedding_values

def extract_and_chunk_pdf(pdf_path: str, chunk_size: int = 1000) -> list[str]:
    """Reads a PDF and breaks it into smaller chunks of text."""
    reader = PdfReader(pdf_path)
    full_text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            full_text += page_text + "\n"
            
    # Simple chunking: Split text every 1000 characters
    chunks = [full_text[i:i + chunk_size] for i in range(0, len(full_text), chunk_size)]
    print(f"  -> Extracted {len(full_text)} characters and created {len(chunks)} chunks.")
    print(f"  -> Example Chunk 1: '{chunks[0][:150].replace('\n', ' ')}...'")
    return chunks

def main():
    print("Welcome to the Simple PDF RAG CLI")
     
     
    pdf_path = "GirdharKumawat.pdf"

    print("📄 Reading and chunking PDF...")
    chunks = extract_and_chunk_pdf(pdf_path)
    print(f"✅ PDF processed into {len(chunks)} chunks.")
    
 
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.create_collection(name="pdf_collection")
   
    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk)
        # Create a unique ID from the PDF name and chunk index
        unique_id = f"{pdf_path}_chunk_{i}"
        collection.upsert(
            ids=[unique_id],
            embeddings=[embedding],
            documents=[chunk]
        )
       
    
    # 5. The Chat Loop
    while True:
        query = input("You: ").strip()
        if query.lower() == 'exit':
            break
        if not query:
            continue
            
        print("\n" + "="*50)
        print(f"🔍 Processing query: '{query}'")
        
        # A. Convert user question to vector
        
        query_vector = get_embedding(query)
        
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=3
        )
        
        retrieved_chunks = results.get("documents", [[]])[0]
        print(f"  -> Found {len(retrieved_chunks)} relevant chunks.")
        context = "\n\n---\n\n".join(retrieved_chunks)
        
        # C. Create the prompt for Gemini
        prompt = (
            f"Answer the user's question using ONLY the context provided below.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}"
        ) 
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        
        print(f"\nGemini: {response.text}\n")
        print("="*50 + "\n")

if __name__ == "__main__":
    main()