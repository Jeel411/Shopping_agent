from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import chromadb

load_dotenv()

client = chromadb.Client()

collection = client.create_collection("news_articals")

#chromadb uses by default model all-MiniLM-L6-v2 from sentance-transformer
#collection is like sql table with name news... and i'm adding sentances using ids for embedding, this is 2D array.
collection.add(
    ids=["id1","id2","id3","id4"],
    documents=[
        "Apple is good",
        "Apple is multi national smart phone company",
        "Tesla is advance car company",
        "SpaceX is rocket company"
    ]
)

#to fetch created, stored embeddings and documents
data = collection.get(
    include=["documents","embeddings","metadatas"]
)

for i,emb in enumerate(data["embeddings"]):
    print(f"\n Documents{i+1}")
    print("Text:", data["documents"][i])
    print("Embedding length:",len(emb))
    print("First 10 values:", emb[:10])

#this user question which would be connecting with spaceX and Tesla through semantic search
result = collection.query(
    query_texts=["this is for elen musk"],
    n_results=2
)

print(f"\n result is \n {result}")