#do one google search for each functions
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

#loading PDF page by page
from langchain_community.document_loaders import PyPDFLoader
PDF = "4-RAG/telecom_guide.pdf"

loader=PyPDFLoader(PDF)
pages = loader.load()
 
#page_content contains the actual content of page
print(f"loaded {len(pages)} from PDF")
#print(f"\n content \n {pages[0].page_content[:500]}")

#chunking is needed as sometime this page/doc can be very big that it exceed's context window limit
#600 characters of chunks ~150 tokens and it is recursive

from langchain_text_splitters import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 600,
    chunk_overlap = 100,
    separators = ["\n\n","\n",","," "],
)

chunks = splitter.split_documents(pages)

print(len(chunks))
print(f"\n {chunks[0]}")
print(f"\n {chunks[0].page_content}") 

#creating embeddings and storing in chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = Chroma.from_documents(chunks, embeddings)

print(f"\n {vector_store._collection.count()}") 

#retriving top k chunks
retriever = vector_store.as_retriever(search_kwargs={"k":3})

query="what is VoLTE and how to improve call quality"
retrieved = retriever.invoke(query)

#building RAG pipeline

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

system_prompt="""
You are a helpful telecom assistant.
Answer the question using ONLY the context provided below.
If the context does not contain enough information, say so clearly.
    
    Context:
    {context}
"""
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human","{query}"),
])

#this one will join all the retrived chunks(here 3 chunks) into single text
def format_docs(docs):
    return "\n\n --- \n\n".join(doc.page_content for doc in docs)

llm = ChatGoogleGenerativeAI(model="gemma-4-31b-it", temperature=0, reasoning_format="parsed")

chain = (
    {"context":retriever | format_docs, "query": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

query= "how does international roaming work and what charges should I expect?"

print(f"Q: {query}\n")
print("A:",chain.invoke(query))