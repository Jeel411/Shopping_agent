#creates env variable google api key into venv and loads value from .env file
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

#choosing model and sending user query 
llm = ChatGoogleGenerativeAI(model="gemma-4-31b-it")
#print(llm.invoke("what is capital of gujarat").text)

#writing system prompt which goes along side user query
response = llm.invoke([
    ["system","Give answer in one word only"],
    ["human","what is capital of india"]
])

print(response.text)