#multi agent and memory layer
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain.agents import create_agent

load_dotenv()

#first database
PRODUCTS = {
    "wireless headphones": {"price": 79.99,  "rating": 4.6, "description": "Over-ear Bluetooth, 30-hr battery, active noise cancellation."},
    "smart watch":         {"price": 199.99, "rating": 4.3, "description": "Tracks heart rate and sleep. 5-day battery, water-resistant."},
    "mechanical keyboard": {"price": 129.00, "rating": 4.8, "description": "Tenkeyless, Cherry MX Brown switches, per-key RGB."},
    "laptop stand":        {"price": 34.99,  "rating": 4.5, "description": "Adjustable aluminium, fits 11-17 inch laptops, folds flat."},
}
#tool to find the product name from query sentance
@tool
def get_product(name: str) -> str:
    """Look up a product by name and return its price, rating, and description."""
    p = PRODUCTS.get(name.lower())
    #condition checks if p is empty then returns not available
    if not p:
        return f"Product not found. Available: {', '.join(PRODUCTS)}"
    return str(p)

llm = ChatGoogleGenerativeAI(model="gemma-4-31b-it", temperature = 0)
#agent configuration
agent = create_agent(
    llm,
    tools = [get_product],
    system_prompt="You are a helpful product assistant for an online tech store.",
)
#ask function for invoking agent
def ask(question: str):
    result=agent.invoke({"messages":[{"role": "user","content":question}]})
    print(f"\n {result["messages"][-1].content}")

#for specific queries you can get answer as per too, else you will be getting general info as per LLMs, like these 2 queries 
ask("tell me something about wireless headphones")
#ask("what is the price of MI mobile")

#for this query you will be getting proper answer
ask("what is the price of wireless headphones")

#second database
REVIEWS = {
    "wireless headphones": {"reviews": 1262, "rating": 4.6},
    "smart watch":         {"reviews": 340,  "rating": 3.9},
    "mechanical keyboard": {"reviews": 67,   "rating": 4.8},
    "laptop stand":        {"reviews": 781,  "rating": 4.5},
}


#this line is called doc string which specifies task and scope of agent
@tool
def get_review(name: str) -> str:
    """Look up a product review by prduct name. Return product name, number of reviews and rating"""
    r=REVIEWS
    if not r:
        return f"Review is not available"
    return str(r)

agent2 = create_agent(
    llm,
    tools = [get_product, get_review],
    system_prompt="You are a helpful product assistant for an Online tech store.",
)

def ask2(question: str):
    result=agent2.invoke({"messages": [{"role": "user", "content":question}]})
    print(result["messages"][-1].content) 
    #this will give thinking of llm, question, all the answers, final answer, metadat
    #print(f"\n {result}")

ask2("what are the reviews smart watch")

#adding memory to LLM agent
from langgraph.checkpoint.memory import InMemorySaver

agent2 = create_agent(
    llm,
    tools = [get_product, get_review],
    system_prompt="You are a helpful product assistant for an Online tech store.",
    checkpointer=InMemorySaver()
)

def ask2(question: str):
    config = {"configurable": {"thread_id": "user_session_1"}}
    result = agent2.invoke(
        {"messages":[{"role":"user","content": question}]},
        config
    )
    print(result["messages"][-1].content)

ask2("what is price of wireless headphones.")
ask2("what are reviews on this product")