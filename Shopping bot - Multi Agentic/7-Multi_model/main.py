#for reading images and processing using LLM
#OCR can be used for this for saving tokens
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
import base64       #to convert images into embedding, 
from pathlib import Path
from langchain.agents import create_agent

load_dotenv()

#reading image with full path and converting with base64, rb means reading in binary mode
image_path = Path(__file__).with_name("blood_work.png")
with image_path.open("rb") as f:
    image_b64=base64.b64encode(f.read()).decode()

#image_b64[:200]  #prints embedding of converted image

llm = ChatGroq(model="qwen/qwen3.6-27b")
llm2 = ChatGoogleGenerativeAI(model="gemma-4-31b-it", temperature = 0)

#imoprted from core module in order to provide LLM input prompt
message = HumanMessage(content=[
    {"type": "image_url", "image_url": {"url":f"data:image/png;base64,{image_b64}"}},
    {"type":"text", "text":"This is blood work data. Extract all test results and flag any values outside normal range. format it properly"}
])
#why this is not getting formatted like \n is getting printed and it is not getting in new line
#response = llm2.invoke([message])
#print(response.content)

#tool created
from langchain.tools import tool
@tool
def get_diet_plan(condition:str) -> str:
    """As per the Given health condition, return diet plan. Condition must be one of: normal, high_colesterol, high_sugar."""

    diet_plans = {
        "high_cholesterol": {
            "eat":        ["fruits", "vegetables", "whole grains", "lean protein"],
            "do_not_eat": ["red meat", "fried food", "full-fat dairy", "processed snacks"],
        },
        "high_sugar": {
            "eat":        ["vegetables", "whole grains", "legumes", "nuts"],
            "do_not_eat": ["white rice", "white sugar", "junk food", "sugary drinks"],
        },
        "normal": {
            "eat":        ["vegetables", "fruits", "whole grains", "lean protein"],
            "do_not_eat": ["excessive sugar", "processed food", "trans fats"],
        },
    }
    return diet_plans.get(condition,diet_plans["normal"])

SYSTEM_PROMPT = """
You are a helpful medical and nutrition assistant.
For the input blood work image, extract the numbers and the normal range, then categorize
the condition as one of: normal, high_cholesterol, high_sugar.
Then call the appropriate tool to retrieve and present the diet plan.
"""

#agent created
from langgraph.checkpoint.memory import InMemorySaver
agent=create_agent(
    llm2,
    tools=[get_diet_plan],
    system_prompt=SYSTEM_PROMPT,
  #  checkpointer=InMemorySaver()
)

#agent is getting invoked from here
result = agent.invoke({
    "messages":[HumanMessage(content=[
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
        {"type": "text", "text": "Analyse this blood work report and suggest a diet plan."},
    ])]
})

print(result["messages"][-1].content)

#result is correct but formating is not good. need to check for correct formatting