from dotenv import load_dotenv
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

#reading the text file, w for writing 
with open("2-health_analyser/blood_work.txt","r") as f:
    blood_report = f.read()

#200 is a character limit, this prints first 200 character from report
print(blood_report[:200])  

prompt = f"""
You are a medical data extraction assistant.

From the blood report below, extract ALL test values and classify each one as HIGH, LOW, or NORMAL
based on the reference ranges provided in the report.

Format your response as below format in JSON format:
- Test Name: value | Status: HIGH/LOW/NORMAL | Reference: range
Blood Report:{blood_report}
"""

llm = ChatGoogleGenerativeAI(model = "gemma-4-31b-it")

response = llm.invoke(prompt)
value= response.text
print(f"this is health data \n {value} \n this is diet plan")

diet_prompt="""
You are a clinical nutritionist specializing in Indian dietary habits.
    
Based on the blood work analysis below, write:
1. A short health summary in 3 lines explaining the patient's condition in simple language
2. A short, practical Indian diet plan having only two sections (1) Foods to avoid (2) Foods to eat more of.
Do not include any other sections in diet plan.

Blood Work Analysis:{value}
"""
diet_response = llm.invoke(diet_prompt)

print(f"this is diet plan \n {diet_response.text}")