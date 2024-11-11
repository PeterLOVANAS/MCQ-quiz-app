from openai import OpenAI
import json
import os
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List
import pandas as pd
load_dotenv()

class Question_llm(BaseModel):
  level: str
  question: str
  code: str
  explanation: str
  opt_1: str
  opt_2: str
  opt_3: str
  opt_4: str
  correct_opt: str


class MCQs_llm(BaseModel):
  questions: List[Question_llm]


class Agent:
  def __init__(self, template_file: str, model_name = 'gpt-4o-2024-08-06'):
    self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    self.model_name = model_name
    with open(template_file) as f:
      template = f.read()
    self.template = template
  
  def generate(self,n, topics, levels):
    response = self.client.beta.chat.completions.parse(
      model = self.model_name,
      messages=[
        {'role' : 'user' , 'content' : self.template.format(n, topics, levels)}
      ],
      response_format=MCQs_llm
    )

    raw_response = response.choices[0].message.content
    json_response = json.loads(raw_response)
    return json_response