from typing import Optional

from langchain_core.prompts import PromptTemplate
from pydantic import Field
from pydantic import BaseModel

from longchain_demo.my_llm import llm


class Joke(BaseModel):
    setup:str = Field(description="笑话的开头部分")
    punchline: str = Field(description="笑话的包袱")
    rating: Optional[int] = Field(description="笑话的有趣程度，从一到十")
prompt_template = PromptTemplate.from_template("帮我生成一个简短的，关于{topic}的笑话")
runnable = llm.with_structured_output(Joke)

chain = prompt_template | runnable
resp = chain.invoke({"topic":"狗"})
print(resp)


