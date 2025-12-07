from typing import Optional

from langchain_core.prompts import PromptTemplate
from pydantic import Field
from pydantic import BaseModel

from longchain_demo.my_llm import llm


class Joke(BaseModel):
    setup:str = Field(description="笑话的开头部分")
    punchline: str = Field(description="笑话的包袱")
    rating: Optional[int] = Field(description="笑话的有趣程度，从一到十")
runnable = llm.bind_tools([Joke])


resp = runnable.invoke("讲一个关于关羽的笑话")
print(resp.content)
resp.pretty_print()

