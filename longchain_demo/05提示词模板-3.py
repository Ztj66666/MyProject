from langchain_core.prompts import ChatPromptTemplate

from longchain_demo.my_llm import llm

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "你是一个幽默的电视台主持人！"),
    ("user", "帮我生成一个简短的，关于{topic}的报幕词。")
])
chain = prompt_template | llm
resp = chain.invoke({"topic": "相声"})


print(resp)