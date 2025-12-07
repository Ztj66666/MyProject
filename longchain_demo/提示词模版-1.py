from langchain_core.prompts import PromptTemplate

from longchain_demo.my_llm import llm
prompt = (
    PromptTemplate.from_template("帮我生成一个简短的，关于{topic}的介绍")
    +",要求:1.需要搞笑一点,"
    +"要求2：输出的内容才用{language}"
)

chain = prompt | llm

resp = chain.invoke({"topic":"相声","language":"英语"})
print(resp)

