from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate, MessagesPlaceholder

from longchain_demo.my_llm import llm

examples = [
    {'input':"2🦅2","output":"4"},
    {'input':"2🦅3","output":"5"}
]

base_prompt = ChatPromptTemplate.from_messages(
    [
        ('human','{input}'),
        ('ai','{output}'),
    ]
)

few_short_prompty= FewShotChatMessagePromptTemplate(
    examples=examples,
    example_prompt=base_prompt,
)

final_template = ChatPromptTemplate.from_messages(
    [
        ("system","你是一个智能机器人ai助手"),
        few_short_prompty,
        MessagesPlaceholder("msgs")
    ]
)

#chain = final_template | llm
chain  = final_template | llm | StrOutputParser()
print(chain.invoke({"msgs":[HumanMessage(content="讲讲马克思")]}))