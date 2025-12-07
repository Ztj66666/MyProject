from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from env_utils import GEMINI_API_KEY, GEMINI_BASE_URL

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.8,
    api_key=GEMINI_API_KEY,
    # 移除 base_url=GEMINI_BASE_URL,
)
message=[
    ('system','你是一个只能助手'),
    ('human',"请介绍一下什么是深度学习")
]

resp = llm.invoke(message)
print(resp)