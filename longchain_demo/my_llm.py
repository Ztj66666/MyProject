from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from env_utils import GEMINI_API_KEY, OPENAI_AUDIO_KEY

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.8,
    api_key=GEMINI_API_KEY,
    model_kwargs={"response_format": {"type":"json_object"}},
    # 移除 base_url=GEMINI_BASE_URL,
)

