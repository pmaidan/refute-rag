from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_fireworks import ChatFireworks
from langchain_deepseek import ChatDeepSeek
from langchain_huggingface import ChatHuggingFace
from langchain_aws import ChatBedrock
from langchain_litellm import ChatLiteLLM

def get_llm(model_type: str, model_name: str):
    match model_type:
        case "openai":
            return ChatOpenAI(model_name=model_name)
        case "anthropic":
            return ChatAnthropic(model=model_name)
        case "local":
            return ChatOllama(base_url="http://localhost:11434", model=model_name)
        case "google_genai":
            return ChatGoogleGenerativeAI(model=model_name)
        case "groq":
            return ChatGroq(model_type=model_name)
        case "fireworks":
            return ChatFireworks(model_type=model_name)
        case "deepseek":
            return ChatDeepSeek(model_type=model_name)
        case "huggingface":
            return ChatHuggingFace(model=model_name)
        case "aws":
            return ChatBedrock(model=model_name)
        case "litellm":
            return ChatLiteLLM(model=model_name)
        case _:
            raise ValueError(f"Unsupported model type: {model_type}")
        