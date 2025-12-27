from typing import Protocol
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace


class AgentProtocol(Protocol):
    def __init__(self, model: str, max_tokens: int, temperature: float):
        ...
        
        
class OpenAIAgents:
    def __init__(self, model: str = "gpt-5.1", max_tokens: int = 1024, temperature: float = 0.1):
        from langchain_openai import ChatOpenAI
        self.agent = ChatOpenAI(model=model, max_tokens=max_tokens, temperature=temperature)
        
           
class GoogleGenerativeAIAgents:
    def __init__(self, model: str = "gemini-2.0-flash", max_tokens: int = 1024, temperature: float = 0.2):
        from langchain_google_genai import ChatGoogleGenerativeAI
        self.agent = ChatGoogleGenerativeAI(model=model, max_tokens=max_tokens, temperature=temperature)


class AnthropicAgents:
    def __init__(self, model: str = "claude-3", max_tokens: int = 1024, temperature: float = 0.2):
        from langchain_anthropic import ChatAnthropic
        self.agent = ChatAnthropic(model=model, max_tokens=max_tokens, temperature=temperature)


# class HuggingFaceAgents:
    # def __init__(self, model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct", max_tokens: int = 1024, temperature: float = 0.2):
    #     from langchain_huggingface import ChatHuggingFace
    #     self.agent = ChatHuggingFace(model=model, max_tokens=max_tokens, temperature=temperature)
    

class HuggingFaceAgents:
    def __init__(self, model: str = "microsoft/Phi-3-mini-4k-instruct"):
        pipe = pipeline(
            "text-generation",
            model=model,
            max_new_tokens=512,
        )

        llm = HuggingFacePipeline(pipeline=pipe)
        self.agent = ChatHuggingFace(llm=llm)



      
def get_agent(agent_type: str) -> AgentProtocol:
    agents = {
        "openai": OpenAIAgents,
        "google": GoogleGenerativeAIAgents,
        "anthropic": AnthropicAgents,
        "huggingface": HuggingFaceAgents,
    }
    
    agent_class = agents.get(agent_type.lower())
    if not agent_class:
        raise ValueError(f"Unsupported agent type: {agent_type}")
    
    return agent_class().agent
