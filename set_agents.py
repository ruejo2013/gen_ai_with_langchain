from pyexpat import model
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


class DallEAgents:
    def __init__(self, model: str = "dall-e-3", size: str = "1024x1024", temperature: float = 0.2):
        from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper as DallE
        self.agent = DallE(
            model=model, 
            size=size, 
            quality="standard",
            n=1
        )
    

class HuggingFaceAgents:
    def __init__(self, model: str = "microsoft/Phi-3-mini-4k-instruct"):
        pipe = pipeline(
            "text-generation",
            model=model,
            max_new_tokens=512,
        )

        llm = HuggingFacePipeline(pipeline=pipe)
        self.agent = ChatHuggingFace(llm=llm)

class ReplicateAgents:
    def __init__(
        self,
        model: str = "stability-ai/stable-diffusion-3.5-large",
    ):
        from langchain_community.llms.replicate import Replicate

        self.agent = Replicate(
            model=model,
            model_kwargs={
                "prompt_strength": 0.85,
                "cfg": 4.5,
                "steps": 40,
                "aspect_ratio": "1:1",
                "output_format": "webp",
                "output_quality": 90,
            }
        )
        
        
class GoogleVertextAIAgents:
    # this agent uses the Application Default Credentials (ADC) set in your environment to Authenticate
    def __init__(self, model: str = "gemini-2.0-flash", max_tokens: int = 1024, temperature: float = 0.2):
        from langchain_google_genai import ChatGoogleGenerativeAI
        self.agent= ChatGoogleGenerativeAI(
            model=model,
            project="gen-lang-client-0621997816",
            temperature=0,
        )
    
    
class OllamaAgents:
    def __init__(self, model="deepseek-r1:1.5b", temperature=0):
        from langchain_ollama import ChatOllama
        self.agent = ChatOllama(model=model, temperature=temperature)

      
def get_agent(agent_type: str) -> AgentProtocol:
    agents = {
        "openai": OpenAIAgents,
        "google": GoogleGenerativeAIAgents,
        "anthropic": AnthropicAgents,
        "huggingface": HuggingFaceAgents,
        "ollama": OllamaAgents,
        "dalle": DallEAgents,
        "replicate": ReplicateAgents,
        "googleadc": GoogleVertextAIAgents,
    }
    
    agent_class = agents.get(agent_type.lower())
    if not agent_class:
        raise ValueError(f"Unsupported agent type: {agent_type}")
    
    return agent_class().agent
