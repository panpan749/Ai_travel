from prompt_system import PromptSystem
from base import LLM
import asyncio

llm = LLM('gpt-3.5-turbo')
resp = llm.invoke('你好')
# prompt_sys = PromptSystem.getSingleton() 

print(resp)
