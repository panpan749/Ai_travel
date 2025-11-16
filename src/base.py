from __future__ import annotations
import yaml
from pathlib import Path
import asyncio
from openai import AsyncOpenAI

class Colors:
    RESET = "\033[0m"
    RED = "\033[31m"
    YELLOW = "\033[33m"
    GREEN = "\033[32m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

class Config:
    
    configDirPath = Path(__file__).parent.joinpath('configs')
    file_path : str
    config : dict
    global_config : Config = None
    def __init__(self, file_name : str) -> None:
        self.file_path = self.configDirPath.joinpath(file_name)
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
        except Exception as e:
            self.config = {}
            print(f'{Colors.YELLOW}Warning{Colors.RESET}: {self.file_path} not found')

    def get_global_config() -> Config:
        if Config.global_config is None:
            Config.global_config = Config('global.yaml')
        return Config.global_config

class System:
    systemName : str
    config: Config
    enable: bool
    def __init__(self, _systemName : str) -> None:
        self.systemName = _systemName
        self.config = Config(_systemName + '.yaml')
        self.enable = self.config.config.get('enable', True)

    def getConfig(self, _key : str):
        if _key not in self.config.config:
            print(f'{Colors.YELLOW}Warning{Colors.RESET}: {_key} not in {self.systemName} config')
            return None
        return self.config.config[_key]
    
    def setConfig(self, _key : str, _value : str):
        self.config.config[_key] = _value
    
    def enable_guard(func):
        def wrapper(self, *args, **kwargs):
            if not getattr(self, 'enable', False):
                return None
            return func(self, *args, **kwargs)
        async def wrapper_async(self, *args, **kwargs):
            if not getattr(self, 'enable', False):
                return None
            return await func(self, *args, **kwargs)  
        
        if asyncio.iscoroutinefunction(func):
            return wrapper_async
        return wrapper

class LLM:
    
    model_name : str
    system_prompt: str

    def __init__(self, model_name : str, system_prompt : str = None) -> None:
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.api_url = Config.get_global_config().config['api_url']
        self.api_key = Config.get_global_config().config['api_key']
    
    async def ask_llm(self, prompt):
        client = AsyncOpenAI(api_key=self.api_key,base_url=self.api_url,timeout=300)       
        messages = [
            {"role": "system", "content": self.system_prompt},
        ] if self.system_prompt else []

        messages.append({"role": "user", "content": prompt})

        model_name = self.model_name
        completion = await client.chat.completions.create(model=model_name, messages=messages,max_tokens = 8192) 
        
        return completion.choices[0].message.content
    async def invoke(self, prompt : str):
        while True:
            try:
                resp = await self.ask_llm(prompt)
                return resp
            except Exception as e:
                print(f'发生报错 {e},重试中')