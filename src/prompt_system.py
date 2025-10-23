from __future__ import annotations
from base import System

class PromptSystem(System):

    singleton: PromptSystem = None

    def __init__(self) -> None:
        self.systemName = 'PromptSystem'
        super().__init__(self.systemName)

    @System.enable_guard
    def getPrompt(self, _key : str):
        if _key is None or _key not in self.prompts:
            print(f'Prompt of {_key} not found')
            return ""
        return self.prompts[_key]

    @staticmethod
    def getSingleton() -> PromptSystem:
        if PromptSystem.singleton == None:
            PromptSystem.singleton = PromptSystem()
        return PromptSystem.singleton
    
    @property
    def prompts(self):
        return self.config.config

    
