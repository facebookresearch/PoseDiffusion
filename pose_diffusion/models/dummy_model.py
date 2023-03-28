import torch
from typing import Dict, List, Optional, Union


class DummyModel:
    def __init__(
        self,
        name: str,
        backbone: Dict,
        transformer: Dict,
    ):
        self.name = name
        
        print(f"build a dummy model called {self.get_name()}")
        
    def get_name(self) -> str:
        return self.name


