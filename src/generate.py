from prompt_system import PromptSystem
from base import LLM
import asyncio
from IR import *
import json


# prompt_sys = PromptSystem.getSingleton() 
# static_prompt = prompt_sys.getPrompt('static')
# dynamic_prompt = prompt_sys.getPrompt('dynamic')
# objective_prompt = prompt_sys.getPrompt('objective')
# llm = LLM('qwen-max')

def ir_from_json(json_str: str) -> IR:
    """从JSON字符串生成IR实例"""
    # 1. 解析JSON为字典
    data: Dict[str, Any] = json.loads(json_str)
    
    # 2. 处理约束条件（将字典转为Expr对象）
    def parse_constraint(constraint_data: Optional[Dict[str, Any]]) -> Optional[Expr]:
        if constraint_data is None:
            return None
        return Expr.from_dict(constraint_data)
    
    # 3. 构建并返回IR实例
    return IR(
        start_date=data["start_date"],
        peoples=data["peoples"],
        travel_days=data["travel_days"],
        original_city=data["original_city"],
        destinate_city=data["destinate_city"],
        budgets=data.get("budgets", 0),  # 支持默认值
        attraction_constraints=parse_constraint(data.get("attraction_constraints")),
        accommodation_constraints=parse_constraint(data.get("accommodation_constraints")),
        restaurant_constraints=parse_constraint(data.get("restaurant_constraints")),
        transport_constraints=parse_constraint(data.get("transport_constraints"))
    )

json_str = """
{
  "start_date": "2025-06-10",
  "peoples": 2,
  "travel_days": 3,
  "original_city": "广州",
  "destinate_city": "西安",
  "budgets": 5000,
  "attraction_constraints": {
    "type": "op",
    "op": ">=",
    "left": {"type": "field", "field": "rating"},
    "right": {"type": "value", "value": 4.5}
  }
}
"""

ir = ir_from_json(json_str)
print(ir.budgets)