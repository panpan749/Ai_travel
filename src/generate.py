from prompt_system import PromptSystem
from base import LLM
import asyncio
from IR import *
from dataclasses import asdict
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

def dynamic_constraint_from_json(json_str: str) -> dynamic_constraint:
    """从JSON字符串生成dynamic_constraint实例"""
    # 1. 解析JSON为字典
    data: Dict[str, Any] = json.loads(json_str)
    
    # 2. 工具函数：将字典转为Expr对象（处理None情况）
    def parse_expr(expr_data: Optional[Dict[str, Any]]) -> Optional[Expr]:
        if expr_data is None:
            return None
        return Expr.from_dict(expr_data)
    
    # 3. 映射所有字段（基础字段直接取，Expr字段通过parse_expr转换）
    return dynamic_constraint(
        # 时间相关
        daily_total_time=parse_expr(data.get("daily_total_time")),
        daily_queue_time=parse_expr(data.get("daily_queue_time")),
        daily_total_meal_time=parse_expr(data.get("daily_total_meal_time")),
        daily_transportation_time=parse_expr(data.get("daily_transportation_time")),
        total_active_time=parse_expr(data.get("total_active_time")),
        total_queue_time=parse_expr(data.get("total_queue_time")),
        total_resturant_time=parse_expr(data.get("total_resturant_time")),  # 保持原始拼写
        total_transportation_time=parse_expr(data.get("total_transportation_time")),
        
        # POI相关
        num_attractions_per_day=parse_expr(data.get("num_attractions_per_day")),
        num_restaurants_per_day=parse_expr(data.get("num_restaurants_per_day")),
        num_hotels_per_day=parse_expr(data.get("num_hotels_per_day")),
        
        # 交通相关
        infra_city_transportation=data.get("infra_city_transportation", "none"),  # 使用默认值
        
        # 预算相关
        total_budget=parse_expr(data.get("total_budget")),
        total_meal_budget=parse_expr(data.get("total_meal_budget")),
        total_attraction_ticket_budget=parse_expr(data.get("total_attraction_ticket_budget")),
        total_hotel_budget=parse_expr(data.get("total_hotel_budget")),
        total_transportation_budget=parse_expr(data.get("total_transportation_budget")),
        daily_total_budget=parse_expr(data.get("daily_total_budget")),
        daily_total_meal_budget=parse_expr(data.get("daily_total_meal_budget")),
        daily_total_attraction_ticket_budget=parse_expr(data.get("daily_total_attraction_ticket_budget")),
        daily_total_hotel_budget=parse_expr(data.get("daily_total_hotel_budget")),
        daily_total_transportation_budget=parse_expr(data.get("daily_total_transportation_budget")),
        
        # 额外信息
        extra=data.get("extra")
    )

def ir_to_json(ir: IR) -> str:
    """将IR实例序列化为JSON字符串"""
    # 先将IR转为字典，约束字段通过to_dict()序列化
    ir_dict = asdict(ir)
    # 处理约束字段的序列化
    return json.dumps(ir_dict, ensure_ascii=False, indent=2)
def dynamic_constraint_to_dict(dc: "dynamic_constraint") -> Dict[str, Any]:
    """将 dynamic_constraint 实例转为字典（替代 dataclasses.asdict()）"""
    # 明确列出 dynamic_constraint 的所有字段（需与类定义完全一致）
    fields = [
        # 时间相关
        "daily_total_time", "daily_queue_time", "daily_total_meal_time", "daily_transportation_time",
        "total_active_time", "total_queue_time", "total_resturant_time", "total_transportation_time",
        # POI 相关
        "num_attractions_per_day", "num_restaurants_per_day", "num_hotels_per_day",
        # 交通相关
        "infra_city_transportation",
        # 预算相关
        "total_budget", "total_meal_budget", "total_attraction_ticket_budget", "total_hotel_budget",
        "total_transportation_budget", "daily_total_budget", "daily_total_meal_budget",
        "daily_total_attraction_ticket_budget", "daily_total_hotel_budget", "daily_total_transportation_budget",
        # 额外信息
        "extra"
    ]
    
    result = {}
    for field in fields:
        # 获取字段值
        value = getattr(dc, field, None)
        # 如果是 Expr 类型，调用 to_dict() 序列化
        if isinstance(value, Expr):
            result[field] = value.to_dict()
        else:
            result[field] = value
            
    return json.dumps(result, ensure_ascii=False, indent=2)

json_str = """
{
  "daily_total_time": {
    "type": "op",
    "op": "<=",
    "left": {"type": "field", "field": "daily_total_time"},
    "right": {"type": "value", "value": 900}
  },
  "num_attractions_per_day": {
    "type": "op",
    "op": "==",
    "left": {"type": "field", "field": "num_attractions_per_day"},
    "right": {"type": "value", "value": 2}
  },
  "infra_city_transportation": "public_transportation",
  "total_budget": {
    "type": "op",
    "op": "<=",
    "left": {"type": "field", "field": "total_budget"},
    "right": {"type": "value", "value": 8000}
  },
  "extra": "亲子旅行，优先选择无障碍设施完善的景点"
}
"""

# ir = ir_from_json(json_str)
# print(ir_to_json(ir))

dc = dynamic_constraint_from_json(json_str)
print(dynamic_constraint_to_dict(dc))