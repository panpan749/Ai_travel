from __future__ import annotations
from typing import Any, Callable, Optional
from dataclasses import dataclass

class Expr:
    def eval(self, context: dict) -> Any:
        raise NotImplementedError

@dataclass
class ValueNode(Expr):
    value: Any
    def eval(self, context: dict) -> Any:
        return self.value
    def to_dict(self): return {"type": "value", "value": self.value}

@dataclass
class FieldNode(Expr):
    field: str
    def eval(self, context: dict) -> Any:
        return context.get(self.field, None)
    def to_dict(self): return {"type": "field", "field": self.field}

@dataclass
class OpNode(Expr):
    op: str
    left: Expr
    right: Expr

    def eval(self, context: dict) -> Any:
        lval = self.left.eval(context)
        rval = self.right.eval(context)
        return self.apply_op(lval, rval)

    def apply_op(self, lval, rval):
        ops: dict[str, Callable[[Any, Any], bool]] = {
            "==": lambda a, b: a == b,
            "!=": lambda a, b: a != b,
            ">": lambda a, b: a > b,
            ">=": lambda a, b: a >= b,
            "<": lambda a, b: a < b,
            "<=": lambda a, b: a <= b,
            "include": lambda a, b: b in a if a else False,
            "intersect": lambda a, b: bool(set(a) & set(b)),
            "and": lambda a, b: a and b,
            "or": lambda a, b: a or b,
        }
        if self.op not in ops:
            raise ValueError(f"Unknown operator: {self.op}")
        return ops[self.op](lval, rval)

    def to_dict(self):
        return {"type": "op", "op": self.op, "left": self.left.to_dict(), "right": self.right.to_dict()}

@dataclass
class UnaryOpNode(Expr):
    op: str
    operand: Expr
    def eval(self, context: dict) -> Any:
        val = self.operand.eval(context)
        if self.op == "not":
            return not val
        raise ValueError(f"Unknown unary op {self.op}")
    def to_dict(self):
        return {"type": "unary", "op": self.op, "operand": self.operand.to_dict()}


@dataclass
class IR:
    start_date: str
    peoples: int
    travel_days: int
    original_city: str
    destinate_city: str

    attraction_constraints: Optional[Expr] = None
    accommodation_constraints: Optional[Expr] = None
    restaurant_constraints: Optional[Expr] = None
    transport_constraints: Optional[Expr] = None



@dataclass
class dynamic_constraint:

    num_travlers: int
    rooms_per_night: int

    ## 时间相关
    daily_total_time_max: int = 840
    daily_total_time_min: int = 1
    daily_queue_time_max: int

    daily_total_meal_time_max: int
    daily_total_meal_time_min: int = 1

    daily_transportation_time_max: int

    total_active_time_max: int
    total_active_time_min: int = 1
    total_queue_time_max: int
    total_transportation_time_max: int
    ## POI相关
    num_attractions_per_day: int = 1
    meal_frequency: int = 3
    hotel_frequency: int = 1

    ## 交通相关
    infra_city_transportation: str = 'public_transportation' # or taxi or none

    ## 预算相关
    total_budget: int = 1e12
    total_meal_budget: int = 1e12
    total_attraction_ticket_budget: int = 1e12
    total_hotel_budget: int = 1e12
    total_transportation_budget: int = 1e12

    daily_total_budget: int = 1e12
    daily_total_meal_budget: int = 1e12
    daily_total_attraction_ticket_budget: int = 1e12
    daily_total_hotel_budget: int = 1e12
    daily_total_transportation_budget: int = 1e12
