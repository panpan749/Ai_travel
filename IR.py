from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union


class Expr:

    def eval(self, context: Dict[str, Any]) -> Any:
        raise NotImplementedError

    def to_dict(self) -> Dict[str, Any]:
        raise NotImplementedError

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Expr":
        node_type = data.get("type")
        if node_type == "value":
            return ValueNode(value=data["value"])
        if node_type == "field":
            return FieldNode(field=data["field"])
        if node_type == "op":
            return OpNode(
                op=data["op"],
                left=Expr.from_dict(data["left"]),
                right=Expr.from_dict(data["right"]),
            )
        if node_type == "unary":
            return UnaryOpNode(
                op=data["op"],
                operand=Expr.from_dict(data["operand"]),
            )
        if node_type == "arith":
            return ArithmeticOpNode(
                op=data["op"],
                left=Expr.from_dict(data["left"]),
                right=Expr.from_dict(data["right"]),
            )
        if node_type == "aggregate":
            return AggregateNode(
                func=data["func"],
                items=[Expr.from_dict(x) for x in data.get("items", [])],
            )
        raise ValueError(f"Unknown expr type: {node_type}")

@dataclass
class ArithmeticOpNode(Expr):
    op: str  # '+', '-', '*', '/'
    left: Expr
    right: Expr

    def eval(self, context: Dict[str, Any]) -> Any:
        lval = self.left.eval(context)
        rval = self.right.eval(context)
        if self.op == "+":
            return lval + rval
        if self.op == "-":
            return lval - rval
        if self.op == "*":
            return lval * rval
        if self.op == "/":
            return lval / rval
        raise ValueError(f"Unknown arithmetic op {self.op}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "arith",
            "op": self.op,
            "left": self.left.to_dict(),
            "right": self.right.to_dict(),
        }

@dataclass
class AggregateNode(Expr):
    """Aggregation over a list of items.

    func: 'sum' | 'min' | 'max' | 'count'
    items: list of expressions producing numbers/iterables depending on func
    """

    func: str
    items: List[Expr]

    def eval(self, context: Dict[str, Any]) -> Any:
        values = [item.eval(context) for item in self.items]
        if self.func == "sum":
            return sum(values)
        if self.func == "min":
            return min(values)
        if self.func == "max":
            return max(values)
        if self.func == "count":
            return len(values)
        raise ValueError(f"Unknown aggregate func {self.func}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "aggregate",
            "func": self.func,
            "items": [x.to_dict() for x in self.items],
        }

def _is_iterable(obj: Any) -> bool:
    if obj is None:
        return False
    try:
        iter(obj)
        return True
    except TypeError:
        return False


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

    num_travlers: int = None
    rooms_per_night: int = None
    change_hotel: bool = False

    ## 时间相关
    daily_total_time: Optional[Expr] = None
    daily_queue_time: Optional[Expr] = None
    daily_total_meal_time: Optional[Expr] = None

    daily_transportation_time: Optional[Expr] = None

    total_active_time: Optional[Expr] = None
    total_queue_time: Optional[Expr] = None
    total_resturant_time: Optional[Expr] = None
    total_transportation_time: Optional[Expr] = None
    ## POI相关
    num_attractions_per_day: Optional[Expr] = None
    meal_frequency: Optional[Expr] = None
    hotel_frequency: Optional[Expr] = None

    ## 交通相关
    infra_city_transportation: str = 'public_transportation' # or taxi or none

    ## 预算相关
    total_budget: Optional[Expr] = None
    total_meal_budget: Optional[Expr] = None
    total_attraction_ticket_budget: Optional[Expr] = None
    total_hotel_budget: Optional[Expr] = None
    total_transportation_budget: Optional[Expr] = None

    daily_total_budget: Optional[Expr] = None
    daily_total_meal_budget: Optional[Expr] = None
    daily_total_attraction_ticket_budget: Optional[Expr] = None
    daily_total_hotel_budget: Optional[Expr] = None
    daily_total_transportation_budget: Optional[Expr] = None

    extra: str = None


