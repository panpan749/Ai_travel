from __future__ import annotations
from dataclasses import dataclass,field
from typing import Any, Callable, Dict, Optional,List



class Expr:
    """表达式抽象基类，用于构建旅行规划约束的抽象语法树(AST)。
    
    所有具体的表达式节点都需要实现 eval() 和 to_dict() 方法。
    eval() 用于在给定上下文中计算表达式的值，to_dict() 用于序列化。
    """
    
    def eval(self, context: Dict[str, Any]) -> Any:
        """在给定上下文中计算表达式的值。
        
        Args:
            context: 包含变量和数据的上下文字典，如 {"budget": 5000, "rating": 4.5}
            
        Returns:
            计算后的表达式值
        """
        raise NotImplementedError

    def to_dict(self) -> Dict[str, Any]:
        """将表达式序列化为字典格式，便于存储和传输。
        
        Returns:
            包含表达式类型和参数的字典
        """
        raise NotImplementedError

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> Expr:
        """从字典反序列化表达式对象，支持多态创建。
        
        Args:
            data: 包含表达式信息的字典，必须包含 "type" 字段
            
        Returns:
            对应类型的表达式对象
            
        Raises:
            ValueError: 当遇到未知的表达式类型时
        """
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
                return_field=data["return_field"],
                field=data["field"],
                filter=Expr.from_dict(data["filter"]) if data.get("filter") else None
            )            
        raise ValueError(f"Unknown expr type: {node_type}")

@dataclass
class ArithmeticOpNode(Expr):
    """算术运算表达式节点，支持基本的四则运算。
    
    用于构建如 "budget * 0.8" 或 "rating + 0.5" 这样的算术表达式。
    """
    op: str  # 运算符：'+', '-', '*', '/'
    left: Expr  # 左操作数
    right: Expr  # 右操作数

    def eval(self, context: Dict[str, Any]) -> Any:
        """计算算术表达式的值。
        
        Args:
            context: 包含变量的上下文字典
            
        Returns:
            算术运算的结果
            
        Raises:
            ValueError: 当遇到未知的算术运算符时
        """
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
        """序列化为字典格式。"""
        return {
            "type": "arith",
            "op": self.op,
            "left": self.left.to_dict(),
            "right": self.right.to_dict(),
        }

@dataclass
class AggregateNode(Expr):
    """聚合函数表达式节点，对一组值进行聚合计算。

    支持 sum、min、max、count 等聚合操作，常用于计算总费用、最高评分等。
    
    Attributes:
        func: 聚合函数类型，支持 'sum' | 'min' | 'max' | 'count'
        items: 参与聚合计算的表达式列表
    """

    func: str
    field: str ##list 字段提取
    return_field: str ##返回字段，仅在min和max中生效
    filter: Optional[Expr] = None

    def eval(self, context: Dict[str, Any]) -> Any:
        """计算聚合表达式的值。
        
        Args:
            context: 包含变量的上下文字典
            
        Returns:
            聚合计算的结果
            
        Raises:
            ValueError: 当遇到未知的聚合函数时
        """
        list_context = context['global'] or []
        if self.func != 'sum':
            values = [
                item
                for item in list_context
                if self.filter is None or self.filter.eval(item)
            ]
        else:
            values = [
                item[self.field]
                for item in list_context
                if self.filter is None or self.filter.eval(item)
            ]
        
        if self.func == "sum":
            if values == []: return 0
            return sum(values)
        if self.func == "min":
            if values == []: return []
            min_item = min(values, key=lambda x: x[self.field])
            if self.return_field == '*':
                return [item for item in values if item[self.field] == min_item[self.field]]
            elif isinstance(self.return_field,str):
                return [item[self.return_field] for item in values if item[self.field] == min_item[self.field]]
        if self.func == "max":
            if values == []: return []
            max_item = max(values, key=lambda x: x[self.field])
            if self.return_field == '*':
                return [item for item in values if item[self.field] == max_item[self.field]]
            elif isinstance(self.return_field,str):
                return [item[self.return_field] for item in values if item[self.field] == max_item[self.field]]
        if self.func == "count":
            if values == []: return 0
            return len(values)
        raise ValueError(f"Unknown aggregate func {self.func}")

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典格式。"""
        return {
            "type": "aggregate",
            "func": self.func,
            'filter': self.filter.to_dict() if self.filter else None,
            'field': self.field,
            'return_field': self.return_field,
        }



@dataclass
class ValueNode(Expr):
    """字面量值节点，表示常量或固定值。
    
    用于表示如 5000、4.5、"hotel" 这样的固定值。
    """
    value: Any  # 字面量值
    
    def eval(self, context: dict) -> Any:
        """直接返回存储的值，不依赖上下文。"""
        return self.value
    
    def to_dict(self): 
        return {"type": "value", "value": self.value}

@dataclass
class FieldNode(Expr):
    """字段访问节点，从上下文中获取指定字段的值。
    
    用于访问如 budget、rating、hotel_name 等上下文变量。
    """
    field: str  # 字段名
    
    def eval(self, context: dict) -> Any:
        """从上下文中获取字段值，如果不存在则返回None。"""
        return context.get(self.field, None)
    
    def to_dict(self): 
        return {"type": "field", "field": self.field}

@dataclass
class OpNode(Expr):
    """二元操作符节点，支持比较、逻辑、集合等操作。
    
    用于构建如 "rating >= 4.5"、"budget <= 5000"、"city in ['北京', '上海']" 等条件表达式。
    """
    op: str  # 操作符
    left: Expr  # 左操作数
    right: Expr  # 右操作数

    def eval(self, context: dict) -> Any:
        """计算二元操作的结果。
        
        Args:
            context: 包含变量的上下文字典
            
        Returns:
            操作结果，通常是布尔值
        """
        lval = self.left.eval(context)
        rval = self.right.eval(context)
        return self.apply_op(lval, rval)

    def apply_op(self, lval, rval):
        """应用具体的操作符逻辑。
        
        支持的操作符：
        - 比较操作：==, !=, >, >=, <, <=
        - 集合操作：include (包含), intersect (交集)
        - 逻辑操作：and, or
        
        Args:
            lval: 左操作数的值
            rval: 右操作数的值
            
        Returns:
            操作结果
            
        Raises:
            ValueError: 当遇到未知的操作符时
        """
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
        """序列化为字典格式。"""
        return {"type": "op", "op": self.op, "left": self.left.to_dict(), "right": self.right.to_dict()}

@dataclass
class UnaryOpNode(Expr):
    """一元操作符节点，支持逻辑非等单操作数运算。
    
    用于构建如 "not (rating < 3.0)" 这样的逻辑表达式。
    """
    op: str  # 一元操作符
    operand: Expr  # 操作数
    
    def eval(self, context: dict) -> Any:
        """计算一元操作的结果。
        
        Args:
            context: 包含变量的上下文字典
            
        Returns:
            操作结果
            
        Raises:
            ValueError: 当遇到未知的一元操作符时
        """
        val = self.operand.eval(context)
        if self.op == "not":
            return not val
        raise ValueError(f"Unknown unary op {self.op}")
    
    def to_dict(self):
        """序列化为字典格式。"""
        return {"type": "unary", "op": self.op, "operand": self.operand.to_dict()}

@dataclass
class stage:

    origin_city: str
    destinate_city: str
    travel_days: int
    attraction_constraints: Optional[Expr] = None  # 景点约束
    accommodation_constraints: Optional[Expr] = None  # 住宿约束
    restaurant_constraints: Optional[Expr] = None  # 餐厅约束


@dataclass
class IR:
    """旅行规划问题的中间表示(Intermediate Representation)。
    
    这是整个系统的核心数据结构，用于表示一个完整的旅行规划请求。
    包含了基本的旅行信息以及各个类别的约束条件。
    
    Attributes:
        start_date: 旅行开始日期，格式如 "2025年6月10日"
        peoples: 旅行人数
        travel_days: 旅行天数
        original_city: 出发城市
        destinate_city: 目的地城市（注意：此处有拼写错误，应为destination_city）
        budgets: 总预算（单位：元）
        attraction_constraints: 景点选择约束表达式
        accommodation_constraints: 住宿选择约束表达式
        restaurant_constraints: 餐厅选择约束表达式
        transport_constraints: 交通选择约束表达式
    """
    start_date: str  # 旅行开始日期
    peoples: int  # 旅行人数
    total_travel_days: int  # 旅行天数
    stages: List[stage] #多阶段旅行

    budgets: int = 0  # 总预算 默认不设总预算
    children_num: int  = 0
    departure_transport_constraints: Optional[Expr] = None  # 交通约束
    intermediate_transport_constraints: Optional[Expr] = None #中转约束
    back_transport_constraints: Optional[Expr] = None



@dataclass
class dynamic_constraint:
    """动态约束类，用于表示旅行规划中的各种动态约束条件。
    
    这些约束条件会根据具体的旅行需求动态调整，包括时间、预算、选择频率等。
    使用表达式树(Expr)来表示复杂的约束逻辑，支持运行时计算。
    """
    num_travlers: int = None
    rooms_per_night: int = None
    children_num: int = 0
    multi_stage: bool = False
    ## 时间相关
    daily_total_time: Optional[Expr] = field(default_factory= lambda: OpNode('<=',FieldNode('daily_total_time'),ValueNode(840)))
    daily_queue_time: Optional[Expr] = None
    daily_total_restaurant_time: Optional[Expr] = None

    daily_transportation_time: Optional[Expr] = None

    total_active_time: Optional[Expr] = None
    total_queue_time: Optional[Expr] = None
    total_restaurant_time: Optional[Expr] = None
    total_transportation_time: Optional[Expr] = None
    ## 交通相关
    infra_city_transportation: dict = None # 'public_transportation' or 'taxi' or 'none'

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

    # 额外信息
    extra: str = None  # 其他约束或备注信息