from __future__ import annotations
import pyomo.environ as pyo 
from pyomo.gdp import Disjunct, Disjunction
from pyomo.core import TransformationFactory
import requests
from datetime import timedelta, datetime
from dataclasses import dataclass,field,asdict
from typing import Any, Callable, Dict, Optional,List
import json

_ConstraintClass = pyo.Constraint

def _coerce_bool_to_constraint(val):
    if val is True:
        return _ConstraintClass.Feasible
    if val is False:
        return _ConstraintClass.Infeasible
    return val


def Constraint(*args, **kwargs):
    # 包 rule
    if 'rule' in kwargs and kwargs['rule'] is not None:
        _rule = kwargs['rule']
        def _wrapped_rule(m, *idx):
            try:
                out = _rule(m)
            except TypeError:
                out = _rule(m, *idx)
            return _coerce_bool_to_constraint(out)
        kwargs['rule'] = _wrapped_rule
    # 包 expr（直接给了 True/False 的情况）
    if 'expr' in kwargs and isinstance(kwargs['expr'], bool):
        kwargs['expr'] = _coerce_bool_to_constraint(kwargs['expr'])
    return _ConstraintClass(*args, **kwargs)


Constraint.Skip = _ConstraintClass.Skip
Constraint.Feasible = _ConstraintClass.Feasible
Constraint.Infeasible = _ConstraintClass.Infeasible

pyo.Constraint = Constraint

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
    peoples_per_car: int = 4
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

def ir_from_json(json_str: str) -> IR:
    """从JSON字符串生成IR实例"""
    # 1. 解析JSON为字典
    data: Dict[str, Any] = json.loads(json_str)
    
    # 2. 处理约束条件（将字典转为Expr对象）
    def parse_constraint(constraint_data: Optional[Dict[str, Any]]) -> Optional[Expr]:
        if constraint_data is None:
            return None
        return Expr.from_dict(constraint_data)
    
    def parse_stage(stage_data: Dict[str, Any]) -> stage:
        return stage(
            origin_city=stage_data["original_city"],
            destinate_city=stage_data["destinate_city"],
            travel_days=stage_data["travel_days"],
            attraction_constraints=parse_constraint(stage_data.get("attraction_constraints")),
            accommodation_constraints=parse_constraint(stage_data.get("accommodation_constraints")),
            restaurant_constraints=parse_constraint(stage_data.get("restaurant_constraints")),
        )
    
    # 3. 构建并返回IR实例
    return IR(
        start_date=data["start_date"],
        peoples=data.get("peoples", 1),
        total_travel_days=data["total_travel_days"],
        budgets=data.get("budgets", 0),  # 支持默认值
        children_num=data.get("children_num", 0),
        stages=[parse_stage(stage_data) for stage_data in data["stages"]],
        departure_transport_constraints=parse_constraint(data.get("departure_transport_constraints")),
        back_transport_constraints=parse_constraint(data.get("back_transport_constraints")),
        intermediate_transport_constraints=parse_constraint(data.get("intermediate_transport_constraints")),
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
    num_travlers=data.get("num_travlers",1)
    return dynamic_constraint(
        # 基础字段
        num_travlers=data.get("num_travlers",1),
        children_num=data.get("children_num", 0),
        rooms_per_night=data.get("rooms_per_night", num_travlers // 2),
        peoples_per_car = data.get("peoples_per_car",4),
        multi_stage=data.get("multi_stage",False),
        # 时间相关
        daily_total_time=parse_expr(data.get("daily_total_time")),
        daily_queue_time=parse_expr(data.get("daily_queue_time")),
        daily_total_restaurant_time=parse_expr(data.get("daily_total_restaurant_time")),
        daily_transportation_time=parse_expr(data.get("daily_transportation_time")),
        total_active_time=parse_expr(data.get("total_active_time")),
        total_queue_time=parse_expr(data.get("total_queue_time")),
        total_restaurant_time=parse_expr(data.get("total_restaurant_time")),  # 保持原始拼写
        total_transportation_time=parse_expr(data.get("total_transportation_time")),

        
        # 交通相关
        infra_city_transportation=data.get("infra_city_transportation", None),  # 使用默认值
        
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
        extra=data.get("extra") or ""
    )


def fetch_data(ir: IR):
    url = "http://localhost:12457"

    for i in range(len(ir.stages)):
        if '市' not in ir.stages[i].origin_city:
            ir.stages[i].origin_city = ir.stages[i].origin_city + '市'
        if '市' not in ir.stages[i].destinate_city:
            ir.stages[i].destinate_city = ir.stages[i].destinate_city + '市'

    city_to_day_range = {}
    day_to_city = {}
    curr_day = 1
    for stg in ir.stages:
        city_to_day_range[stg.destinate_city] = [curr_day, curr_day + stg.travel_days - 1]
        for day in range(curr_day,curr_day + stg.travel_days):
            day_to_city[day] = stg.destinate_city
        curr_day += stg.travel_days

    cross_city_train_departure = []
    cross_city_train_back = []
    cross_city_train_transfer = []

    origin_city = ir.stages[0].origin_city
    if len(ir.stages) > 1:
        transfer_city = ir.stages[0].destinate_city
        destination_city = ir.stages[1].destinate_city
        cross_city_train_departure = requests.get(
        url + f"/cross-city-transport?origin_city={origin_city}&destination_city={transfer_city}").json()
        cross_city_train_transfer = requests.get(
            url + f"/cross-city-transport?origin_city={transfer_city}&destination_city={destination_city}").json()
    else:
        destination_city = ir.stages[0].destinate_city
        cross_city_train_departure = requests.get(
        url + f"/cross-city-transport?origin_city={origin_city}&destination_city={destination_city}").json()

    cross_city_train_back = requests.get(
        url + f"/cross-city-transport?origin_city={destination_city}&destination_city={origin_city}").json()

    attraction_list = []
    accommodation_list = []   
    restaurant_list = [] 
    infra_trans = {}
    for stg in ir.stages:
        origin_city = stg.origin_city
        destination_city = stg.destinate_city 
        max_retry = 3
        while max_retry > 0:
            try:
                attrs = requests.get(url + f"/attractions/{destination_city}").json()
                accos = requests.get(url + f"/accommodations/{destination_city}").json()
                rests = requests.get(url + f"/restaurants/{destination_city}").json()
                intra_city_trans = requests.get(url + f"/intra-city-transport/{destination_city}").json()
                for item in attrs:
                    item['start_stage'] = city_to_day_range[destination_city][0]
                    item['end_stage'] = city_to_day_range[destination_city][1]
                for item in accos:
                    item['start_stage'] = city_to_day_range[destination_city][0]
                    item['end_stage'] = city_to_day_range[destination_city][1]
                for item in rests:
                    item['start_stage'] = city_to_day_range[destination_city][0]
                    item['end_stage'] = city_to_day_range[destination_city][1]
                
                attraction_list.append(attrs)
                accommodation_list.append(accos)
                restaurant_list.append(rests)
                infra_trans = infra_trans | intra_city_trans
                break
            except:
                max_retry -= 1

    return cross_city_train_departure,cross_city_train_transfer,cross_city_train_back,{'attractions':attraction_list,'accommodations':accommodation_list,'restaurants':restaurant_list},infra_trans

def rough_rank(cross_city_train_departure:list[dict],cross_city_train_transfer,cross_city_train_back,poi_data,ir:IR):
    def create_context(item,key,value):
        return {**item,key:value}
    
    if ir.departure_transport_constraints:
        if isinstance(ir.departure_transport_constraints,OpNode) or isinstance(ir.departure_transport_constraints, UnaryOpNode):
            cross_city_train_departure = [item for item in cross_city_train_departure if ir.departure_transport_constraints.eval(create_context(item,'global',cross_city_train_departure))]
        elif isinstance(ir.departure_transport_constraints, AggregateNode) and (ir.departure_transport_constraints.func == 'min' or ir.departure_transport_constraints.func == 'max'):
            cross_city_train_departure = ir.departure_transport_constraints.eval({'global':cross_city_train_departure})
  
    if ir.back_transport_constraints:
        if isinstance(ir.back_transport_constraints,OpNode) or isinstance(ir.back_transport_constraints, UnaryOpNode):
            cross_city_train_back = [item for item in cross_city_train_back if ir.back_transport_constraints.eval(create_context(item,'global',cross_city_train_back))]
        elif isinstance(ir.back_transport_constraints, AggregateNode) and (ir.back_transport_constraints.func == 'min' or ir.back_transport_constraints.func == 'max'):
            cross_city_train_back = ir.back_transport_constraints.eval({'global':cross_city_train_back})

    if ir.intermediate_transport_constraints:
        if isinstance(ir.intermediate_transport_constraints,OpNode) or isinstance(ir.intermediate_transport_constraints, UnaryOpNode):
            cross_city_train_transfer = [item for item in cross_city_train_transfer if ir.intermediate_transport_constraints.eval(create_context(item,'global',cross_city_train_transfer))]
        elif isinstance(ir.intermediate_transport_constraints, AggregateNode) and (ir.intermediate_transport_constraints.func == 'min' or ir.intermediate_transport_constraints.func == 'max'):
            cross_city_train_transfer = ir.intermediate_transport_constraints.eval({'global':cross_city_train_transfer})

    attraction_list = []
    accommodation_list = []
    restaurant_list = []

    for idx,stg in enumerate(ir.stages):
        src_attrs = poi_data['attractions'][idx]
        dest_attrs = src_attrs
        if stg.attraction_constraints:
            if isinstance(stg.attraction_constraints,OpNode) or isinstance(stg.attraction_constraints, UnaryOpNode):
                dest_attrs = [item for item in src_attrs if stg.attraction_constraints.eval(create_context(item,'global',src_attrs))]
            elif isinstance(stg.attraction_constraints, AggregateNode) and (stg.attraction_constraints.func == 'min' or stg.attraction_constraints.func == 'max'):
                dest_attrs = stg.attraction_constraints.eval({'global':src_attrs})
        attraction_list.extend(dest_attrs)
        src_rests = poi_data['restaurants'][idx]
        dest_rests = src_rests
        if stg.restaurant_constraints:
            if isinstance(stg.restaurant_constraints,OpNode) or isinstance(stg.restaurant_constraints, UnaryOpNode):
                dest_rests = [item for item in src_rests if stg.restaurant_constraints.eval(create_context(item,'global',src_rests))]
            elif isinstance(stg.restaurant_constraints, AggregateNode) and (stg.restaurant_constraints.func == 'min' or stg.restaurant_constraints.func == 'max'):
                dest_rests = stg.restaurant_constraints.eval({'global':src_rests})
        restaurant_list.extend(dest_rests)
        src_accos = poi_data['accommodations'][idx]
        dest_accos = src_accos
        if stg.accommodation_constraints:
            if isinstance(stg.accommodation_constraints,OpNode) or isinstance(stg.accommodation_constraints, UnaryOpNode):
                dest_accos = [item for item in src_accos if stg.accommodation_constraints.eval(create_context(item,'global',src_accos))]
            elif isinstance(stg.accommodation_constraints, AggregateNode) and (stg.accommodation_constraints.func == 'min' or stg.accommodation_constraints.func == 'max'):
                dest_accos = stg.accommodation_constraints.eval({'global':src_accos})
        accommodation_list.extend(dest_accos)

    attraction_dict = {a['id']: a for a in attraction_list}
    hotel_dict = {h['id']: h for h in accommodation_list}
    restaurant_dict = {r['id']: r for r in restaurant_list}
    train_departure_dict = {t['train_number']: t for t in cross_city_train_departure}
    train_back_dict = {t['train_number']: t for t in cross_city_train_back}
    train_transfer_dict = {t['train_number']: t for t in cross_city_train_transfer}
    pois = {'attractions': attraction_dict,'accommodations': hotel_dict,'restaurants': restaurant_dict}

    return train_departure_dict,train_transfer_dict,train_back_dict, pois

def get_trans_params(intra_city_trans, hotel_id, attr_id, param_type):
    for key in [f"{hotel_id},{attr_id}", f"{attr_id},{hotel_id}"]:
        if key in intra_city_trans:
            data = intra_city_trans[key]
            return {
                'taxi_duration': float(data.get('taxi_duration')),
                'taxi_cost': float(data.get('taxi_cost')),
                'bus_duration': float(data.get('bus_duration')),
                'bus_cost': float(data.get('bus_cost'))
            }[param_type]
        
    return 1000000 #不可达的地点距离无穷大
    

class template:

    model: pyo.Model
    ir: IR
    cfg: dynamic_constraint
    cross_city_train_departure: dict
    cross_city_train_back: dict
    cross_city_train_transfer: dict
    poi_data: dict
    intra_city_trans: dict
    use_disj: bool
    city_to_day_range: dict
    day_to_city: dict
    _cid: int ##用于标识
    def __init__(self,cross_city_train_departure, cross_city_train_transfer, cross_city_train_back ,poi_data,intra_city_trans,ir,model = None):
        if not model:
            self.model = pyo.ConcreteModel()
        else: self.model = model

        self.cross_city_train_departure = cross_city_train_departure
        self.cross_city_train_back = cross_city_train_back
        self.cross_city_train_transfer = cross_city_train_transfer
        self.poi_data = poi_data
        self.intra_city_trans = intra_city_trans
        self.ir = ir
        self._cid = 0
        self.use_disj = False
        self.city_to_day_range = {}
        self.day_to_city = {}



    def get_daily_total_time(self,day):
        activity_time = sum(
            self.model.select_attr[day, a] * self.model.attr_data[a]['duration']
            for a in self.model.attr_by_day[day]
        ) + sum(
            self.model.select_rest[day, r] * (self.model.rest_data[r]['duration'] + self.model.rest_data[r]['queue_time'])
            for r in self.model.rest_by_day[day]
        )
        if self.ir.total_travel_days > 1:
            trans_time = sum(self.model.z_time_var[day,a] for a in self.model.attr_by_day[day]) 
        else:
            trans_time = 0

        return activity_time + trans_time

    def get_daily_total_attraction_time(self,day):
        return sum(
            self.model.select_attr[day, a] * self.model.attr_data[a]['duration']
            for a in self.model.attr_by_day[day]
        )
    def get_daily_total_rating(self,day):
        sum_rating = sum(
            self.model.select_attr[day, a] * self.model.attr_data[a]['rating']
            for a in self.model.attr_by_day[day]
        ) + sum(
            self.model.select_rest[day, r] * self.model.rest_data[r]['rating']
            for r in self.model.rest_by_day[day]
        )
        if day == 1 or (self.cfg.multi_stage and day == self.ir.stages[0].travel_days + 1):
            sum_rating += sum(
                self.model.select_hotel[day, h] * self.model.hotel_data[h]['rating']
                for h in self.model.accommodations
            )
        return sum_rating
    
    def get_daily_attraction_rating(self,day):
        return sum(
            self.model.select_attr[day, a] * self.model.attr_data[a]['rating']
            for a in self.model.attr_by_day[day]
        )
    
    def get_daily_restaurant_rating(self,day):
        return sum(
            self.model.select_rest[day, r] * self.model.rest_data[r]['rating']
            for r in self.model.rest_by_day[day]
        )
    
    def get_daily_hotel_rating(self,day):
        if day == self.ir.total_travel_days:
            return 0
        return sum(
            self.model.select_hotel[day, h] * self.model.hotel_data[h]['rating']
            for h in self.model.accommodations
        )

    def get_daily_queue_time(self,day):
        return sum(
            self.model.select_rest[day, r] * self.model.rest_data[r]['queue_time']
            for r in self.model.rest_by_day[day]
        )
    
    def get_daily_total_restaurant_time(self,day):
        return sum(
            self.model.select_rest[day, r] * self.model.rest_data[r]['duration']
            for r in self.model.rest_by_day[day]
        )
    
    def get_daily_total_transportation_time(self,day):
        return sum(
                self.model.z_time_var[day,a] for a in self.model.attr_by_day[day]
            ) if self.ir.total_travel_days > 1 else 0
    
    def get_daily_total_cost(self,day):
        ## 景点，酒店，交通，吃饭，高铁, 人数
        # print('call total cost')
        real_peoples = self.ir.peoples - self.ir.children_num
        attraction_cost = sum(
            self.model.select_attr[day, a] * self.model.attr_data[a]['cost']
            for a in self.model.attr_by_day[day]
        )
        if day == self.ir.total_travel_days:
            hotel_cost = 0
        else:
            transfer_day = self.ir.stages[0].travel_days
            if day == transfer_day:
                hotel_cost = sum(
                    self.model.select_hotel[day + 1, h] * self.model.hotel_data[h]['cost'] * self.cfg.rooms_per_night
                    for h in self.model.accommodations
                )
            else:
                hotel_cost = sum(
                    self.model.select_hotel[day, h] * self.model.hotel_data[h]['cost'] * self.cfg.rooms_per_night
                    for h in self.model.accommodations
                )

        transport_cost = sum(
            self.model.z_price_var[day,a] for a in self.model.attr_by_day[day]
        ) if self.ir.total_travel_days > 1 else 0

        restaurant_cost = sum(
            self.model.select_rest[day, r] * self.model.rest_data[r]['cost']
            for r in self.model.rest_by_day[day]
        )
        train_cost = 0
        if day == 1:
            train_cost += sum(self.model.select_train_departure[t] * self.model.train_departure_data[t]['cost']
                               for t in self.model.train_departure)
        elif day == self.ir.total_travel_days:
            train_cost += sum(self.model.select_train_back[t] * self.model.train_back_data[t]['cost']
                               for t in self.model.train_back)
        elif day == self.ir.stages[0].travel_days:
            train_cost += sum(self.model.select_train_transfer[t] * self.model.train_transfer_data[t]['cost']
                              for t in self.model.train_transfer)
            
        return transport_cost + hotel_cost + real_peoples * (attraction_cost + restaurant_cost + train_cost)

    def get_daily_total_restaurant_cost(self,day):
        real_peoples = self.ir.peoples - self.ir.children_num
        return sum(
            self.model.select_rest[day, r] * self.model.rest_data[r]['cost'] * real_peoples
            for r in self.model.rest_by_day[day]
        )
    
    def get_daily_total_attraction_cost(self,day):
        real_peoples = self.ir.peoples - self.ir.children_num
        return sum(
            self.model.select_attr[day, a] * self.model.attr_data[a]['cost'] * real_peoples
            for a in self.model.attr_by_day[day]
        )

    def get_daily_total_hotel_cost(self,day):
        if day == self.ir.total_travel_days:
            return 0
        else:
            transfer_day = self.ir.stages[0].travel_days
            if day == transfer_day:
                hotel_cost = sum(
                    self.model.select_hotel[day + 1, h] * self.model.hotel_data[h]['cost'] * self.cfg.rooms_per_night
                    for h in self.model.accommodations
                )
            else:
                hotel_cost = sum(
                    self.model.select_hotel[day, h] * self.model.hotel_data[h]['cost'] * self.cfg.rooms_per_night
                    for h in self.model.accommodations
                )
            return hotel_cost
    def get_daily_total_transportation_cost(self,day):
        if self.ir.total_travel_days <= 1:
            return 0
        real_peoples = self.ir.peoples - self.ir.children_num
        transport_cost = sum(self.model.z_price_var[day,a] for a in self.model.attr_by_day[day])
        
        train_cost = 0
        if day == 1:
            train_cost += sum(self.model.select_train_departure[t] * self.model.train_departure_data[t]['cost']
                               for t in self.model.train_departure)
        elif day == self.ir.total_travel_days:
            train_cost += sum(self.model.select_train_back[t] * self.model.train_back_data[t]['cost']
                               for t in self.model.train_back)
        elif day == self.ir.stages[0].travel_days:
            train_cost += sum(self.model.select_train_transfer[t] * self.model.train_transfer_data[t]['cost']
                              for t in self.model.train_transfer)
        
        return transport_cost + train_cost * real_peoples

    def field_extract_adapter(self, field:str, context: dict):
        field = field.lower()
        current_indices:dict = context.get('current_indices', {})
        if field == 'num_attractions_per_day':
            day = current_indices.get('day', 1)
            return sum(self.model.select_attr[day,a] for a in self.model.attractions)
        elif field == 'num_restaurants_per_day':
            day = current_indices.get('day', 1)
            return sum(self.model.select_rest[day,r] for r in self.model.restaurants)
        elif field == 'num_hotels_per_day':
            day = current_indices.get('day', 1)
            return sum(self.model.select_hotel[day,h] for h in self.model.accommodations)
        elif field == 'daily_total_time':
            day = current_indices.get('day', 1)
            return self.get_daily_total_time(day)
                
        elif field == 'daily_queue_time':
            day = current_indices.get('day', 1)
            return self.get_daily_queue_time(day)
        
        elif field == 'daily_total_restaurant_time':
            day = current_indices.get('day', 1)
            return self.get_daily_total_restaurant_time(day)
        
        elif field == 'daily_transportation_time':
            day = current_indices.get('day', 1)
            return self.get_daily_total_transportation_time(day)
        
        elif field == 'total_active_time':
            sum_time = 0
            for day in range(self.ir.total_travel_days):
                sum_time += self.get_daily_total_time(day + 1) 
            return sum_time  
        elif field == 'total_transportation_time':
            sum_time = 0
            for day in range(self.ir.total_travel_days):
                sum_time += self.get_daily_total_transportation_time(day + 1) 
            return sum_time
        elif field == 'total_queue_time':
            sum_time = 0
            for day in range(self.ir.total_travel_days):
                sum_time += self.get_daily_queue_time(day + 1) 
            return sum_time
        elif field == 'total_restaurant_time':
            sum_time = 0
            for day in range(self.ir.total_travel_days):
                sum_time += self.get_daily_total_restaurant_time(day + 1) 
            return sum_time
        elif field == 'total_cost': 
            return sum(self.get_daily_total_cost(day + 1) for day in range(self.ir.total_travel_days)) 
        elif field == 'total_hotel_cost':
            return sum(self.get_daily_total_hotel_cost(day + 1) for day in range(self.ir.total_travel_days))
        elif field == 'total_attraction_cost':
            return sum(self.get_daily_total_attraction_cost(day + 1) for day in range(self.ir.total_travel_days))
        elif field == 'total_restaurant_cost':
            return sum(self.get_daily_total_restaurant_cost(day + 1) for day in range(self.ir.total_travel_days))
        elif field == 'total_transportation_cost':
            return sum(self.get_daily_total_transportation_cost(day + 1) for day in range(self.ir.total_travel_days))
        elif field == 'daily_total_cost':
            day = current_indices.get('day', 1)
            return self.get_daily_total_cost(day)
        elif field == 'daily_total_attraction_cost':
            day = current_indices.get('day', 1)
            return self.get_daily_total_attraction_cost(day)
        elif field == 'daily_total_restaurant_cost':
            day = current_indices.get('day', 1)
            return self.get_daily_total_restaurant_cost(day)
        elif field == 'daily_total_hotel_cost':
            day = current_indices.get('day', 1)
            return self.get_daily_total_hotel_cost(day)
        elif field == 'daily_total_transportation_cost':
            day = current_indices.get('day', 1)
            return self.get_daily_total_transportation_cost(day)
        else: return 0
    def ast_to_pyomo_constraints(
        self,
        model: pyo.ConcreteModel,
        ast_node: Expr,
        context: Dict[str, Any],
        constraint_prefix: str = "constraint",
        constraint_indices: Optional[List[pyo.Set]] = None  # 多索引集合列表
    ) -> List[pyo.Constraint]:
        try:
            constraints = []
            if model is None:
                model = self.model

            # 处理OpNode的and/or逻辑（核心完善部分）
            if isinstance(ast_node, OpNode):
                # 1. 处理AND逻辑：递归拆分左右子节点，为每个子节点生成对应约束
                if ast_node.op == "and":
                    # 左子节点约束（如cond1）
                    left_constraints = self.ast_to_pyomo_constraints(
                        ast_node=ast_node.left,
                        model=model,
                        context=context.copy(),  # 复制上下文，避免左右子节点互相干扰
                        constraint_prefix=f"{constraint_prefix}_and_left",
                        constraint_indices=constraint_indices  # 继承父节点的多索引
                    )
                    # 右子节点约束（如cond2）
                    right_constraints = self.ast_to_pyomo_constraints(
                        ast_node=ast_node.right,
                        model=model,
                        context=context.copy(),
                        constraint_prefix=f"{constraint_prefix}_and_right",
                        constraint_indices=constraint_indices  # 继承父节点的多索引
                    )
                    constraints.extend(left_constraints + right_constraints)
                    return constraints  # 直接返回拆分后的约束列表

                # 2. 处理OR逻辑：在当前索引下合并为一个约束（用|连接）
                elif ast_node.op == "or":
                    self._cid += 1
                    disjunct_id = self._cid
                    self.use_disj = True
                    if not constraint_indices:
                        dL = Disjunct()
                        dR = Disjunct()
                        setattr(model, f"disjunct_{self._cid}_L", dL)
                        setattr(model, f"disjunct_{self._cid}_R", dR)
                        # 无索引场景
                        left_constraints = self.ast_to_pyomo_constraints(
                            ast_node=ast_node.left,
                            model=dL,
                            context=context.copy(),  # 复制上下文，避免左右子节点互相干扰
                            constraint_prefix=f"{constraint_prefix}_or_left",
                            constraint_indices=constraint_indices  # 继承父节点的多索引
                        )
                        # 右子节点约束（如cond2）
                        right_constraints = self.ast_to_pyomo_constraints(
                            ast_node=ast_node.right,
                            model=dR,
                            context=context.copy(),
                            constraint_prefix=f"{constraint_prefix}_or_right",
                            constraint_indices=constraint_indices  # 继承父节点的多索引
                        )
                        constraints.extend(left_constraints + right_constraints)
                        
                        # 无索引：全局一个析取集合
                        setattr(model, f"disjunction_{disjunct_id}", 
                                Disjunction(expr=[dL, dR]))
                    else:
                        index_sets = constraint_indices  # e.g. [self.model.days]
                        # 正确：创建“带索引”的 Disjunct / Disjunction
                                                # 用 rule 在每个索引下填充各自的子约束
                        def _fill_left(disj, *idx):
                            ctx = context.copy()
                            idx_names = ctx.get("index_names", [f"index_{i}" for i in range(len(idx))])
                            ctx["current_indices"] = dict(zip(idx_names, idx))
                            self.ast_to_pyomo_constraints(model=disj, ast_node=ast_node.left, context=ctx, constraint_prefix=f"{constraint_prefix}_or_left", constraint_indices=None)
                            return pyo.Constraint.Skip

                        def _fill_right(disj, *idx):
                            ctx = context.copy()
                            idx_names = ctx.get("index_names", [f"index_{i}" for i in range(len(idx))])
                            ctx["current_indices"] = dict(zip(idx_names, idx))
                            self.ast_to_pyomo_constraints(model=disj, ast_node=ast_node.right, context=ctx, constraint_prefix=f"{constraint_prefix}_or_right", constraint_indices=None)
                            return pyo.Constraint.Skip

                        model.add_component(nameL := f"disj_{disjunct_id}_L", Disjunct(*index_sets,rule = _fill_left))
                        model.add_component(nameR := f"disj_{disjunct_id}_R", Disjunct(*index_sets, rule = _fill_right))

                        dL = getattr(model, nameL); dR = getattr(model, nameR)

                        model.add_component(f"disjunction_{disjunct_id}", Disjunction(*index_sets, rule=lambda m, *idx: [dL[idx], dR[idx]]))

                    return constraints           
                    
            # 3. 处理非逻辑运算符（比较、算术等）：生成单约束（支持多索引）
            # 无索引场景
            if not constraint_indices:
                def rule(m):
                    return self._get_pyomo_expr(ast_node, context)
                constraint_name = f"{constraint_prefix}_no_index"
                setattr(model, constraint_name, pyo.Constraint(rule=rule))
                constraints.append(getattr(model, constraint_name))

            # 多索引场景
            else:
                def indexed_rule(m, *index_args):
                    index_names = context.get("index_names", [f"index_{i}" for i in range(len(index_args))])
                    context["current_indices"] = dict(zip(index_names, index_args))
                    return self._get_pyomo_expr(ast_node, context)

                constraint_name = f"{constraint_prefix}_multi_index"
                setattr(model, constraint_name, pyo.Constraint(*constraint_indices, rule=indexed_rule))
                constraints.append(getattr(model, constraint_name))

            return constraints
        
        except: pass
    def _get_pyomo_expr(self, ast_node: Expr, context: Dict[str, Any]) -> pyo.Expression:
        """
        辅助函数：将AST节点转换为Pyomo的Expression（用于构建约束）。
        
        Args:
            ast_node: AST节点
            model: Pyomo模型
            context: 上下文（包含变量映射，如{"x": model.x}）
        
        Returns:
            Pyomo表达式（如model.x[A] + model.x[B] <= 50）
        """
        # 处理ValueNode（常量）
        model = self.model
        if isinstance(ast_node, ValueNode):
            return ast_node.value
        
        # 处理FieldNode（变量，如"x"对应model.x）
        elif isinstance(ast_node, FieldNode):
            # 假设context中存储了“字段名→Pyomo变量”的映射，如{"total": model.total, "x": model.x}
            return self.field_extract_adapter(ast_node.field,context=context)
        
        # 处理OpNode（比较/逻辑）
        elif isinstance(ast_node, OpNode):
            left_expr = self._get_pyomo_expr(ast_node.left, context)
            right_expr = self._get_pyomo_expr(ast_node.right, context)
            
            # 比较运算符映射（Pyomo支持直接用<=、>=等）
            op_map = {
                "==": lambda a, b: a == b,
                "!=": lambda a, b: a != b,
                ">": lambda a, b: a > b,
                ">=": lambda a, b: a >= b,
                "<": lambda a, b: a < b,
                "<=": lambda a, b: a <= b,
            }
            
            if ast_node.op not in op_map:
                raise ValueError(f"不支持的运算符 {ast_node.op}（仅支持比较和or）")
            return op_map[ast_node.op](left_expr, right_expr)
        
        # 处理ArithmeticOpNode（算术运算，如x*0.8 + y*1.2）
        elif isinstance(ast_node, ArithmeticOpNode):
            left_expr = self._get_pyomo_expr(ast_node.left, context)
            right_expr = self._get_pyomo_expr(ast_node.right, context)
            
            op_map = {
                "+": lambda a, b: a + b,
                "-": lambda a, b: a - b,
                "*": lambda a, b: a * b,
                "/": lambda a, b: a / b
            }
            
            if ast_node.op not in op_map:
                raise ValueError(f"不支持的算术运算符 {ast_node.op}")
            return op_map[ast_node.op](left_expr, right_expr)
        
        # 处理AggregateNode（聚合，如sum(x)）
        elif isinstance(ast_node, AggregateNode):
            # 假设context中"global"对应聚合所需的列表（如产品列表、费用列表）
            agg_list = context.get("global", [])
            # 提取聚合字段（如model.x[p]中的x）
            pyomo_var = context.get(ast_node.field)
            if not pyomo_var:
                raise ValueError(f"Context中未找到聚合字段 {ast_node.field} 对应的Pyomo变量")
            
            # 生成聚合表达式（如sum(model.x[p] for p in products)）
            if ast_node.func == "sum":
                return sum(pyomo_var[p] for p in agg_list)
            elif ast_node.func in ["min", "max"]:
                return getattr(pyomo_var, ast_node.func)(p for p in agg_list)
            elif ast_node.func == "count":
                return len([p for p in agg_list if ast_node.filter.eval({"p": p})])
            
            raise ValueError(f"不支持的聚合函数 {ast_node.func}")
        
        else:
            raise TypeError(f"不支持的AST节点类型 {type(ast_node)}")
    
    def get_city_to_day_range(self):
            city_to_day_range = {}
            day_to_city = {}
            curr_day = 1
            for stg in self.ir.stages:
                city_to_day_range[stg.destinate_city] = [curr_day, curr_day + stg.travel_days - 1]
                for day in range(curr_day,curr_day + stg.travel_days):
                    day_to_city[day] = stg.destinate_city
                curr_day += stg.travel_days
            self.city_to_day_range = city_to_day_range
            self.day_to_city = day_to_city
            return city_to_day_range,day_to_city

    def get_attraction_hotel_taxi_time(self, a, h):
        return (get_trans_params(self.intra_city_trans, a, h, 'taxi_duration') + get_trans_params(self.intra_city_trans, h, a, 'taxi_duration')) 
    
    def get_attraction_hotel_taxi_cost(self, a, h):
        real_peoples = self.ir.peoples - self.ir.children_num
        cars = (real_peoples + self.cfg.peoples_per_car - 1) // self.cfg.peoples_per_car
        return  cars * (get_trans_params(self.intra_city_trans, a, h, 'taxi_cost') + get_trans_params(self.intra_city_trans, h, a, 'taxi_cost')) 
    
    def get_attraction_hotel_bus_time(self, a, h):
        return (get_trans_params(self.intra_city_trans, a, h, 'bus_duration') + get_trans_params(self.intra_city_trans, h, a, 'bus_duration')) 

    def get_attraction_hotel_bus_cost(self, a, h):
        real_peoples = self.ir.peoples - self.ir.children_num
        return real_peoples * (get_trans_params(self.intra_city_trans, a, h, 'bus_cost') + get_trans_params(self.intra_city_trans, h, a, 'bus_cost'))
            
    def make(self, cfg: dynamic_constraint):

        city_to_day_range,day_to_city = self.get_city_to_day_range()
        self.cfg = cfg
        
        attraction_dict = self.poi_data['attractions'] ## {'attractions':{'id_1':{...},'id_2':{...},...}}
        hotel_dict = self.poi_data['accommodations']
        restaurant_dict = self.poi_data['restaurants']
        
        days = range(1,self.ir.total_travel_days + 1)
        self.model.days = pyo.Set(initialize=days)
        self.model.attractions = pyo.Set(initialize=attraction_dict.keys())
        self.model.accommodations = pyo.Set(initialize=hotel_dict.keys())
        self.model.restaurants = pyo.Set(initialize=restaurant_dict.keys())
        self.model.train_departure = pyo.Set(initialize=self.cross_city_train_departure.keys())
        self.model.train_back = pyo.Set(initialize=self.cross_city_train_back.keys())

        self.model.attr_data = pyo.Param(
            self.model.attractions,
            initialize=lambda m, a: {
                'id': attraction_dict[a]['id'],
                'name': attraction_dict[a]['name'],
                'cost': float(attraction_dict[a]['cost']),
                'type': attraction_dict[a]['type'],
                'rating': float(attraction_dict[a]['rating']),
                'duration': float(attraction_dict[a]['duration']),
                'start_stage': int(attraction_dict[a]['start_stage']),
                'end_stage': int(attraction_dict[a]['end_stage'])
            },
            within=pyo.Any
        )

        self.model.hotel_data = pyo.Param(
            self.model.accommodations,
            initialize=lambda m, h: {
                'id': hotel_dict[h]['id'],
                'name': hotel_dict[h]['name'],
                'cost': float(hotel_dict[h]['cost']),
                'type': hotel_dict[h]['type'],
                'rating': float(hotel_dict[h]['rating']),
                'feature': hotel_dict[h]['feature'],
                'start_stage': int(hotel_dict[h]['start_stage']),
                'end_stage': int(hotel_dict[h]['end_stage'])
            },
            within=pyo.Any
        )

        self.model.rest_data = pyo.Param(
            self.model.restaurants,
            initialize=lambda m, r: {
                'id': restaurant_dict[r]['id'],
                'name': restaurant_dict[r]['name'],
                'cost': float(restaurant_dict[r]['cost']),
                'type': restaurant_dict[r]['type'],
                'rating': float(restaurant_dict[r]['rating']),
                'recommended_food': restaurant_dict[r]['recommended_food'],
                'queue_time': float(restaurant_dict[r]['queue_time']),
                'duration': float(restaurant_dict[r]['duration']),
                'start_stage': int(restaurant_dict[r]['start_stage']),
                'end_stage': int(restaurant_dict[r]['end_stage'])
            },
            within=pyo.Any
        )

        self.model.train_departure_data = pyo.Param(
            self.model.train_departure,
            initialize=lambda m, t: {
                'train_number': self.cross_city_train_departure[t]['train_number'],
                'cost': float(self.cross_city_train_departure[t]['cost']),
                'duration': float(self.cross_city_train_departure[t]['duration']),
                'origin_id': self.cross_city_train_departure[t]['origin_id'],
                'origin_station': self.cross_city_train_departure[t]['origin_station'],
                'destination_id': self.cross_city_train_departure[t]['destination_id'],
                'destination_station': self.cross_city_train_departure[t]['destination_station']
            },
            within=pyo.Any
        )
        self.model.train_back_data = pyo.Param(
            self.model.train_back,
            initialize=lambda m, t: {
                'train_number': self.cross_city_train_back[t]['train_number'],
                'cost': float(self.cross_city_train_back[t]['cost']),
                'duration': float(self.cross_city_train_back[t]['duration']),
                'origin_id': self.cross_city_train_back[t]['origin_id'],
                'origin_station': self.cross_city_train_back[t]['origin_station'],
                'destination_id': self.cross_city_train_back[t]['destination_id'],
                'destination_station': self.cross_city_train_back[t]['destination_station']
            },
            within=pyo.Any
        )
        if self.cfg.multi_stage:
            self.model.train_transfer = pyo.Set(initialize=self.cross_city_train_transfer.keys())
            self.model.train_transfer_data = pyo.Param(
                self.model.train_transfer,
                initialize=lambda m, t: {
                    'train_number': self.cross_city_train_transfer[t]['train_number'],
                    'cost': float(self.cross_city_train_transfer[t]['cost']),
                    'duration': float(self.cross_city_train_transfer[t]['duration']),
                    'origin_id': self.cross_city_train_transfer[t]['origin_id'],
                    'origin_station': self.cross_city_train_transfer[t]['origin_station'],
                    'destination_id': self.cross_city_train_transfer[t]['destination_id'],
                    'destination_station': self.cross_city_train_transfer[t]['destination_station']
                },
                within=pyo.Any
        )
        ## variables
        self.model.select_hotel = pyo.Var(self.model.days, self.model.accommodations, domain=pyo.Binary)
        # self.model.select_attr = pyo.Var(self.model.days, self.model.attractions, domain=pyo.Binary)
        # self.model.select_rest = pyo.Var(self.model.days, self.model.restaurants, domain=pyo.Binary)
        self.model.trans_mode = pyo.Var(self.model.days, domain=pyo.Binary) # 1为公交 0为打车
        self.model.select_train_departure = pyo.Var(self.model.train_departure, domain=pyo.Binary)
        self.model.select_train_back = pyo.Var(self.model.train_back, domain=pyo.Binary)
        if self.cfg.multi_stage:
            self.model.select_train_transfer = pyo.Var(self.model.train_transfer, domain=pyo.Binary) ## todo 换乘数据初始化
        ## last day hotel constraint
        if self.ir.total_travel_days > 1:
            def last_day_hotel_constraint(model,h):
                N = self.ir.total_travel_days
                return model.select_hotel[N-1,h] == model.select_hotel[N,h]
            
            self.model.last_day_hotel = pyo.Constraint(
                self.model.accommodations,
                rule=last_day_hotel_constraint
            )

        attr_by_day = {
            d: [a for a, v in attraction_dict.items() if v['start_stage'] <= d <= v['end_stage']]
            for d in days
        }
        rest_by_day = {
            d: [r for r, v in restaurant_dict.items() if v['start_stage'] <= d <= v['end_stage']]
            for d in days
        }
        self.model.attr_by_day = pyo.Set(self.model.days, dimen=1, initialize=attr_by_day)
        self.model.rest_by_day = pyo.Set(self.model.days, dimen=1, initialize=rest_by_day)
        AD_idx = [
            (d, a)
            for d in days
            for a in attr_by_day[d]
        ]

        RD_idx = [
            (d, r)
            for d in days
            for r in rest_by_day[d]
        ]
        
        self.model.attr_idx = pyo.Set(dimen=2, initialize=AD_idx)
        self.model.select_attr = pyo.Var(self.model.attr_idx, domain=pyo.Binary)
        self.model.rest_idx = pyo.Set(dimen=2, initialize=RD_idx)
        self.model.select_rest = pyo.Var(self.model.rest_idx, domain=pyo.Binary)

        self.model.taxi_time = pyo.Param(self.model.attractions, self.model.accommodations, initialize=lambda m,a,h: self.get_attraction_hotel_taxi_time(a,h))
        self.model.bus_time = pyo.Param(self.model.attractions, self.model.accommodations, initialize=lambda m,a,h: self.get_attraction_hotel_bus_time(a,h))
        self.model.taxi_cost = pyo.Param(self.model.attractions, self.model.accommodations, initialize=lambda m,a,h: self.get_attraction_hotel_taxi_cost(a,h))
        self.model.bus_cost = pyo.Param(self.model.attractions, self.model.accommodations, initialize=lambda m,a,h: self.get_attraction_hotel_bus_cost(a,h))
        self.model.attr_taxi_time = pyo.Expression(
            self.model.days, self.model.attractions,
            rule=lambda m, d, a: sum(m.taxi_time[a,h] * m.select_hotel[d,h] for h in m.accommodations)
        )
        self.model.attr_bus_time = pyo.Expression(
            self.model.days, self.model.attractions, 
            rule=lambda m, d, a: sum(m.bus_time[a,h] * m.select_hotel[d,h] for h in m.accommodations))
        self.model.attr_taxi_cost = pyo.Expression(
            self.model.days, self.model.attractions,
            rule=lambda m, d, a: sum(m.taxi_cost[a,h] * m.select_hotel[d,h] for h in m.accommodations))
        self.model.attr_bus_cost = pyo.Expression(
            self.model.days, self.model.attractions,
            rule=lambda m, d, a: sum(m.bus_cost[a,h] * m.select_hotel[d,h] for h in m.accommodations))


        self.model.z_time  = pyo.Var(self.model.days, self.model.attractions, domain=pyo.NonNegativeReals)
        self.model.z_price = pyo.Var(self.model.days, self.model.attractions, domain=pyo.NonNegativeReals)

        BIG_M = 10000

        # ===== time =====
        def ztime_upper1(m, d, a):
            return m.z_time[d,a] <= m.attr_bus_time[d,a] + BIG_M*(1 - m.trans_mode[d])

        def ztime_upper2(m, d, a):
            return m.z_time[d,a] <= m.attr_taxi_time[d,a] + BIG_M*(m.trans_mode[d])

        def ztime_lower1(m, d, a):
            return m.z_time[d,a] >= m.attr_bus_time[d,a] - BIG_M*(1 - m.trans_mode[d])

        def ztime_lower2(m, d, a):
            return m.z_time[d,a] >= m.attr_taxi_time[d,a] - BIG_M*(m.trans_mode[d])

        # ===== cost =====
        def zprice_upper1(m, d, a):
            return m.z_price[d,a] <= m.attr_bus_cost[d,a] + BIG_M*(1 - m.trans_mode[d])

        def zprice_upper2(m, d, a):
            return m.z_price[d,a] <= m.attr_taxi_cost[d,a] + BIG_M*(m.trans_mode[d])

        def zprice_lower1(m, d, a):
            return m.z_price[d,a] >= m.attr_bus_cost[d,a] - BIG_M*(1 - m.trans_mode[d])

        def zprice_lower2(m, d, a):
            return m.z_price[d,a] >= m.attr_taxi_cost[d,a] - BIG_M*(m.trans_mode[d])


        self.model.z_time_up1 = pyo.Constraint(self.model.days, self.model.attractions, rule=ztime_upper1)
        self.model.z_time_up2 = pyo.Constraint(self.model.days, self.model.attractions, rule=ztime_upper2)
        self.model.z_time_lo1 = pyo.Constraint(self.model.days, self.model.attractions, rule=ztime_lower1)
        self.model.z_time_lo2 = pyo.Constraint(self.model.days, self.model.attractions, rule=ztime_lower2)

        self.model.z_price_up1 = pyo.Constraint(self.model.days, self.model.attractions, rule=zprice_upper1)
        self.model.z_price_up2 = pyo.Constraint(self.model.days, self.model.attractions, rule=zprice_upper2)
        self.model.z_price_lo1 = pyo.Constraint(self.model.days, self.model.attractions, rule=zprice_lower1)
        self.model.z_price_lo2 = pyo.Constraint(self.model.days, self.model.attractions, rule=zprice_lower2)
  
        self.model.z_time_var  = pyo.Var(self.model.attr_idx, domain=pyo.NonNegativeReals)
        self.model.z_price_var = pyo.Var(self.model.attr_idx, domain=pyo.NonNegativeReals) 

        def ztime_var1(m, d, a):
            return m.z_time_var[d,a] <= BIG_M * m.select_attr[d,a]

        def ztime_var2(m, d, a):
            return m.z_time_var[d,a] <= m.z_time[d,a] 

        def ztime_var3(m, d, a):
            return m.z_time_var[d,a] >= m.z_time[d,a] - BIG_M*(1 - m.select_attr[d,a])

        def ztime_var4(m, d, a):
            return m.z_time_var[d,a] >= 0

        self.model.z_time_var1 = pyo.Constraint(self.model.attr_idx, rule=ztime_var1)
        self.model.z_time_var2 = pyo.Constraint(self.model.attr_idx, rule=ztime_var2)
        self.model.z_time_var3 = pyo.Constraint(self.model.attr_idx, rule=ztime_var3)
        self.model.z_time_var4 = pyo.Constraint(self.model.attr_idx, rule=ztime_var4)

        def zprice_var1(m, d, a):
            return m.z_price_var[d,a] <= BIG_M * m.select_attr[d,a]

        def zprice_var2(m, d, a):
            return m.z_price_var[d,a] <= m.z_price[d,a] 

        def zprice_var3(m, d, a):
            return m.z_price_var[d,a] >= m.z_price[d,a] - BIG_M*(1 - m.select_attr[d,a])

        def zprice_var4(m, d, a):
            return m.z_price_var[d,a] >= 0

        self.model.z_price_var1 = pyo.Constraint(self.model.attr_idx, rule=zprice_var1)
        self.model.z_price_var2 = pyo.Constraint(self.model.attr_idx, rule=zprice_var2)
        self.model.z_price_var3 = pyo.Constraint(self.model.attr_idx, rule=zprice_var3)
        self.model.z_price_var4 = pyo.Constraint(self.model.attr_idx, rule=zprice_var4)       


        self.model.unique_attr = pyo.Constraint(
            self.model.attractions,
            rule=lambda m, a: sum(m.select_attr[d, a] for d in range(m.attr_data[a]['start_stage'],m.attr_data[a]['end_stage'] + 1)) <= 1
        )

        self.model.unique_rest = pyo.Constraint(
            self.model.restaurants,
            rule=lambda m, r: sum(m.select_rest[d, r] for d in range(m.rest_data[r]['start_stage'],m.rest_data[r]['end_stage'] + 1)) <= 1
        )

        transfer_day = 0
        if self.cfg.multi_stage:
            transfer_day = self.ir.stages[0].travel_days
        def same_hotel_rule(m, d, h):
            if d == 1 or d == transfer_day + 1: #select_hotel 表示当天出去玩出发的酒店，用于计算交通
                return pyo.Constraint.Skip
            return m.select_hotel[d, h] == m.select_hotel[d-1, h]
        
        self.model.same_hotel = pyo.Constraint(self.model.days, self.model.accommodations, rule=same_hotel_rule)

              
        def stage_hotel_rule(m, d, h):
            transfer_day = self.ir.stages[0].travel_days
            if self.cfg.multi_stage and d == transfer_day:
                return pyo.Constraint.Skip ##换乘酒店跨市
            return m.select_hotel[d, h] * (d - self.model.hotel_data[h]['start_stage']) * (d - self.model.hotel_data[h]['end_stage']) <= 0
        
        self.model.stage_hotel = pyo.Constraint(self.model.days, self.model.accommodations, rule=stage_hotel_rule)



        if len(self.cross_city_train_departure) > 0:
            self.model.one_departure = pyo.Constraint(
                rule=lambda m: sum(m.select_train_departure[t] for t in m.train_departure) == 1
            )
        if len(self.cross_city_train_back) > 0:
            self.model.one_back = pyo.Constraint(
                rule=lambda m: sum(m.select_train_back[t] for t in m.train_back) == 1
            )
        if len(self.cross_city_train_transfer) and self.cfg.multi_stage > 0:
            self.model.one_transfer = pyo.Constraint(
                rule=lambda m: sum(m.select_train_transfer[t] for t in m.train_transfer) == 1
            )
        ##约束1

        def attr_num_rule(m, d):
            # 只在当天可选的景点上求和
            return sum(m.select_attr[d, a] for (dd, a) in m.attr_idx if dd == d) == 1

        self.model.attr_num = pyo.Constraint(self.model.days, rule=attr_num_rule)

        def rest_num_rule(m, d):
            # 只在当天可选的景点上求和
            return sum(m.select_rest[d, r] for (dd, r) in m.rest_idx if dd == d) == 3

        self.model.rest_num = pyo.Constraint(self.model.days, rule=rest_num_rule)
        

        self.model.hotel_num = pyo.Constraint(
            self.model.days,
            rule=lambda m, d:  sum(m.select_hotel[d, h] for h in m.accommodations) == 1
        )

        ##约束2
        
        def infra_trans_rule(m, d):
            city = day_to_city[d]
            if not cfg.infra_city_transportation:
                return pyo.Constraint.Skip
            if cfg.infra_city_transportation[city] == 'public_transportation':
                return m.trans_mode[d] == 1
            elif cfg.infra_city_transportation[city] == 'taxi':
                return m.trans_mode[d] == 0
            else:
                return pyo.Constraint.Skip
            
        self.model.trans_mode_rule = pyo.Constraint(
            self.model.days,
            rule=infra_trans_rule
        )

        ##约束4 每日活动时间约束
        if cfg.daily_total_time :
            self.ast_to_pyomo_constraints(self.model,cfg.daily_total_time,{'index_names':['day']},"daily_total_time",[self.model.days])
        else:
            self.model.daily_time_rule = pyo.Constraint(
                self.model.days,
                rule=lambda m, d: self.get_daily_total_time(d) <= 840
            )
        ## 用户约束
        if cfg.daily_queue_time :
            self.ast_to_pyomo_constraints(self.model,cfg.daily_queue_time,{'index_names':['day']},"daily_queue_time",[self.model.days])
        
        if cfg.daily_total_restaurant_time :
            self.ast_to_pyomo_constraints(self.model,cfg.daily_total_restaurant_time,{'index_names':['day']},"daily_total_restaurant_time",[self.model.days])
        
        if cfg.daily_transportation_time :
            self.ast_to_pyomo_constraints(self.model,cfg.daily_transportation_time,{'index_names':['day']},"daily_transportation_time",[self.model.days])

        if cfg.total_active_time :
            self.ast_to_pyomo_constraints(self.model,cfg.total_active_time,{},"total_active_time",constraint_indices=None)

        if cfg.total_restaurant_time:
            self.ast_to_pyomo_constraints(self.model,cfg.total_restaurant_time,{},"total_restaurant_time",constraint_indices=None)

        if cfg.total_queue_time :
            self.ast_to_pyomo_constraints(self.model,cfg.total_queue_time,{},"total_queue_time",constraint_indices=None)
        
        if cfg.total_transportation_time :
            self.ast_to_pyomo_constraints(self.model,cfg.total_transportation_time,{},"total_transportation_time",constraint_indices=None)
        
        if cfg.total_budget :
            self.ast_to_pyomo_constraints(self.model,cfg.total_budget,{},"total_budget",constraint_indices=None)
        
        if cfg.total_meal_budget :
            self.ast_to_pyomo_constraints(self.model,cfg.total_meal_budget,{},"total_meal_budget",constraint_indices=None)
        
        if cfg.total_attraction_ticket_budget :
            self.ast_to_pyomo_constraints(self.model,cfg.total_attraction_ticket_budget,{},"total_attraction_budget",constraint_indices=None)
        
        if cfg.total_hotel_budget :
            self.ast_to_pyomo_constraints(self.model,cfg.total_hotel_budget,{},"total_hotel_budget",constraint_indices=None)
        
        if cfg.total_transportation_budget :
            self.ast_to_pyomo_constraints(self.model,cfg.total_transportation_budget,{},"total_transportation_budget",constraint_indices=None)
        
        if cfg.daily_total_budget:
            self.ast_to_pyomo_constraints(self.model,cfg.daily_total_budget,{'index_names':['day']},"daily_total_budget",[self.model.days])
        
        if cfg.daily_total_meal_budget:
            self.ast_to_pyomo_constraints(self.model,cfg.daily_total_meal_budget,{'index_names':['day']},"daily_total_meal_budget",[self.model.days])
        
        if cfg.daily_total_attraction_ticket_budget:
            self.ast_to_pyomo_constraints(self.model,cfg.daily_total_attraction_ticket_budget,{'index_names':['day']},"daily_total_attraction_budget",[self.model.days])
        
        if cfg.daily_total_hotel_budget:
            self.ast_to_pyomo_constraints(self.model,cfg.daily_total_hotel_budget,{'index_names':['day']},"daily_total_hotel_budget",[self.model.days])
        
        if cfg.daily_total_transportation_budget:
            self.ast_to_pyomo_constraints(self.model,cfg.daily_total_transportation_budget,{'index_names':['day']},"daily_total_transportation_budget",[self.model.days])

        try:
        #############extra code##############


            
            pass
        #############extra code##############
        except: pass

    def configure_solver(self):
        solver = pyo.SolverFactory('scip')
        solver.options['limits/time'] = 50
        solver.options['limits/gap'] = 0.0
        solver.options['presolving/emphasis'] = 'fast'  
        solver.options["separating/maxrounds"] = 5  # 缩短 cuts 生成时间
        return solver
    
    def solve(self):
        solver = self.configure_solver()
        results = solver.solve(self.model, tee=True)
        return results
    

    def generate_date_range(self, start_date, date_format="%Y年%m月%d日"):
        start = datetime.strptime(start_date, date_format)
        days = self.ir.total_travel_days
        return [
            (start + timedelta(days=i)).strftime(date_format)
            for i in range(days)
        ]


    def get_selected_train(self,train_type='departure'):
        model = self.model
        if train_type not in ['departure', 'back', 'intermediate']:
            raise ValueError("train_type must in ['departure', 'back', 'intermediate']")

        if train_type == 'back':
            train_set = model.train_back
            train_data = model.train_back_data
        elif train_type == 'departure':
            train_set = model.train_departure
            train_data = model.train_departure_data
        else:
            train_set = model.train_transfer
            train_data = model.train_transfer_data

        selected_train = [
            train_data[t]
            for t in train_set
            if pyo.value(
                model.select_train_departure[t] if train_type == 'departure'
                else model.select_train_back[t] if train_type == 'back'
                else model.select_train_transfer[t] if train_type == 'intermediate'
                else 0
            ) > 0.9
        ]
        return selected_train[0] if selected_train else 'null'


    def get_selected_poi(self, type, day, selected_poi):
        model = self.model
        if type == 'restaurant':
            poi_set = model.rest_by_day[day]
            poi_data = model.rest_data
            select_set = model.select_rest
        else:
            poi_set = model.attr_by_day[day]
            poi_data = model.attr_data
            select_set = model.select_attr

        selected_poi = [
            poi_data[t]
            for t in poi_set
            if t not in selected_poi and pyo.value(select_set[day, t]) > 0.9
        ]
        return selected_poi


    def get_selected_hotel(self,day):
        model = self.model
        transfer_day = self.ir.stages[0].travel_days
        if len(self.ir.stages) > 1 and day == transfer_day:
            selected_hotel = [
                model.hotel_data[t]
                for t in model.accommodations
                if pyo.value(model.select_hotel[day + 1,t]) > 0.9
            ]
        else:
            selected_hotel = [
                model.hotel_data[t]
                for t in model.accommodations
                if pyo.value(model.select_hotel[day,t]) > 0.9
            ]
        return selected_hotel[0] if selected_hotel else 'null'


    def get_time(self, selected_attr, selected_rest, selected_hotel, day, transfer_departure_hotel = None):
        intra_city_trans = self.intra_city_trans
        model = self.model
        daily_time = 0
        transport_time = 0
        daily_time += selected_attr['duration']
        for r in selected_rest:
            daily_time += r['queue_time'] + r['duration']
        if transfer_departure_hotel:
            transfer_hotel = transfer_departure_hotel
        else:
            transfer_hotel = selected_hotel

        if self.ir.total_travel_days > 1 :
            ## 提取出景点顺序
            if pyo.value(model.trans_mode[day]) > 0.9:
                transport_time = get_trans_params(
                    intra_city_trans,
                    transfer_hotel['id'],
                    selected_attr['id'],
                    'bus_duration'
                ) + get_trans_params(
                    intra_city_trans,
                    selected_attr['id'],
                    transfer_hotel['id'],
                    'bus_duration'
                )
            else:
                transport_time = get_trans_params(
                    intra_city_trans,
                    transfer_hotel['id'],
                    selected_attr['id'],
                    'taxi_duration'
                ) + get_trans_params(
                    intra_city_trans,
                    selected_attr['id'],
                    transfer_hotel['id'],
                    'taxi_duration'
                )

        return daily_time + transport_time, transport_time


    def get_cost(self, selected_attr, selected_rest, departure_trains,transfer_trains, back_trains, selected_hotel, day, transfer_departure_hotel = None):

        model = self.model
        intra_city_trans = self.intra_city_trans
        daily_cost = 0
        transport_cost = 0
        real_peoples = self.ir.peoples - self.ir.children_num
        daily_cost += real_peoples * selected_attr['cost']
        travel_days = self.ir.total_travel_days
        if transfer_departure_hotel:
            transfer_hotel = transfer_departure_hotel
        else:
            transfer_hotel = selected_hotel
        for r in selected_rest:
            if r:
                daily_cost += real_peoples * r['cost']
        if travel_days > 1 :
            if pyo.value(model.trans_mode[day]) > 0.9:
                transport_cost = real_peoples * get_trans_params(
                    intra_city_trans,
                    transfer_hotel['id'],
                    selected_attr['id'],
                    'bus_cost'
                ) + real_peoples * get_trans_params(
                    intra_city_trans,
                    selected_attr['id'],
                    transfer_hotel['id'],
                    'bus_cost'
                )
            else:
                transport_cost = ((real_peoples + self.cfg.peoples_per_car - 1) // self.cfg.peoples_per_car) * get_trans_params(
                    intra_city_trans,
                    transfer_hotel['id'],
                    selected_attr['id'],
                    'taxi_cost'
                ) + ((real_peoples + self.cfg.peoples_per_car - 1) // self.cfg.peoples_per_car) * get_trans_params(
                    intra_city_trans,
                    selected_attr['id'],
                    transfer_hotel['id'],
                    'taxi_cost'
                )

        if day != travel_days:
            daily_cost += selected_hotel['cost'] * self.cfg.rooms_per_night

        transfer_day = self.ir.stages[0].travel_days
        if day == 1 and isinstance(departure_trains,dict):
            daily_cost += real_peoples * departure_trains['cost']
        elif day == travel_days and isinstance(back_trains,dict):
            daily_cost += real_peoples * back_trains['cost']
        elif len(self.ir.stages) > 1 and day == transfer_day:
            daily_cost += real_peoples * transfer_trains['cost']

        return daily_cost + transport_cost, transport_cost

    def generate_daily_plan(self):
        model = self.model
        multi_stage = self.cfg.multi_stage
        intra_city_trans = self.intra_city_trans
        departure_trains = self.get_selected_train('departure')
        back_trains = self.get_selected_train('back')
        transfer_trains = {}
        if multi_stage:
            transfer_trains = self.get_selected_train('intermediate')
            intermediate_city = self.ir.stages[0].destinate_city
        total_cost = 0
        daily_plans = []
        select_at = []
        select_re = []
        travel_days = self.ir.total_travel_days
        transfer_day = self.ir.stages[0].travel_days
        date = self.generate_date_range(self.ir.start_date) ##todo
        for day in range(1, travel_days + 1):
            attr_details = self.get_selected_poi('attraction', day, select_at)[0]
            select_at.append(attr_details['id'])
            rest_details = []
            rest_details = self.get_selected_poi('restaurant', day, select_re)
            rest_details = (rest_details + [None, None, None])[:3]
            for r in rest_details:
                if r:
                    select_re.append(r['id'])

            meal_allocation = {
                'breakfast': rest_details[0],
                'lunch': rest_details[1],
                'dinner': rest_details[2]
            }
            selected_hotel = self.get_selected_hotel(day)        
            if self.cfg.multi_stage and day == transfer_day and day > 1:
                daily_time, transport_time = self.get_time(attr_details, rest_details, selected_hotel, day, transfer_departure_hotel=self.get_selected_hotel(day - 1))
                daily_cost, transport_cost = self.get_cost(attr_details, rest_details, departure_trains,transfer_trains,back_trains, selected_hotel, day, transfer_departure_hotel=self.get_selected_hotel(day - 1))
            else:
                daily_time, transport_time = self.get_time(attr_details, rest_details, selected_hotel, day)
                daily_cost, transport_cost = self.get_cost(attr_details, rest_details, departure_trains,transfer_trains,back_trains, selected_hotel, day)
            day_plan = {
                "date": f"{date[day - 1]}",
                "cost": round(daily_cost, 2),
                "cost_time": round(daily_time, 2),
                "hotel": selected_hotel if day != travel_days else "null",
                "attractions": attr_details,
                "restaurants": [
                    {
                        "type": meal_type,
                        "restaurant": rest if rest else None
                    } for meal_type, rest in meal_allocation.items()
                ],
                "transport": {
                    "mode": "bus" if pyo.value(model.trans_mode[day]) > 0.9 else "taxi",
                    "cost": round(transport_cost, 2),
                    "duration": round(transport_time, 2)
                }
            }
            daily_plans.append(day_plan)
            total_cost += daily_cost

        origin_city = self.ir.stages[0].origin_city
        if len(self.ir.stages) > 1:
            intermediate_city = self.ir.stages[0].destinate_city
            destinate_city = self.ir.stages[1].destinate_city
        else:
            destinate_city = self.ir.stages[0].destinate_city
        result = { 
            "budget": self.ir.budgets,
            "peoples": self.ir.peoples,
            "travel_days": travel_days,
            "origin_city": origin_city,
            "destination_city": destinate_city,
            "start_date": self.ir.start_date,
            "end_date": date[-1],
            "daily_plans": daily_plans,
            "departure_trains": departure_trains,
            "back_trains": back_trains,
            "total_cost": round(total_cost, 2),
            "objective_value": round(pyo.value(model.obj), 2)
        }
        if len(self.ir.stages) > 1:
            result['intermediate_city'] = intermediate_city
            result['intermediate_trains'] = transfer_trains
        return result
    
def get_solution(omo:template):
    if omo.use_disj:
        TransformationFactory('gdp.bigm').apply_to(omo.model)
    solver = omo.configure_solver()
    results = solver.solve(omo.model, tee=True)
    plan = omo.generate_daily_plan()
    print(f"```generated_plan\n{plan}\n```")



#Todo 目标函数 + LLM交互文件

if __name__ == '__main__':
    cross_city_train_departure = []
    cross_city_train_back = []
    cross_city_train_transfer = []
    poi_data = {'attractions': {}, 'accommodations': {}, 'restaurants': {}}
    intra_city_trans = {}

    ###########################IR && dynamic_constraint#############################



    ##########################################################

    try:
        cross_city_train_departure,cross_city_train_transfer,cross_city_train_back, poi_data, intra_city_trans = fetch_data(ir) #TODO 将该行改为const.py中的代码
        # from mock import get_mock_data
        # cross_city_train_departure,cross_city_train_transfer,cross_city_train_back, poi_data, intra_city_trans = get_mock_data(days=[item.travel_days for item in ir.stages])
        for item in cross_city_train_departure:
            item['cost'] = float(item['cost'])
            item['duration'] = float(item['duration'])

        for item in cross_city_train_transfer:
                item['cost'] = float(item['cost'])
                item['duration'] = float(item['duration'])

        for item in cross_city_train_back:
            item['cost'] = float(item['cost'])
            item['duration'] = float(item['duration'])

        for _ in poi_data['attractions']:
            for item in _:
                item['rating'] = float(item['rating'])
                item['cost'] = float(item['cost'])
                item['duration'] = float(item['duration'])
        
        for _ in poi_data['accommodations']:
            for item in _:
                item['rating'] = float(item['rating'])
                item['cost'] = float(item['cost'])

        for _ in poi_data['restaurants']:
            for item in _:
                item['rating'] = float(item['rating'])
                item['cost'] = float(item['cost'])
                item['duration'] = float(item['duration'])
                item['queue_time'] = float(item['queue_time'])
        # fetch 数据
        cross_city_train_departure, cross_city_train_transfer,cross_city_train_back, poi_data = rough_rank(cross_city_train_departure=cross_city_train_departure,cross_city_train_transfer=cross_city_train_transfer,cross_city_train_back=cross_city_train_back,poi_data=poi_data,ir=ir)
        # TODO 在下面一行加断点检查数据合法性（看poi_data）
        tp = template(cross_city_train_departure=cross_city_train_departure,cross_city_train_transfer=cross_city_train_transfer,cross_city_train_back=cross_city_train_back,poi_data=poi_data,intra_city_trans=intra_city_trans,ir=ir)
        tp.make(dc)
        # 构造template
        # make
        get_solution(tp)
        # get_solution
    except Exception as e:
        # raise e
        def generate_date_range(start_date, date_format="%Y年%m月%d日"):
            start = datetime.strptime(start_date, date_format)
            days = ir.total_travel_days
            return [
                (start + timedelta(days=i)).strftime(date_format)
                for i in range(days)
            ]
        date = generate_date_range(ir.start_date)
        departure_trains = back_trains = transfer_trains = 'null'
        use_transfer = False
        if isinstance(cross_city_train_departure, dict) and cross_city_train_departure:
            first_key = next(iter(cross_city_train_departure))
            departure_trains = cross_city_train_departure[first_key]
        if isinstance(cross_city_train_back, dict) and cross_city_train_back:
            first_key = next(iter(cross_city_train_back))
            back_trains = cross_city_train_back[first_key]
        if isinstance(cross_city_train_transfer, dict) and cross_city_train_transfer:
            use_transfer = True
            first_key = next(iter(cross_city_train_transfer))
            transfer_trains = cross_city_train_transfer[first_key]
        daily_plans = []
        selected_hotel = 'null'
        attr_details = 'null'
        meal_allocation = {
            'breakfast': 'null',
            'lunch': 'null',
            'dinner': 'null'
        }
        for day in range(1, ir.total_travel_days + 1):
            
            if len(poi_data['accommodations']) >= day:
                key = list(poi_data['accommodations'].keys())[day - 1]
                selected_hotel = poi_data['accommodations'][key]
            
            if len(poi_data['attractions']) >= day:
                key = list(poi_data['attractions'].keys())[day - 1]
                attr_details = poi_data['attractions'][key]

            if len(poi_data['restaurants']) >= day * 3:
                key = list(poi_data['restaurants'].keys())[(day-1) * 3 ]
                key_1 = list(poi_data['restaurants'].keys())[(day-1) * 3 + 1]
                key_2 = list(poi_data['restaurants'].keys())[(day-1) * 3 + 2]
                meal_allocation = { #todo
                    'breakfast': poi_data['restaurants'][key],
                    'lunch': poi_data['restaurants'][key_1],
                    'dinner': poi_data['restaurants'][key_2]
                }
            elif meal_allocation['breakfast'] == 'null' and len(poi_data['restaurants']) > 0:
                key_1 = list(poi_data['restaurants'].keys())[0]
                key_2 = list(poi_data['restaurants'].keys())[1] if len(poi_data['restaurants']) > 1 else None
                meal_allocation = {
                    'breakfast': poi_data['restaurants'][key_1],
                    'lunch': poi_data['restaurants'][key_2] if key_2 else 'null',
                    'dinner': 'null'
                }
            trans_mode = 'bus' if '公共交通' in user_question or '公交' in user_question or '地铁' in user_question else 'taxi'
            day_plan = {
                "date": f"{date[day - 1]}",
                "cost": 0,
                "cost_time": 0,
                "hotel": selected_hotel if day != ir.total_travel_days else "null",
                "attractions": attr_details,
                "restaurants": [
                    {
                        "type": meal_type,
                        "restaurant": rest if rest else None
                    } for meal_type, rest in meal_allocation.items()
                ],
                "transport": {
                    "mode": trans_mode,
                    "cost": 0,
                    "duration": 0
                }
            }
            daily_plans.append(day_plan)

        origin_city = ir.stages[0].origin_city
        if use_transfer:
            intermediate_city = ir.stages[0].destinate_city
            destinate_city = ir.stages[1].destinate_city
        else:
            destinate_city = ir.stages[0].destinate_city    
        plan = {
            "budget": ir.budgets,
            "peoples": ir.peoples,
            "travel_days": ir.total_travel_days,
            "origin_city": origin_city,
            "destination_city": destinate_city,
            "start_date": ir.start_date,
            "end_date": date[-1],
            "daily_plans": daily_plans,
            "departure_trains": departure_trains,
            "back_trains": back_trains,
            "total_cost": 0,
            "objective_value": 0
        }
        if use_transfer:
            plan['intermediate_city'] = intermediate_city
            plan['intermediate_trains'] = transfer_trains

        print(f"```generated_plan\n{plan}\n```")