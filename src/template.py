from __future__ import annotations
import pyomo.environ as pyo 
from pyomo.gdp import Disjunct, Disjunction
from pyomo.core import TransformationFactory
import requests
from datetime import timedelta, datetime
from dataclasses import dataclass,field,asdict
from typing import Any, Callable, Dict, Optional,List
import json


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

def _is_iterable(obj: Any) -> bool:
    """检查对象是否可迭代，用于安全地处理集合操作。
    
    Args:
        obj: 待检查的对象
        
    Returns:
        如果对象可迭代返回True，否则返回False
    """
    if obj is None:
        return False
    try:
        iter(obj)
        return True
    except TypeError:
        return False


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
    travel_days: int  # 旅行天数
    original_city: str  # 出发城市
    destinate_city: str  # 目的地城市（拼写错误，应为destination_city）
    budgets: int = 0  # 总预算 默认不设总预算

    # 各类别的约束条件，使用表达式树表示
    attraction_constraints: Optional[Expr] = None  # 景点约束
    accommodation_constraints: Optional[Expr] = None  # 住宿约束
    restaurant_constraints: Optional[Expr] = None  # 餐厅约束
    departure_transport_constraints: Optional[Expr] = None  # 交通约束
    back_transport_constraints: Optional[Expr] = None



@dataclass
class dynamic_constraint:
    """动态约束类，用于表示旅行规划中的各种动态约束条件。
    
    这些约束条件会根据具体的旅行需求动态调整，包括时间、预算、选择频率等。
    使用表达式树(Expr)来表示复杂的约束逻辑，支持运行时计算。
    """
    num_travlers: int = None
    rooms_per_night: int = None
    change_hotel: bool = False
    ## 时间相关
    daily_total_time: Optional[Expr] = field(default_factory= lambda: OpNode('<=',FieldNode('daily_total_time'),ValueNode(840)))
    daily_queue_time: Optional[Expr] = None
    daily_total_restaurant_time: Optional[Expr] = None

    daily_transportation_time: Optional[Expr] = None

    total_active_time: Optional[Expr] = None
    total_queue_time: Optional[Expr] = None
    total_restaurant_time: Optional[Expr] = None
    total_transportation_time: Optional[Expr] = None
    ## POI相关
    num_attractions_per_day: Optional[Expr] = field(default_factory= lambda: OpNode('==',FieldNode('num_attractions_per_day'),ValueNode(1)))
    num_restaurants_per_day: Optional[Expr] = field(default_factory= lambda: OpNode('==',FieldNode('num_restaurants_per_day'),ValueNode(3)))
    num_hotels_per_day: Optional[Expr] = field(default_factory= lambda: OpNode('==',FieldNode('num_hotels_per_day'),ValueNode(1)))

    ## 交通相关
    infra_city_transportation: str = 'none' # 'public_transportation' or 'taxi' or 'none'

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
        departure_transport_constraints=parse_constraint(data.get("departure_transport_constraints")),
        back_transport_constraints=parse_constraint(data.get("back_transport_constraints")),
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
        # 基础字段
        num_travlers=data["num_travlers"],
        rooms_per_night=data["rooms_per_night"],
        change_hotel=data["change_hotel"],
        # 时间相关
        daily_total_time=parse_expr(data.get("daily_total_time")),
        daily_queue_time=parse_expr(data.get("daily_queue_time")),
        daily_total_restaurant_time=parse_expr(data.get("daily_total_restaurant_time")),
        daily_transportation_time=parse_expr(data.get("daily_transportation_time")),
        total_active_time=parse_expr(data.get("total_active_time")),
        total_queue_time=parse_expr(data.get("total_queue_time")),
        total_restaurant_time=parse_expr(data.get("total_restaurant_time")),  # 保持原始拼写
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
        extra=data.get("extra") or ""
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
        # 基础字段
        "num_travlers", "rooms_per_night", "change_hotel",
        # 时间相关
        "daily_total_time", "daily_queue_time", "daily_total_restaurant_time", "daily_transportation_time",
        "total_active_time", "total_queue_time", "total_restaurant_time", "total_transportation_time",
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

def fetch_data(ir: IR):
    origin_city = ir.original_city
    destination_city = ir.destinate_city 
    if '市' not in origin_city:
        origin_city = origin_city + '市'
    if '市' not in destination_city:
        destination_city = destination_city + '市'

    url = "http://localhost:12457"
    max_retry = 3
    while max_retry > 0:
        try:
            cross_city_train_departure = requests.get(
                url + f"/cross-city-transport?origin_city={origin_city}&destination_city={destination_city}").json()
            cross_city_train_back = requests.get(
                url + f"/cross-city-transport?origin_city={destination_city}&destination_city={origin_city}").json()

            poi_data = {
                'attractions': requests.get(url + f"/attractions/{destination_city}").json(),
                'accommodations': requests.get(url + f"/accommodations/{destination_city}").json(),
                'restaurants': requests.get(url + f"/restaurants/{destination_city}").json()
            }

            intra_city_trans = requests.get(url + f"/intra-city-transport/{destination_city}").json()
            return cross_city_train_departure, cross_city_train_back, poi_data, intra_city_trans
        except:
            max_retry -= 1
    return [],[],{'attractions':[],'accommodations':[],'restaurants':[]},{}

def rough_rank(cross_city_train_departure:list[dict],cross_city_train_back,poi_data,ir:IR):
    def create_context(item,key,value):
        return {**item,key:value}
    
    if ir.departure_transport_constraints:
        if isinstance(ir.departure_transport_constraints,OpNode):
            cross_city_train_departure = [item for item in cross_city_train_departure if ir.departure_transport_constraints.eval(create_context(item,'global',cross_city_train_departure))]
        elif isinstance(ir.departure_transport_constraints, AggregateNode) and (ir.departure_transport_constraints.func == 'min' or ir.departure_transport_constraints.func == 'max'):
            cross_city_train_departure = ir.departure_transport_constraints.eval({'global':cross_city_train_departure})
  
    if ir.back_transport_constraints:
        if isinstance(ir.back_transport_constraints,OpNode):
            cross_city_train_back = [item for item in cross_city_train_back if ir.back_transport_constraints.eval(create_context(item,'global',cross_city_train_back))]
        elif isinstance(ir.back_transport_constraints, AggregateNode) and (ir.back_transport_constraints.func == 'min' or ir.back_transport_constraints.func == 'max'):
            cross_city_train_back = ir.back_transport_constraints.eval({'global':cross_city_train_back})

    if ir.accommodation_constraints:
        if isinstance(ir.accommodation_constraints,OpNode):
            poi_data['accommodations'] = [item for item in poi_data['accommodations'] if ir.accommodation_constraints.eval(create_context(item,'global',poi_data['accommodations']))]
        elif isinstance(ir.accommodation_constraints, AggregateNode) and (ir.accommodation_constraints.func == 'min' or ir.accommodation_constraints.func == 'max'):
            poi_data['accommodations'] = ir.accommodation_constraints.eval({'global':poi_data['accommodations']})
    
    if ir.restaurant_constraints:
        if isinstance(ir.restaurant_constraints,OpNode):
            poi_data['restaurants'] = [item for item in poi_data['restaurants'] if ir.restaurant_constraints.eval(create_context(item,'global',poi_data['restaurants']))]
        elif isinstance(ir.restaurant_constraints, AggregateNode) and (ir.restaurant_constraints.func == 'min' or ir.restaurant_constraints.func == 'max'):
            poi_data['restaurants'] = ir.restaurant_constraints.eval({'global':poi_data['restaurants']})
    
    if ir.attraction_constraints:
        if isinstance(ir.attraction_constraints,OpNode):
            poi_data['attractions'] = [item for item in poi_data['attractions'] if ir.attraction_constraints.eval(create_context(item,'global',poi_data['attractions']))]
        elif isinstance(ir.attraction_constraints, AggregateNode) and (ir.attraction_constraints.func == 'min' or ir.attraction_constraints.func == 'max'):
            poi_data['attractions'] = ir.attraction_constraints.eval({'global':poi_data['attractions']})

    attraction_dict = {a['id']: a for a in poi_data['attractions']}
    hotel_dict = {h['id']: h for h in poi_data['accommodations']}
    restaurant_dict = {r['id']: r for r in poi_data['restaurants']}
    train_departure_dict = {t['train_number']: t for t in cross_city_train_departure}
    train_back_dict = {t['train_number']: t for t in cross_city_train_back}
    pois = {'attractions': attraction_dict,'accommodations': hotel_dict,'restaurants': restaurant_dict}
    return train_departure_dict, train_back_dict, pois

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
    poi_data: dict
    intra_city_trans: dict
    use_disj: bool
    _cid: int ##用于标识
    def __init__(self,cross_city_train_departure, cross_city_train_back,poi_data,intra_city_trans,ir,model = None):
        if not model:
            self.model = pyo.ConcreteModel()
        else: self.model = model

        self.cross_city_train_departure = cross_city_train_departure
        self.cross_city_train_back = cross_city_train_back
        self.poi_data = poi_data
        self.intra_city_trans = intra_city_trans
        self.ir = ir
        self._cid = 0
        self.use_disj = False



    def get_daily_total_time(self,day):
        activity_time = sum(
            self.model.select_attr[day, a] * self.model.attr_data[a]['duration']
            for a in self.model.attractions
        ) + sum(
            self.model.select_rest[day, r] * (self.model.rest_data[r]['duration'] + self.model.rest_data[r]['queue_time'])
            for r in self.model.restaurants
        )
        if self.ir.travel_days > 1:
            trans_time = sum(
                self.model.poi_poi[day, p1, p2] * (
                        (1 - self.model.trans_mode[day]) * (
                        get_trans_params(self.intra_city_trans, p1, p2, 'taxi_duration')
                ) + \
                        self.model.trans_mode[day] * (
                                get_trans_params(self.intra_city_trans, p1, p2, 'bus_duration')
                        )
                )
                for p1 in self.model.pois
                for p2 in self.model.pois
            )
        else:
            trans_time = 0

        return activity_time + trans_time

    def get_daily_total_attraction_time(self,day):
        return sum(
            self.model.select_attr[day, a] * self.model.attr_data[a]['duration']
            for a in self.model.attractions
        )
    def get_daily_total_rating(self,day):
        sum_rating = sum(
            self.model.select_attr[day, a] * self.model.attr_data[a]['rating']
            for a in self.model.attractions
        ) + sum(
            self.model.select_rest[day, r] * self.model.rest_data[r]['rating']
            for r in self.model.restaurants
        )
        if day != self.ir.travel_days:
            sum_rating += sum(
                self.model.select_hotel[day, h] * self.model.hotel_data[h]['rating']
                for h in self.model.accommodations
            )
        return sum_rating
    
    def get_daily_attraction_rating(self,day):
        return sum(
            self.model.select_attr[day, a] * self.model.attr_data[a]['rating']
            for a in self.model.attractions
        )
    
    def get_daily_restaurant_rating(self,day):
        return sum(
            self.model.select_rest[day, r] * self.model.rest_data[r]['rating']
            for r in self.model.restaurants
        )
    
    def get_daily_hotel_rating(self,day):
        if day == self.ir.travel_days:
            return 0
        return sum(
            self.model.select_hotel[day, h] * self.model.hotel_data[h]['rating']
            for h in self.model.accommodations
        )

    def get_daily_queue_time(self,day):
        return sum(
            self.model.select_rest[day, r] * self.model.rest_data[r]['queue_time']
            for r in self.model.restaurants
        )
    
    def get_daily_total_restaurant_time(self,day):
        return sum(
            self.model.select_rest[day, r] * self.model.rest_data[r]['duration']
            for r in self.model.restaurants
        )
    
    def get_daily_total_transportation_time(self,day):
        return sum(
                self.model.poi_poi[day, p1, p2] * (
                        (1 - self.model.trans_mode[day]) * (
                        get_trans_params(self.intra_city_trans, p1, p2, 'taxi_duration')
                ) + \
                        self.model.trans_mode[day] * (
                                get_trans_params(self.intra_city_trans, p1, p2, 'bus_duration')
                        )
                )
                for p1 in self.model.pois
                for p2 in self.model.pois
            ) if self.ir.travel_days > 1 else 0
    
    def get_daily_total_cost(self,day):
        ## 景点，酒店，交通，吃饭，高铁, 人数
        peoples = self.ir.peoples
        attraction_cost = sum(
            self.model.select_attr[day, a] * self.model.attr_data[a]['cost']
            for a in self.model.attractions
        )
        if day == self.ir.travel_days:
            hotel_cost = 0
        else:
            hotel_cost = sum(
                self.model.select_hotel[day, h] * self.model.hotel_data[h]['cost'] * self.cfg.rooms_per_night
                for h in self.model.accommodations
            )

        transport_cost = sum(
            self.model.poi_poi[day, p1, p2] * (
                    (1 - self.model.trans_mode[day]) * ((peoples) / 4 + int(peoples % 4 > 0) ) * (
                    get_trans_params(self.intra_city_trans, p1, p2, 'taxi_cost') 
                    )
                )   + \
            self.model.poi_poi[day, p1, p2] * (
                        self.model.trans_mode[day] * peoples * (
                        get_trans_params(self.intra_city_trans, p1, p2, 'bus_cost') 
                        )
                    ) 
            for p1 in self.model.pois
            for p2 in self.model.pois
        ) if self.ir.travel_days > 1 else 0

        restaurant_cost = sum(
            self.model.select_rest[day, r] * self.model.rest_data[r]['cost']
            for r in self.model.restaurants
        )
        train_cost = 0
        if day == 1:
            train_cost += sum(self.model.select_train_departure[t] * self.model.train_departure_data[t]['cost']
                               for t in self.model.train_departure)
        elif day == self.ir.travel_days:
            train_cost += sum(self.model.select_train_back[t] * self.model.train_back_data[t]['cost']
                               for t in self.model.train_back)
        
        return transport_cost + hotel_cost + peoples * (attraction_cost + restaurant_cost + train_cost)

    def get_daily_total_restaurant_cost(self,day):
        peoples = self.ir.peoples
        return sum(
            self.model.select_rest[day, r] * self.model.rest_data[r]['cost'] * peoples
            for r in self.model.restaurants
        )
    
    def get_daily_total_attraction_cost(self,day):
        peoples = self.ir.peoples
        return sum(
            self.model.select_attr[day, a] * self.model.attr_data[a]['cost'] * peoples
            for a in self.model.attractions
        )

    def get_daily_total_hotel_cost(self,day):
        if day == self.ir.travel_days:
            return 0
        return sum(
            self.model.select_hotel[day, h] * self.model.hotel_data[h]['cost'] * self.cfg.rooms_per_night
            for h in self.model.accommodations
        )

    def get_daily_total_transportation_cost(self,day):
        if self.ir.travel_days <= 1:
            return 0
        peoples = self.ir.peoples
        transport_cost = sum(
            self.model.poi_poi[day, p1, p2] * (
                    (1 - self.model.trans_mode[day]) * ((peoples) / 4 + int(peoples % 4 > 0) ) * (
                    get_trans_params(self.intra_city_trans, p1, p2, 'taxi_cost') 
                    )
                )   + \
            self.model.poi_poi[day, p1, p2] * (
                        self.model.trans_mode[day] * peoples * (
                        get_trans_params(self.intra_city_trans, p1, p2, 'bus_cost') 
                        )
                    ) 
            for p1 in self.model.pois
            for p2 in self.model.pois
        )
        train_cost = 0
        if day == 1:
            train_cost += sum(self.model.select_train_departure[t] * self.model.train_departure_data[t]['cost']
                               for t in self.model.train_departure)
        elif day == self.ir.travel_days:
            train_cost += sum(self.model.select_train_back[t] * self.model.train_back_data[t]['cost']
                               for t in self.model.train_back)
            
        return transport_cost + peoples * train_cost
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
            for day in range(self.ir.travel_days):
                sum_time += self.get_daily_total_time(day + 1) 
            return sum_time  
        elif field == 'total_transportation_time':
            sum_time = 0
            for day in range(self.ir.travel_days):
                sum_time += self.get_daily_total_transportation_time(day + 1) 
            return sum_time
        elif field == 'total_queue_time':
            sum_time = 0
            for day in range(self.ir.travel_days):
                sum_time += self.get_daily_queue_time(day + 1) 
            return sum_time
        elif field == 'total_restaurant_time':
            sum_time = 0
            for day in range(self.ir.travel_days):
                sum_time += self.get_daily_total_restaurant_time(day + 1) 
            return sum_time
        elif field == 'total_cost': 
            return sum(self.get_daily_total_cost(day + 1) for day in range(self.ir.travel_days)) 
        elif field == 'total_hotel_cost':
            return sum(self.get_daily_total_hotel_cost(day + 1) for day in range(self.ir.travel_days))
        elif field == 'total_attraction_cost':
            return sum(self.get_daily_total_attraction_cost(day + 1) for day in range(self.ir.travel_days))
        elif field == 'total_restaurant_cost':
            return sum(self.get_daily_total_restaurant_cost(day + 1) for day in range(self.ir.travel_days))
        elif field == 'total_transportation_cost':
            return sum(self.get_daily_total_transportation_cost(day + 1) for day in range(self.ir.travel_days))
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
    def make(self, cfg: dynamic_constraint):
        self.cfg = cfg
        pois = [a_id for a_id in self.poi_data['attractions']] + [h_id for h_id in self.poi_data['accommodations']]

        attraction_dict = self.poi_data['attractions'] ## {'attractions':{'id_1':{...},'id_2':{...},...}}
        hotel_dict = self.poi_data['accommodations']
        restaurant_dict = self.poi_data['restaurants']
        
        days = range(1,self.ir.travel_days + 1)
        self.model.days = pyo.Set(initialize=days)
        self.model.attractions = pyo.Set(initialize=attraction_dict.keys())
        self.model.accommodations = pyo.Set(initialize=hotel_dict.keys())
        self.model.restaurants = pyo.Set(initialize=restaurant_dict.keys())
        self.model.train_departure = pyo.Set(initialize=self.cross_city_train_departure.keys())
        self.model.train_back = pyo.Set(initialize=self.cross_city_train_back.keys())
        self.model.pois = pyo.Set(initialize=pois)

        self.model.attr_data = pyo.Param(
            self.model.attractions,
            initialize=lambda m, a: {
                'id': attraction_dict[a]['id'],
                'name': attraction_dict[a]['name'],
                'cost': float(attraction_dict[a]['cost']),
                'type': attraction_dict[a]['type'],
                'rating': float(attraction_dict[a]['rating']),
                'duration': float(attraction_dict[a]['duration'])
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
                'feature': hotel_dict[h]['feature']
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
                'duration': float(restaurant_dict[r]['duration'])
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

        ## variables
        self.model.select_hotel = pyo.Var(self.model.days, self.model.accommodations, domain=pyo.Binary)
        self.model.select_attr = pyo.Var(self.model.days, self.model.attractions, domain=pyo.Binary)
        self.model.select_rest = pyo.Var(self.model.days, self.model.restaurants, domain=pyo.Binary)
        self.model.trans_mode = pyo.Var(self.model.days, domain=pyo.Binary) # 1为公交 0为打车
        self.model.select_train_departure = pyo.Var(self.model.train_departure, domain=pyo.Binary)
        self.model.select_train_back = pyo.Var(self.model.train_back, domain=pyo.Binary)

        ## last day hotel constraint
        if self.ir.travel_days > 1:
            def last_day_hotel_constraint(model,h):
                N = self.ir.travel_days
                return model.select_hotel[N-1,h] == model.select_hotel[N,h]
            
            self.model.last_day_hotel = pyo.Constraint(
                self.model.accommodations,
                rule=last_day_hotel_constraint
            )

        self.model.poi_poi = pyo.Var(
            self.model.days, self.model.pois, self.model.pois,
            domain=pyo.Binary,
            initialize=0,
            bounds=(0, 1)
        )

        ## 一致性约束
        def self_loop_constraint(model,d, p):
            return model.poi_poi[d, p, p] == 0
        

        self.model.self_loop = pyo.Constraint(
            self.model.days,self.model.pois,
            rule=self_loop_constraint
        )

        self.model.u = pyo.Var(self.model.days, self.model.attractions, domain=pyo.NonNegativeReals) ##描述景点的顺序
        ## join
        def a_degree_constraint_out(model,d,a):
            return sum(model.poi_poi[d, a, p] for p in model.pois) == model.select_attr[d, a]
        
        def a_degree_constraint_in(model,d,a):
            return sum(model.poi_poi[d, p, a] for p in model.pois) == model.select_attr[d, a]
        
        def h_degree_constraint_out(model,d,h):
            return sum(model.poi_poi[d, h, p] for p in model.pois) == model.select_hotel[d, h]
        
        def h_degree_constraint_in(model,d,h):
            return sum(model.poi_poi[d, p, h] for p in model.pois) == model.select_hotel[d, h]
        
        def mtz_rule(m,d,i,j):
            M = len(self.model.attractions)
            if i == j:
                return pyo.Constraint.Skip
            return m.u[d, i] - m.u[d, j] + M * m.poi_poi[d, i, j] <= M - 1

        def u_rule_low(m,d,p):
            M = len(self.model.attractions)
            return m.select_attr[d,p] <= m.u[d,p] 
        def u_rule_high(m,d,p):
            M = len(self.model.attractions)
            return m.u[d,p] <= M * m.select_attr[d, p]
        
        self.model.a_degree_constraint_out = pyo.Constraint(
            self.model.days, self.model.attractions,
            rule=a_degree_constraint_out
        )
        self.model.a_degree_constraint_in = pyo.Constraint(
            self.model.days, self.model.attractions, 
            rule=a_degree_constraint_in
        )
        self.model.h_degree_constraint_out = pyo.Constraint(
            self.model.days, self.model.accommodations, 
            rule=h_degree_constraint_out
        )
        self.model.h_degree_constraint_in = pyo.Constraint(
            self.model.days, self.model.accommodations, 
            rule=h_degree_constraint_in
        )
        self.model.mtz = pyo.Constraint(
            self.model.days, self.model.attractions, self.model.attractions,
            rule=mtz_rule
        )
        self.model.u_rule_low = pyo.Constraint(
            self.model.days, self.model.attractions,
            rule=u_rule_low
        )
        self.model.u_rule_high = pyo.Constraint(
            self.model.days, self.model.attractions,
            rule=u_rule_high
        )
        
        self.model.unique_attr = pyo.Constraint(
            self.model.attractions,
            rule=lambda m, a: sum(m.select_attr[d, a] for d in m.days) <= 1
        )

        self.model.unique_rest = pyo.Constraint(
            self.model.restaurants,
            rule=lambda m, r: sum(m.select_rest[d, r] for d in m.days) <= 1
        )
        if not cfg.change_hotel:
            def same_hotel_rule(m, d, h):
                if d == 1:
                    return pyo.Constraint.Skip
                return m.select_hotel[d, h] == m.select_hotel[d-1, h]
            self.model.same_hotel = pyo.Constraint(self.model.days, self.model.accommodations, rule=same_hotel_rule)

        if len(self.cross_city_train_departure) > 0:
            self.model.one_departure = pyo.Constraint(
                rule=lambda m: sum(m.select_train_departure[t] for t in m.train_departure) == 1
            )
        if len(self.cross_city_train_back) > 0:
            self.model.one_back = pyo.Constraint(
                rule=lambda m: sum(m.select_train_back[t] for t in m.train_back) == 1
            )
        ##约束1
        if cfg.num_attractions_per_day:
            self.ast_to_pyomo_constraints(self.model,cfg.num_attractions_per_day,{'index_names':['day']},"num_attr_per_day",[self.model.days])
        else:
            self.model.attr_num = pyo.Constraint(
                self.model.days,
                rule=lambda m, d: sum(m.select_attr[d, a] for a in m.attractions) == 1
            )

        if cfg.num_restaurants_per_day:
            self.ast_to_pyomo_constraints(self.model,cfg.num_restaurants_per_day,{'index_names':['day']},"num_rest_per_day",[self.model.days])
        else:
            self.model.rest_num = pyo.Constraint(
                self.model.days,
                rule=lambda m, d: sum(m.select_rest[d, r] for r in m.restaurants) == 3
            )
        
        if cfg.num_hotels_per_day:
            self.ast_to_pyomo_constraints(self.model,cfg.num_hotels_per_day,{'index_names':['day']},"num_hotels_per_day",[self.model.days])
        else:
            self.model.hotel_num = pyo.Constraint(
                self.model.days,
                rule=lambda m, d:  sum(m.select_hotel[d, h] for h in m.accommodations) == 1
            )

        ##约束2
        if cfg.infra_city_transportation == 'public_transportation':
            self.model.trans_mode_rule = pyo.Constraint(
                self.model.days,
                rule=lambda m, d: m.trans_mode[d] == 1
            )
        elif cfg.infra_city_transportation == 'taxi':
            self.model.trans_mode_rule = pyo.Constraint(
                self.model.days,
                rule=lambda m, d: m.trans_mode[d] == 0
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
        solver.options['limits/time'] = 300
        solver.options['limits/gap'] = 0.0
        return solver
    
    def solve(self):
        solver = self.configure_solver()
        results = solver.solve(self.model, tee=True)
        return results
    

    def generate_date_range(self, start_date, date_format="%Y年%m月%d日"):
        start = datetime.strptime(start_date, date_format)
        days = self.ir.travel_days
        return [
            (start + timedelta(days=i)).strftime(date_format)
            for i in range(days)
        ]


    def get_selected_train(self,train_type='departure'):
        model = self.model
        if train_type not in ['departure', 'back']:
            raise ValueError("train_type must in ['departure', 'back']")

        train_set = model.train_departure if train_type == 'departure' else model.train_back
        train_data = model.train_departure_data if train_type == 'departure' else model.train_back_data
        selected_train = [
            train_data[t]
            for t in train_set
            if pyo.value(
                model.select_train_departure[t] if train_type == 'departure'
                else model.select_train_back[t]
            ) > 0.9
        ]
        return selected_train[0] if selected_train else 'null'


    def get_selected_poi(self, type, day, selected_poi):
        model = self.model
        if type == 'restaurant':
            poi_set = model.restaurants
            poi_data = model.rest_data
            select_set = model.select_rest
        else:
            poi_set = model.attractions
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
        selected_hotel = [
            model.hotel_data[t]
            for t in model.accommodations
            if pyo.value(model.select_hotel[day,t]) > 0.9
        ]
        return selected_hotel[0] if selected_hotel else 'null'


    def get_time(self, selected_attr, selected_rest, selected_hotel, day):
        intra_city_trans = self.intra_city_trans
        model = self.model
        daily_time = 0
        transport_time = 0
        for a in selected_attr:
            daily_time += a['duration']
        for r in selected_rest:
            daily_time += r['queue_time'] +r['duration']

        if self.ir.travel_days > 1 and len(selected_attr) > 0:
            ## 提取出景点顺序
            order = sorted(selected_attr, key=lambda a: model.u[day,a['id']].value)

            if pyo.value(model.trans_mode[day]) > 0.9:
                transport_time = get_trans_params(
                    intra_city_trans,
                    selected_hotel['id'],
                    order[0]['id'],
                    'bus_duration'
                ) + get_trans_params(
                    intra_city_trans,
                    order[-1]['id'],
                    selected_hotel['id'],
                    'bus_duration'
                )
                for i in range(len(order) - 1):
                    transport_time += get_trans_params(
                    intra_city_trans,
                    order[i]['id'],
                    order[i + 1]['id'],
                    'bus_duration'
                )
            else:
                transport_time = get_trans_params(
                    intra_city_trans,
                    selected_hotel['id'],
                    order[0]['id'],
                    'taxi_duration'
                ) + get_trans_params(
                    intra_city_trans,
                    order[-1]['id'],
                    selected_hotel['id'],
                    'taxi_duration'
                )
                for i in range(len(order) - 1):
                    transport_time += get_trans_params(
                    intra_city_trans,
                    order[i]['id'],
                    order[i + 1]['id'],
                    'taxi_duration'
                )

        return daily_time + transport_time, transport_time


    def get_cost(self, selected_attr, selected_rest, departure_trains, back_trains, selected_hotel, day):

        model = self.model
        intra_city_trans = self.intra_city_trans
        daily_cost = 0
        transport_cost = 0
        peoples = self.ir.peoples
        for a in selected_attr:
            daily_cost += peoples * a['cost']
        travel_days = self.ir.travel_days
        for r in selected_rest:
            if r:
                daily_cost += peoples * r['cost']
        if self.ir.travel_days > 1 and len(selected_attr) > 0:
            order = sorted(selected_attr, key=lambda a: model.u[day,a['id']].value)
            if pyo.value(model.trans_mode[day]) > 0.9:
                transport_cost = get_trans_params(
                    intra_city_trans,
                    selected_hotel['id'],
                    order[0]['id'],
                    'bus_cost'
                ) + get_trans_params(
                    intra_city_trans,
                    order[-1]['id'],
                    selected_hotel['id'],
                    'bus_cost'
                )
                for i in range(len(order) - 1):
                    transport_cost += get_trans_params(
                    intra_city_trans,
                    order[i]['id'],
                    order[i + 1]['id'],
                    'bus_cost'
                )
                transport_cost = peoples * transport_cost
            else:
                transport_cost = get_trans_params(
                    intra_city_trans,
                    selected_hotel['id'],
                    order[0]['id'],
                    'taxi_cost'
                ) + get_trans_params(
                    intra_city_trans,
                    order[-1]['id'],
                    selected_hotel['id'],
                    'taxi_cost'
                )
                for i in range(len(order) - 1):
                    transport_cost += get_trans_params(
                    intra_city_trans,
                    order[i]['id'],
                    order[i + 1]['id'],
                    'taxi_cost'
                )
                transport_cost = ((peoples) / 4 + int(peoples % 4 > 0) ) * transport_cost

        if day != travel_days:
            daily_cost += selected_hotel['cost'] * self.cfg.rooms_per_night
        if day == 1 and isinstance(departure_trains,dict):
            daily_cost += peoples * departure_trains['cost']
        if day == travel_days and isinstance(back_trains,dict):
            daily_cost += peoples * back_trains['cost']

        return daily_cost + transport_cost, transport_cost

    def generate_daily_plan(self):
        model = self.model
        intra_city_trans = self.intra_city_trans
        departure_trains = self.get_selected_train('departure')
        back_trains = self.get_selected_train('back')
        total_cost = 0
        daily_plans = []
        select_at = []
        select_re = []
        travel_days = self.ir.travel_days
        date = self.generate_date_range(self.ir.start_date) ##todo
        for day in range(1, travel_days + 1):
            attr_details = []
            attr_details = self.get_selected_poi('attraction', day, select_at)
            select_at += [a['id'] for a in attr_details]
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
            daily_time, transport_time = self.get_time(attr_details, rest_details, selected_hotel, day)
            daily_cost, transport_cost = self.get_cost(attr_details, rest_details, departure_trains, back_trains, selected_hotel, day)
            if len(attr_details) == 1:
                attr_details = attr_details[0]
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

        return { #todo
            "budget": self.ir.budgets,
            "peoples": self.ir.peoples,
            "travel_days": travel_days,
            "origin_city": self.ir.original_city,
            "destination_city": self.ir.destinate_city,
            "start_date": self.ir.start_date,
            "end_date": date[-1],
            "daily_plans": daily_plans,
            "departure_trains": departure_trains,
            "back_trains": back_trains,
            "total_cost": round(total_cost, 2),
            "objective_value": round(pyo.value(model.obj), 2)
        }
    
def get_solution(omo:template):
    if omo.use_disj:
        TransformationFactory('gdp.bigm').apply_to(omo.model)
    solver = omo.configure_solver()
    results = solver.solve(omo.model, tee=True)
    plan = omo.generate_daily_plan()
    print(f"```generated_plan\n{plan}\n```")



#Todo 目标函数 + LLM交互文件

if __name__ == '__main__':
    cross_city_train_departure = {}
    cross_city_train_back = {}
    poi_data = {'attractions': {}, 'accommodations': {}, 'restaurants': {}}
    intra_city_trans = {}

    ###########################IR && dynamic_constraint#############################



    ##########################################################

    try:
        cross_city_train_departure, cross_city_train_back, poi_data, intra_city_trans = fetch_data(ir)
        for item in cross_city_train_departure:
            item['cost'] = float(item['cost'])
            item['duration'] = float(item['duration'])

        for item in cross_city_train_back:
            item['cost'] = float(item['cost'])
            item['duration'] = float(item['duration'])

        for item in poi_data['attractions']:
            item['rating'] = float(item['rating'])
            item['cost'] = float(item['cost'])
            item['duration'] = float(item['duration'])
        
        for item in poi_data['accommodations']:
            item['rating'] = float(item['rating'])
            item['cost'] = float(item['cost'])

        for item in poi_data['restaurants']:
            item['rating'] = float(item['rating'])
            item['cost'] = float(item['cost'])
            item['duration'] = float(item['duration'])
            item['queue_time'] = float(item['queue_time'])
        # fetch 数据
        cross_city_train_departure, cross_city_train_back, poi_data = rough_rank(cross_city_train_departure=cross_city_train_departure,cross_city_train_back=cross_city_train_back,poi_data=poi_data,ir=ir)
        # 粗排: todo
        tp = template(cross_city_train_departure=cross_city_train_departure,cross_city_train_back=cross_city_train_back,poi_data=poi_data,intra_city_trans=intra_city_trans,ir=ir)
        tp.make(dc)
        # 构造template
        # make
        get_solution(tp)
        # get_solution
    except Exception as e:
        def generate_date_range(start_date, date_format="%Y年%m月%d日"):
            start = datetime.strptime(start_date, date_format)
            days = ir.travel_days
            return [
                (start + timedelta(days=i)).strftime(date_format)
                for i in range(days)
            ]
        date = generate_date_range(ir.start_date)
        departure_trains = back_trains = 'null'
        if isinstance(cross_city_train_departure, dict) and cross_city_train_departure:
            first_key = next(iter(cross_city_train_departure))
            departure_trains = cross_city_train_departure[first_key]
        if isinstance(cross_city_train_back, dict) and cross_city_train_back:
            first_key = next(iter(cross_city_train_back))
            back_trains = cross_city_train_back[first_key]
        daily_plans = []
        selected_hotel = 'null'
        attr_details = 'null'
        meal_allocation = {
            'breakfast': 'null',
            'lunch': 'null',
            'dinner': 'null'
        }
        for day in range(1, ir.travel_days + 1):
            
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
                "hotel": selected_hotel if day != ir.travel_days else "null",
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

        plan = {
            "budget": ir.budgets,
            "peoples": ir.peoples,
            "travel_days": ir.travel_days,
            "origin_city": ir.original_city,
            "destination_city": ir.destinate_city,
            "start_date": ir.start_date,
            "end_date": date[-1],
            "daily_plans": daily_plans,
            "departure_trains": departure_trains,
            "back_trains": back_trains,
            "total_cost": 0,
            "objective_value": 0
        }
        print(f"```generated_plan\n{plan}\n```")