from prompt_system import PromptSystem
from base import LLM
import asyncio
from IR_multi_stage import *
from dataclasses import asdict
import json,re
from base import Config
import os
from tqdm import tqdm

prompt_sys = PromptSystem.getSingleton() 
static_prompt = prompt_sys.getPrompt('static')
dynamic_prompt = prompt_sys.getPrompt('dynamic')
objective_prompt = prompt_sys.getPrompt('objective')
# llm_static = LLM('xop3qwen235b2507',system_prompt=static_prompt)
# llm_dynamic = LLM('xop3qwen235b2507',system_prompt=dynamic_prompt)
# llm_objective = LLM('xop3qwen235b2507',system_prompt=objective_prompt)
llm_static = LLM('deepseek-v3.1',system_prompt=static_prompt)
llm_dynamic = LLM('deepseek-v3.1',system_prompt=dynamic_prompt)
llm_objective = LLM('deepseek-v3.1',system_prompt=objective_prompt)
code_config = []
problems = {}

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
        peoples=data["peoples"],
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
    return dynamic_constraint(
        # 基础字段
        num_travlers=int(data["num_travlers"]),
        children_num=int(data["children_num"]),
        rooms_per_night=int(data["rooms_per_night"]),
        multi_stage=data["multi_stage"],
        peoples_per_car = data["peoples_per_car"],
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
        "num_travlers", "rooms_per_night","children_num","multi_stage", "peoples_per_car",
        # 时间相关
        "daily_total_time", "daily_queue_time", "daily_total_meal_time", "daily_transportation_time",
        "total_active_time", "total_queue_time", "total_restaurant_time", "total_transportation_time",
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

ir_json = """
  {
    "start_date": "2025年6月10日",
    "peoples": 2,
    "total_travel_days": 9,
    "children_num": 1,
    "budgets": 7000,
    "stages" :  [ 
        {
            "original_city": "深圳",
            "destinate_city": "上海",
            "travel_days" : 3,
            "attraction_constraints": {
            "type": "op",
            "op": "and",
            "left": {
                "type": "op",
                "op": ">=",
                "left": {"type": "field", "field": "rating"},
                "right": {"type": "value", "value": 4.5}
            },
            "right": {
                "type": "op",
                "op": "<=",
                "left": {"type": "field", "field": "cost"},
                "right": {"type": "value", "value": 800}
            }
            },
            "accommodation_constraints": {
            "type": "op",
            "op": "or",
            "left": {
                "type": "op",
                "op": ">=",
                "left": {"type": "field", "field": "rating"},
                "right": {"type": "value", "value": 4.5}
            },
            "right": {
                "type": "op",
                "op": "<=",
                "left": {"type": "field", "field": "cost"},
                "right": {"type": "value", "value": 800}
            }
            },
            "restaurant_constraints": null
        },
        {
            "original_city": "上海",
            "destinate_city": "深圳",
            "travel_days" : 6,
            "attraction_constraints": null,
            "accommodation_constraints":   {
                "type": "op",
                "op": "or",
                "left":{
                "type": "op",
                "op": "include",
                "left": {
                    "type": "field",
                    "field": "feature"
                },
                "right": {
                    "type": "value",
                    "value": "早餐"
                }
                },
                "right": {
                "type": "op",
                "op": "include",
                "left": {
                    "type": "field",
                    "field": "feature"
                },
                "right": {
                    "type": "value",
                    "value": "早点"
                }
                }
            },
            "restaurant_constraints": null
        }
    ],
    "departure_transport_constraints": {
      "type": "op",
      "op": "==",
      "left": {"type": "field", "field": "origin_station"},
      "right": {"type": "value", "value": "南京南站"}
    },
    "back_transport_constraints": null,
    "intermediate_transport_constraints": null
  }
"""
# ir = ir_from_json(ir_json)
# print(ir_to_json(ir))
dc = """  {
    "num_travlers": 2,
    "rooms_per_night": 1,
    "children_num" : 1,
    "multi_stage": true,
    "daily_total_time": {
      "type": "op",
      "op": "<=",
      "left": { "type": "field", "field": "daily_total_time" },
      "right": { "type": "value", "value": 840 }
    },
    "daily_queue_time": null,
    "daily_total_restaurant_time": {
      "type": "op",
      "op": "<=",
      "left": { "type": "field", "field": "daily_total_restaurant_time" },
      "right": { "type": "value", "value": 120 }
    },
    "daily_transportation_time": {
      "type": "op",
      "op": "<=",
      "left": { "type": "field", "field": "daily_transportation_time" },
      "right": { "type": "value", "value": 150 }
    },
    "total_active_time": null,
    "total_queue_time": null,
    "total_restaurant_time": null,
    "total_transportation_time": {
      "type": "op",
      "op": "<=",
      "left": { "type": "field", "field": "total_transportation_time" },
      "right": { "type": "value", "value": 600 }
    },

    "num_attractions_per_day": {
      "type": "op",
      "op": "==",
      "left": { "type": "field", "field": "num_attractions_per_day" },
      "right": { "type": "value", "value": 1 }
    },
    "num_restaurants_per_day": {
      "type": "op",
      "op": "==",
      "left": { "type": "field", "field": "num_restaurants_per_day" },
      "right": { "type": "value", "value": 3 }
    },
    "num_hotels_per_day": {
      "type": "op",
      "op": "==",
      "left": { "type": "field", "field": "num_hotels_per_day" },
      "right": { "type": "value", "value": 1 }
    },

    "infra_city_transportation": {"深圳市": "public_transportation"},

    "total_budget": null,
    "total_meal_budget": null,
    "total_attraction_ticket_budget": null,
    "total_hotel_budget": null,
    "total_transportation_budget": null,
    "daily_total_budget": {
      "type": "op",
      "op": "<=",
      "left": { "type": "field", "field": "daily_total_cost" },
      "right": { "type": "value", "value": 2000 }
    },
    "daily_total_meal_budget": null,
    "daily_total_attraction_ticket_budget": {
      "type": "op",
      "op": "<=",
      "left": { "type": "field", "field": "daily_total_attraction_cost" },
      "right": { "type": "value", "value": 300 }
    },
    "daily_total_hotel_budget": null,
    "daily_total_transportation_budget": {
      "type": "op",
      "op": "<=",
      "left": { "type": "field", "field": "daily_transportation_cost" },
      "right": { "type": "value", "value": 350 }
    },

    "extra": "
  # 额外动态约束示例：行程中至少包含一家博物馆相关景点(想去某某景点，想吃某某菜品，等等可以参考指南)
  self.model.must_have_museum = pyo.Constraint(
      rule=lambda m: sum(
          self.model.select_attr[d, a]
          for d in self.model.days
          for a in self.model.attractions
          if ('博物馆' in self.model.attr_data[a]['type']) >= 1
  )"
  }
"""

# print(dynamic_constraint_to_dict(dc_ret))
def _escape_newlines_inside_strings(s: str) -> str:
    """
    在 JSON 文本里，仅在**字符串常量**内部，把裸 '\n' 和 '\r' 转义成 '\\n' 和 '\\r'。
    不改变字符串外部的换行（合法的空白）。
    """
    out = []
    in_str = False
    escape = False
    for ch in s:
        if escape:
            out.append(ch)
            escape = False
            continue
        if ch == "\\":
            out.append(ch)
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
            out.append(ch)
            continue
        if in_str and ch == "\n":
            out.append("\\n")
        elif in_str and ch == "\r":
            out.append("\\r")
        else:
            out.append(ch)
    return "".join(out)


def extract_json_block(text: str) -> Optional[str]:
    """
    提取形如 \"\"\"json ... \"\"\" 的字符串块，返回最后一个匹配项
    """
    try:
        x = json.loads(text)
        if not isinstance(x,dict) and not isinstance(x,list):
            raise ValueError('output is not a dict or list')
        return text
    except:
        try:
            pattern = r'(?<=```json)(.*?)(?=```)'   # 非贪婪匹配中间内容
            matches = re.findall(pattern, text,re.DOTALL)
            if matches:
                ret = matches[-1].strip()
                ret = _escape_newlines_inside_strings(ret)
                json.loads(ret)
                return ret
            return None
        except Exception as e:
            return None
        

async def get_llm_result_parallel(problem_id,write_log = False):
    problem = problems.get(str(problem_id))
    tasks = [
        llm_static.invoke(f'用户的问题是：{problem},请冷静下来，一步一步仔细思考，给出一个最合适的答案。'),
        llm_dynamic.invoke(f'用户的问题是：{problem},请冷静下来，一步一步仔细思考，给出一个最合适的答案。'),
        llm_objective.invoke(f'用户的问题是：{problem},请冷静下来，一步一步仔细思考，给出一个最合适的答案。')
    ]
    results = await asyncio.gather(*tasks)
    type_map = {
        "static" : llm_static,
        "dynamic" : llm_dynamic,
        "objective" : llm_objective
    }
    error_block = results
    error_info = ['static','dynamic', 'objective']
    def valid(tp, info):
        if tp == 'objective':
            pattern = r'(?<=```python)(.*?)(?=```)'   # 非贪婪匹配中间内容
            matches = re.findall(pattern, info ,re.DOTALL)
            return matches[-1].strip() if matches else None
        else:
            ret = extract_json_block(info)
            return ret
    result_json = {}
    raw_result = {}
    while True:
        tmp_error = []
        for idx,tp in enumerate(error_info):
            ret = valid(tp, error_block[idx])
            if not ret:
                tmp_error.append(tp)
            else:
                result_json[tp] = ret
                raw_result[tp] = error_block[idx]
        if tmp_error == []:
            if write_log:
                log_path = os.path.join(Config.get_global_config().config['log_path'], f'{problem_id}.log')
                with open(log_path, 'w', encoding='utf-8') as f:
                    f.write('#' * 50 + 'static result' + '#' * 50 + '\n\n')
                    f.write(f"{raw_result['static']}\n\n")
                    f.write('#' * 50 + 'dynamic result' + '#' * 50 + '\n\n')
                    f.write(f"{raw_result['dynamic']}\n\n")
                    f.write('#' * 50 + 'objective result' + '#' * 50 + '\n\n')
                    f.write(f"{raw_result['objective']}\n\n")
            return result_json['static'], result_json['dynamic'], result_json['objective']
        error_info = tmp_error
        print(f'以下类型无法解析出正确结果，待重试 -----------> {error_info}')
        tasks = [type_map[tp].invoke(f'用户的问题是：{problem},请冷静下来，一步一步仔细思考，给出一个最合适的答案。') for tp in error_info]
        error_block = await asyncio.gather(*tasks)


def create_code_file(template_file_path, code_file_path,insert_code:str, extra_code:str):
    
    indent = Config.get_global_config().config['indent']
    lineno = Config.get_global_config().config['lineno']
    extra_lineno = Config.get_global_config().config['extra_lineno']
    extra_indent = Config.get_global_config().config['extra_indent']

    indent_str = ""
    extra_indent_str = ""

    for line in insert_code.splitlines():
        if line.strip() == "":  # 跳过空行
            indent_str += "\n"
        else:
            indent_str += indent + line + "\n"
    for line in extra_code.splitlines():
        if line.strip() == "":  # 跳过空行
            extra_indent_str += "\n"
        else:
            extra_indent_str += extra_indent + line + "\n"
    
    with open(template_file_path, 'r', encoding='utf-8') as in_f, open(code_file_path, 'w', encoding='utf-8') as out_f:
        for idx,line in enumerate(in_f):
            if idx == lineno - 1:
                out_f.write(indent_str)
            elif idx == extra_lineno -1 :
                out_f.write(extra_indent_str)
            else:
                out_f.write(line)
    
    # print(f'成功创建代码文件:{code_file_path}')

def create_code(ir_json,dynamic_json,objective_code,user_problem = ""):
    json_data = json.loads(dynamic_json)
    if json_data['extra']:
        code = json_data['extra'] + '\n' + objective_code
    else:
        code = objective_code
    del json_data['extra']
    dynamic = json.dumps(json_data,indent=2,ensure_ascii=False)
    main_code = f"ir_data = \"\"\"{ir_json}\"\"\"\ndc_data = \"\"\"{dynamic}\"\"\" \nir = ir_from_json(ir_data)\ndc = dynamic_constraint_from_json(dc_data)\nuser_question = \"\"\"{user_problem}\"\"\""
    return main_code,code

async def worker(queue: asyncio.Queue, worker_id: int, tbar: tqdm):
    """工作协程：从队列获取任务并执行"""
    code_path = Config.get_global_config().config['code_path']
    template_file = Config.get_global_config().config['template_file']
    while True:
        # 从队列获取任务（如果队列为空则等待）
        problem = await queue.get()
        try:
            static,dynamic,objective = await get_llm_result_parallel(problem,write_log=True)
            main_code,extra_code = create_code(static,dynamic,objective,user_problem=problems[problem])
            code_file = f"{code_path}/id_{problem}.py"
            create_code_file(template_file_path=template_file,code_file_path=code_file,insert_code=main_code,extra_code=extra_code)
        except Exception as e:
            print(f"任务{problem}执行出错:{e}")
        finally:
            tbar.update(1)
            # 通知队列当前任务已完成（必须调用，否则队列会认为任务未处理）
            queue.task_done()
async def main():
    import time
    from tqdm import tqdm
    print('=======start=======')
    begin = time.time()
    problem_file = Config.get_global_config().config['problem_file']
    MAX_CONCURRENCY = Config.get_global_config().config['max_concurrency']

    break_point = 0
    end_num = 800
    with open(problem_file, 'r', encoding='utf-8') as f:
        json_problems = json.load(f)
        for item in json_problems:
            problems[item['question_id']] = item['question']
    print(f'初始化完成，成功导入数据集: {len(problems)}条')
    queue = asyncio.Queue(maxsize=MAX_CONCURRENCY)
    tbar = tqdm(total=len(problems), desc='进度')
    workers = []
    for i in range(MAX_CONCURRENCY):
        worker_task = asyncio.create_task(worker(queue, i + 1, tbar))
        workers.append(worker_task)

    for problem in problems:
        sample = {
            "question_id": problem,
            "question": problems[problem],
            "code_path": f'code/id_{problem}.py'
        }
        code_config.append(sample)
        if break_point > 0 and int(problem) <= break_point:
            tbar.update(1)
            time.sleep(0.1)
            continue
            
        await queue.put(problem)

        if int(problem) == end_num:
            break

    await queue.join()

    for worker_task in workers:
            worker_task.cancel()

    await asyncio.gather(*workers, return_exceptions=True)
    tbar.close()

    dump_file = Config.get_global_config().config['question_prompt']

    with open(dump_file, 'w', encoding='utf-8') as f:
        json.dump(code_config,f,indent=2,ensure_ascii=False)
    end = time.time()
    print('=======end=======')
    print(f'耗时:{end-begin}秒')

async def test_static():
    import time
    test_problem = "我计划于2025年08月28日从上海市出发，先前往洛阳市旅游3天，随后于2025年08月30日晚乘坐高铁前往广州市继续游玩6天，总预算31000元以内，尽可能减少通勤时间。在广州市期间，我会游览仁威祖庙、邓世昌纪念馆和珠江琶醍啤酒文化创意艺术区，并考虑入住维也纳国际酒店(广州白云站石井地铁站店)或美景大酒店。2025年08月28日上午从上海出发，2025年09月05日晚返回。"
    try:
        print('正在测试...')
        begin = time.time()
        ret = await llm_static.invoke(f'用户的问题是：{test_problem},请冷静下来，一步一步仔细思考，给出一个最合适的答案。')
        json_ret = extract_json_block(ret)
        ir = ir_from_json(str(json_ret))
        end = time.time()
        json_ret = json.loads(json_ret)
        print(f'测试结果：{json.dumps(json_ret,indent=2,ensure_ascii=False)}')
        print(f'耗时：{(end-begin):2f}s')
    except Exception as e:
        print(f'发生错误 {e} ,原始输出为\n{ret}')

async def test_dynamic():
    import time
    test_problem = "我计划于2025年11月03日上午从深圳市出发，先前往贵阳市进行5天4晚旅行，随后于2025年11月07日晚间乘坐高铁抵达广州市，继续开展7天7晚的行程，整个旅程共计12天11晚，总预算为82100元。在贵阳市的旅途中，行程中需有一餐含桑葚，并尽可能延长游玩时间、缩减排队时间。在广州市期间，我会游览广州塔，入住宜必思酒店(广州越秀公园地铁站店)，优先考虑排队时间短、能高效用餐的选项。所有旅行期间，市内交通全程采用公交。2025年11月14日晚间从广州市返程。"
    try:
        print('正在测试...')
        begin = time.time()
        ret = await llm_dynamic.invoke(f'用户的问题是：{test_problem},请冷静下来，一步一步仔细思考，给出一个最合适的答案。')
        json_ret = extract_json_block(ret)
        ir = dynamic_constraint_from_json(str(json_ret))
        end = time.time()
        json_ret = json.loads(json_ret)
        print(f'测试结果：{json.dumps(json_ret,indent=2,ensure_ascii=False)}')
        print(f'耗时：{(end-begin):2f}s')
    except Exception as e:
        print(f'发生错误 {e} ,原始输出为\n{ret}')

async def test_objective():
    import time
    test_problem = "我计划于2025年10月15日至19日从广州前往北京开展为期五天的高品质双人旅行，总预算为20000元，需满足以下需求：全程入住四星级及以上标准酒店，行程中须包含颐和园、恭王府博物馆等风景名胜，并安排一次正宗老北京烤鸭体验。15日早上从洛阳龙门站出发，19日晚返回洛阳。返程指定搭乘G651次列车。每日行程需兼顾热门景点与合理动线，避免过度奔波，市内交通以打车为主，整体行程注重舒适性与文化深度，尽量延长游玩时间、减少通勤和排队时间。"
    try:
        print('正在测试...')
        begin = time.time()
        ret = await llm_objective.invoke(f'用户的问题是：{test_problem},请冷静下来，一步一步仔细思考，给出一个最合适的答案。')
        pattern = r'(?<=```python)(.*?)(?=```)'   # 非贪婪匹配中间内容
        matches = re.findall(pattern, ret,re.DOTALL)
        result = ""
        if matches:
            result = matches[-1].strip()
        end = time.time()
        print(f'测试结果：\n{result}')
        print(f'耗时：{(end-begin):2f}s')
    except Exception as e:
        print(f'发生错误 {e} ,原始输出为\n{ret}')

async def test():
    import time
    begin = time.time()
    problem_file = Config.get_global_config().config['problem_file']
    code_path = Config.get_global_config().config['code_path']
    template_file = Config.get_global_config().config['template_file']

    with open(problem_file, 'r', encoding='utf-8') as f:
        json_problems = json.load(f)
        for item in json_problems:
            problems[item['question_id']] = item['question'] 
    problems['10086'] = "我计划于2025年08月02日从广州市出发，先前往重庆市旅游4天，再前往青岛市旅游4天，总预算为45800元。2025年08月02日上午出发，2025年08月05日晚乘高铁从重庆抵达青岛并入住青岛的欧尊格商务酒店，2025年08月09日晚返回。重庆旅行期间全程坚持公交出行，重点游览重庆长江索道景区；在青岛期间，全程坚持打车出行，我会游览嘉定山公园和冰山之角，就餐选择吉祥馄饨(市北水清沟店)等经济实惠的餐厅，每日就餐时间不超过60分钟，返程乘坐G2080_2次列车从青岛北站至广州北站；旅行全程控制餐饮、住宿及游玩开支."
    test_id = '10086'
    static,dynamic,objective = await get_llm_result_parallel(test_id,write_log=True)
    
    insert_code, extra_code = create_code(static,dynamic,objective,user_problem=problems[test_id])
    create_code_file(
        template_file_path='D:\\资料\\AI攻略生成比赛\\基于多智能体协同的高价值信息生成-数据集相关文件\\基于多智能体协同的高价值信息生成-数据集相关文件\\Ai_travel\\src\\template_multi_stage.py',
        code_file_path='D:\\资料\\AI攻略生成比赛\\基于多智能体协同的高价值信息生成-数据集相关文件\\基于多智能体协同的高价值信息生成-数据集相关文件\\Ai_travel\\prompts\\code\\id_6.py',
        insert_code=insert_code,
        extra_code=extra_code
    )
    ir_from_json(static)
    dynamic_constraint_from_json(dynamic)
    print(static)
    print(dynamic)
    print(objective)
    end = time.time()
    print(f'耗时：{(end-begin):2f}s')

if __name__ == '__main__':
    # asyncio.run(test_static())
    asyncio.run(main())
    # asyncio.run(test_objective())
    # asyncio.run(test())