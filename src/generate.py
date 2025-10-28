from prompt_system import PromptSystem
from base import LLM
import asyncio
from IR import *
from dataclasses import asdict
import json,re
from base import Config
import os

prompt_sys = PromptSystem.getSingleton() 
static_prompt = prompt_sys.getPrompt('static')
dynamic_prompt = prompt_sys.getPrompt('dynamic')
objective_prompt = prompt_sys.getPrompt('objective')
llm_static = LLM('deepseek-v3.1-nothinking',system_prompt=static_prompt)
llm_dynamic = LLM('deepseek-v3.1-nothinking',system_prompt=dynamic_prompt)
llm_objective = LLM('deepseek-v3.1-nothinking',system_prompt=objective_prompt)

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
        depature_transport_constraints=parse_constraint(data.get("depature_transport_constraints")),
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
        daily_total_meal_time=parse_expr(data.get("daily_total_meal_time")),
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
        # 基础字段
        "num_travlers", "rooms_per_night", "change_hotel",
        # 时间相关
        "daily_total_time", "daily_queue_time", "daily_total_meal_time", "daily_transportation_time",
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
    if write_log:
        log_path = os.path.join(Config.get_global_config().config['log_path'], f'{problem_id}.log')
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write('#' * 50 + 'static result' + '#' * 50 + '\n\n')
            f.write(f"{results[0]}\n\n")
            f.write('#' * 50 + 'dynamic result' + '#' * 50 + '\n\n')
            f.write(f"{results[1]}\n\n")
            f.write('#' * 50 + 'objective result' + '#' * 50 + '\n\n')
            f.write(f"{results[2]}\n\n")

    pattern = r'(?<=```python)(.*?)(?=```)'   # 非贪婪匹配中间内容
    matches = re.findall(pattern, results[2] ,re.DOTALL)
    objective_result = ""
    if matches:
        objective_result = matches[-1].strip()
    return extract_json_block(results[0]),extract_json_block(results[1]), objective_result


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


async def main():
    import time
    from tqdm import tqdm
    print('=======start=======')
    begin = time.time()
    problem_file = Config.get_global_config().config['problem_file']
    code_path = Config.get_global_config().config['code_path']
    template_file = Config.get_global_config().config['template_file']

    break_point = 0
    with open(problem_file, 'r', encoding='utf-8') as f:
        json_problems = json.load(f)
        for item in json_problems:
            problems[item['question_id']] = item['question']
    print(f'初始化完成，成功导入数据集: {len(problems)}条')
    for problem in tqdm(problems):
        sample = {
            "question_id": problem,
            "question": problems[problem],
            "code_path": f'code/id_{problem}.py'
        }
        code_config.append(sample)
        if break_point > 0 and int(problem) <= break_point:
            time.sleep(0.1)
            continue
            
        static,dynamic,objective = await get_llm_result_parallel(problem,write_log=True)
        main_code,extra_code = create_code(static,dynamic,objective,user_problem=problems[problem])
        code_file = f"{code_path}/id_{problem}.py"
        create_code_file(template_file_path=template_file,code_file_path=code_file,insert_code=main_code,extra_code=extra_code)

    
    dump_file = Config.get_global_config().config['question_prompt']

    with open(dump_file, 'w', encoding='utf-8') as f:
        json.dump(code_config,f,indent=2,ensure_ascii=False)
    end = time.time()
    print('=======end=======')
    print(f'耗时:{end-begin}秒')

async def test_static():
    import time
    test_problem = "2025年7月1日上午，我将从武汉大学地铁站启程前往厦门，开启为期6天的旅行，并于7月6日晚搭乘高铁返程。出发交通指定搭乘G2045次高铁，本次旅行预算控制在20000元以内，期望能深入体验厦门的本地文化与自然风景。请优先安排POI评分高的景点、餐厅与住宿，入住三星级以上宾馆，要求含早餐，且至少吃一次福建菜，并确保每日动线紧凑、交通方便，提升整体出行舒适度。"
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
    test_problem = "我计划于2025年10月15日至19日从广州前往北京开展为期五天的高品质双人旅行，总预算为20000元，需满足以下需求：全程入住四星级及以上标准酒店，行程中须包含颐和园、恭王府博物馆等风景名胜，并安排一次正宗老北京烤鸭体验。15日早上从洛阳龙门站出发，19日晚返回洛阳。返程指定搭乘G651次列车。每日行程需兼顾热门景点与合理动线，避免过度奔波，市内交通以打车为主，整体行程注重舒适性与文化深度，尽量延长游玩时间、减少通勤和排队时间。"
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

    static,dynamic,objective = await get_llm_result_parallel('6')
    insert_code, extra_code = create_code(static,dynamic,objective,user_problem=problems['6'])
    create_code_file(
        template_file_path='D:\\资料\\AI攻略生成比赛\\基于多智能体协同的高价值信息生成-数据集相关文件\\基于多智能体协同的高价值信息生成-数据集相关文件\\Ai_travel\\src\\template.py',
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
    # asyncio.run(main())
    # asyncio.run(test_dynamic())
    # asyncio.run(test_objective())
    asyncio.run(test())