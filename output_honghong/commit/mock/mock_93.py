# mock.py — 只包含前置模拟数据（适配你已有的生成/包装函数）
import random
import json

# 🚄 跨城交通（盛泽 → 武汉 出发；武汉 → 盛泽 返回）
# 要求：2025-09-05 上午从盛泽站出发，2025-09-09 中午从武汉返回（模拟数据）
cross_city_train_departure = [
    {
        "origin_id": "sz_001",
        "destination_id": "wh_001",
        "train_number": "G3272",
        "departure_time": "2025-09-05 07:55",
        "duration": "270",             # 单位：分钟（4.5小时）
        "cost": "320.0",               # 单程票价（每人，二等座，元）
        "origin_station": "盛泽站",
        "destination_station": "武汉站"
    },
    {
        "origin_id": "sz_002",
        "destination_id": "wh_002",
        "train_number": "G3258",
        "departure_time": "2025-09-05 09:10",
        "duration": "300",             # 5小时
        "cost": "350.0",
        "origin_station": "盛泽站",
        "destination_station": "武汉站"
    }
]

# 如果没有中转，保持空列表（与你模板一致）
cross_city_train_transfer = []

# 返回（武汉 -> 盛泽），要求中午出发（示例：2025-09-09 12:30）
cross_city_train_back = [
    {
        "origin_id": "wh_001",
        "destination_id": "sz_001",
        "train_number": "G3280",
        "departure_time": "2025-09-09 12:30",
        "duration": "270",
        "cost": "330.0",
        "origin_station": "武汉站",
        "destination_station": "盛泽站"
    }
]

# 🎡 景点（多城市嵌套结构：外层为城市列表；本次仅武汉一个城市）
# 字段与之前模板保持一致（id/name/cost/type/rating/duration），并附 start_stage/end_stage 由你的 generate_stage 填充
attractions = [
    [
        {"id":"a_wh_001","name":"黄鹤楼","cost":80.0,"type":"历史地标","rating":4.7,"duration":120.0},
        {"id":"a_wh_002","name":"东湖听涛风景区（含湖滨）","cost":0.0,"type":"自然景区","rating":4.6,"duration":180.0},
        {"id":"a_wh_003","name":"户部巷（老字号小吃街）","cost":40.0,"type":"美食街","rating":4.5,"duration":90.0},
        {"id":"a_wh_004","name":"武汉长江大桥/江汉关观景带","cost":0.0,"type":"观景/步行","rating":4.4,"duration":60.0},
        {"id":"a_wh_005","name":"湖北省博物馆","cost":0.0,"type":"博物馆","rating":4.6,"duration":120.0}
    ]
]

# 🏨 住宿（嵌套：每个城市一个子列表）
# 价格为每间夜价格（人民币），尽量选经济型/舒适型
accommodations = [
    [
        {"id":"h_wh_001","name":"汉庭酒店（武昌地铁站店）","cost":260.0,"type":"经济型连锁","rating":4.4,"feature":"靠近地铁，含免费Wi-Fi"},
        {"id":"h_wh_002","name":"如家快捷（江汉路店）","cost":280.0,"type":"经济型连锁","rating":4.5,"feature":"步行可达江汉路商业街"},
        {"id":"h_wh_003","name":"7天连锁（光谷店）","cost":240.0,"type":"经济型连锁","rating":4.3,"feature":"位于商圈，性价比高"},
        # 干扰项（高价/豪华，不应被选）
        {"id":"h_wh_004","name":"武汉国际大酒店（高星级样例）","cost":980.0,"type":"五星级","rating":4.8,"feature":"高端，不用于经济优先策略"}
    ]
]

# 🍜 餐厅（嵌套：每个城市一个子列表）
restaurants = [
    [
        {"id":"r_wh_001","name":"户部巷小吃-老字号摊位","cost":60.0,"type":"小吃","rating":4.5,"recommended_food":"热干面、豆皮、汤包","queue_time":10.0,"duration":45.0},
        {"id":"r_wh_002","name":"老汉口酒家（家常菜）","cost":120.0,"type":"家常菜","rating":4.4,"recommended_food":"家常湖北菜","queue_time":15.0,"duration":80.0},
        {"id":"r_wh_003","name":"江滩轻食咖啡","cost":80.0,"type":"轻餐","rating":4.2,"recommended_food":"简餐、咖啡","queue_time":5.0,"duration":50.0},
        {"id":"r_wh_004","name":"武汉热干面馆（连锁）","cost":35.0,"type":"快餐","rating":4.3,"recommended_food":"热干面","queue_time":5.0,"duration":30.0},
        # 干扰项（偏贵）
        {"id":"r_wh_005","name":"高档江景西餐厅","cost":420.0,"type":"西餐","rating":4.7,"recommended_food":"牛排","queue_time":20.0,"duration":100.0}
    ]
]

# 🚕 市内交通：由你的 generate_intra_city_transport(hotel_ids, attraction_ids) 生成
# 注意：下面两行会被你的模板函数使用来生成完整的 intra-city 网络
hotel_ids = [item['id'] for _ in accommodations for item in _]
attraction_ids = [item['id'] for _ in attractions for item in _]

# 🚕 市内交通
# 由你的全连通随机函数生成即可，例如：
# generate_intra_city_transport(hotel_ids, attraction_ids)


# 提示：市内交通(intra-city-transport)请继续使用你的全连通随机生成函数，
# 以酒店与景点的ID为节点生成：酒店↔景点、景点↔景点两两可达。



def generate_intra_city_transport(hotel_ids, attraction_ids, seed=42):
    """
    自动生成市内交通数据：
    - 酒店 -> 景点（双向）
    - 景点 -> 景点（双向）
    输出格式符合接口 /intra-city-transport/{city_name}
    """
    random.seed(seed)
    result = {}

    all_pois = hotel_ids + attraction_ids

    def gen_params():
        # 模拟打车与公交时间和价格
        taxi_duration = random.randint(8, 20)
        taxi_cost = round(random.uniform(12, 45), 1)
        bus_duration = taxi_duration * random.randint(2, 3)
        bus_cost = random.randint(2, 8)
        return {
            "taxi_duration": str(taxi_duration),
            "taxi_cost": str(taxi_cost),
            "bus_duration": str(bus_duration),
            "bus_cost": str(bus_cost)
        }

    # 酒店 -> 景点（双向）
    for h in hotel_ids:
        for a in attraction_ids:
            result[f"{h},{a}"] = gen_params()
            result[f"{a},{h}"] = gen_params()

    # 景点 -> 景点（双向）
    for i in range(len(attraction_ids)):
        for j in range(i + 1, len(attraction_ids)):
            a1, a2 = attraction_ids[i], attraction_ids[j]
            result[f"{a1},{a2}"] = gen_params()
            result[f"{a2},{a1}"] = gen_params()

    return result


hotel_ids = [item['id'] for _ in accommodations for item in _]
attraction_ids = [item['id'] for _ in attractions for item in _]

infra_transportation_info = generate_intra_city_transport(hotel_ids, attraction_ids)


# 美观打印 JSON 格式
# print(json.dumps(infra_transportation_info, indent=4, ensure_ascii=False))
def generate_stage(day_list):
    curr_day = 1
    for idx,city in enumerate(attractions):
        for item in city:
            item['start_stage'] = curr_day
            item['end_stage'] = curr_day + day_list[idx] - 1
        curr_day += day_list[idx]

    curr_day = 1
    for idx,city in enumerate(restaurants):
        for item in city:
            item['start_stage'] = curr_day
            item['end_stage'] = curr_day + day_list[idx] - 1
        curr_day += day_list[idx]

    curr_day = 1    
    for idx,city in enumerate(accommodations):
        for item in city:
            item['start_stage'] = curr_day
            item['end_stage'] = curr_day + day_list[idx] - 1    
        curr_day += day_list[idx]
def get_mock_data(is_ground_truth = False):

    if is_ground_truth:
        return cross_city_train_departure, cross_city_train_back,{'attractions': attractions[0], 'accommodations': accommodations[0], 'restaurants': restaurants[0]}, infra_transportation_info

    return cross_city_train_departure, cross_city_train_transfer, cross_city_train_back,{'attractions': attractions, 'accommodations': accommodations, 'restaurants': restaurants}, infra_transportation_info
