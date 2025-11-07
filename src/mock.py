import random
import json

# 🚄 跨城交通（杭州 → 广州 出发；广州 → 杭州 返回）
cross_city_train_departure = [
    {"origin_id":"hz_001","destination_id":"gz_001","train_number":"G1301","duration":"390","cost":"520.0","origin_station":"杭州东站","destination_station":"广州南站"},
    {"origin_id":"hz_002","destination_id":"gz_002","train_number":"G1325","duration":"400","cost":"540.0","origin_station":"杭州东站","destination_station":"广州南站"}
]

cross_city_train_transfer = []  # 无中转城市

cross_city_train_back = [
    {"origin_id":"gz_003","destination_id":"hz_003","train_number":"G1328","duration":"390","cost":"520.0","origin_station":"广州南站","destination_station":"杭州东站"}
]


# 🎡 景点（多城市嵌套：[[广州景点]]）
attractions = [
    [
        {"id":"a_gz_001","name":"广州塔","cost":150.0,"type":"地标建筑","rating":4.8,"duration":180.0},
        {"id":"a_gz_002","name":"沙面岛","cost":0.0,"type":"历史街区","rating":4.7,"duration":120.0},
        {"id":"a_gz_003","name":"越秀公园","cost":10.0,"type":"城市公园","rating":4.6,"duration":150.0},
        {"id":"a_gz_004","name":"陈家祠","cost":25.0,"type":"岭南建筑","rating":4.7,"duration":100.0},
        {"id":"a_gz_005","name":"白云山风景区","cost":30.0,"type":"自然景区","rating":4.7,"duration":180.0}
    ]
]


# 🏨 住宿（禁止四星及以上酒店，仅包含经济型/三星级）
accommodations = [
    [
        {"id":"h_gz_001","name":"如家精选酒店（广州北京路步行街店）","cost":320.0,"type":"三星级","rating":4.6,"feature":"含早餐, 靠近地铁"},
        {"id":"h_gz_002","name":"汉庭酒店（广州天河体育中心店）","cost":290.0,"type":"经济型连锁","rating":4.5,"feature":"地铁直达, 干净整洁"},
        {"id":"h_gz_003","name":"7天优品酒店（广州火车站店）","cost":280.0,"type":"经济型连锁","rating":4.4,"feature":"靠近火车站, 性价比高"},
        {"id":"h_gz_004","name":"锦江之星品尚（广州越秀公园店）","cost":310.0,"type":"三星级","rating":4.7,"feature":"含早餐, 靠近景点"},
        {"id":"h_gz_005","name":"维也纳3好酒店（广州塔店）","cost":390.0,"type":"三星级","rating":4.8,"feature":"含早餐, 景观房"},
        # 干扰项（四星以上，不应被推荐）
        {"id":"h_gz_006","name":"广州白天鹅宾馆","cost":780.0,"type":"五星级","rating":4.9,"feature":"禁止使用"},
        {"id":"h_gz_007","name":"广州花园酒店","cost":850.0,"type":"五星级","rating":4.8,"feature":"禁止使用"}
    ]
]


# 🍜 餐厅（两天行程：6家满足 + 2家干扰）
restaurants = [
    [
        {"id":"r_gz_001","name":"点都德（北京路店）","cost":95.0,"type":"早茶","rating":4.8,"recommended_food":"虾饺皇, 干蒸烧卖","queue_time":20.0,"duration":90.0},
        {"id":"r_gz_002","name":"陶陶居（上下九店）","cost":110.0,"type":"粤菜","rating":4.7,"recommended_food":"烧鹅, 虾饺","queue_time":25.0,"duration":100.0},
        {"id":"r_gz_003","name":"银记肠粉店","cost":40.0,"type":"快餐","rating":4.6,"recommended_food":"牛肉肠粉, 双拼粥","queue_time":10.0,"duration":50.0},
        {"id":"r_gz_004","name":"南信牛奶甜品专家","cost":35.0,"type":"甜品","rating":4.6,"recommended_food":"姜撞奶, 双皮奶","queue_time":8.0,"duration":40.0},
        {"id":"r_gz_005","name":"炳胜品味（珠江新城店）","cost":180.0,"type":"粤菜","rating":4.8,"recommended_food":"烧鸭, 豉汁排骨","queue_time":15.0,"duration":90.0},
        {"id":"r_gz_006","name":"莲香楼（中山路店）","cost":80.0,"type":"早茶","rating":4.6,"recommended_food":"凤爪, 虾饺皇","queue_time":10.0,"duration":80.0},
        {"id":"r_gz_009","name":"南信牛奶甜品专家","cost":35.0,"type":"甜品","rating":4.6,"recommended_food":"姜撞奶, 双皮奶","queue_time":8.0,"duration":40.0},
        {"id":"r_gz_010","name":"炳胜品味（珠江新城店）","cost":180.0,"type":"粤菜","rating":4.8,"recommended_food":"烧鸭, 豉汁排骨","queue_time":15.0,"duration":90.0},
        {"id":"r_gz_011","name":"莲香楼（中山路店）","cost":80.0,"type":"早茶","rating":4.6,"recommended_food":"凤爪, 虾饺皇","queue_time":10.0,"duration":80.0},
        # 干扰项（价格高或非地道）
        {"id":"r_gz_007","name":"高端牛排馆","cost":480.0,"type":"西餐","rating":4.9,"recommended_food":"牛排","queue_time":5.0,"duration":120.0},
        {"id":"r_gz_008","name":"寿司屋SushiOne","cost":280.0,"type":"日料","rating":4.7,"recommended_food":"刺身拼盘","queue_time":10.0,"duration":100.0}
    ]
]


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
