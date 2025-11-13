import random
import json

# 🚄 跨城交通（时间与票价为便于测试的合理占位值）
cross_city_train_departure = [
    {
        "origin_id": "wh_001",
        "destination_id": "sz_001",
        "train_number": "G1025",
        "duration": "255",              # 分钟，武汉→深圳 11/23 上午
        "cost": "460.0",
        "origin_station": "武汉站",
        "destination_station": "深圳北站"
    },
    {
        "origin_id": "wh_002",
        "destination_id": "sz_002",
        "train_number": "G1003",
        "duration": "270",
        "cost": "480.0",
        "origin_station": "武汉站",
        "destination_station": "深圳北站"
    }
]

# 11/27 晚间 深圳→洛阳（确保当晚抵达洛阳住宿）
cross_city_train_transfer = [
    {
        "origin_id": "sz_003",
        "destination_id": "ly_001",
        "train_number": "G2008",
        "duration": "360",              # 分钟，深圳北→洛阳龙门
        "cost": "620.0",
        "origin_station": "深圳北站",
        "destination_station": "洛阳龙门站"
    },
    {
        "origin_id": "sz_004",
        "destination_id": "ly_002",
        "train_number": "G3106",
        "duration": "380",
        "cost": "640.0",
        "origin_station": "深圳北站",
        "destination_station": "洛阳龙门站"
    }
]

# 12/01 晚间 洛阳→武汉
cross_city_train_back = [
    {
        "origin_id": "ly_003",
        "destination_id": "wh_003",
        "train_number": "G673",
        "duration": "210",
        "cost": "230.0",
        "origin_station": "洛阳龙门站",
        "destination_station": "武汉站"
    },
    {
        "origin_id": "ly_004",
        "destination_id": "wh_004",
        "train_number": "D2221",
        "duration": "240",
        "cost": "210.0",
        "origin_station": "洛阳龙门站",
        "destination_station": "武昌站"
    }
]


# 🎡 景点（多城市嵌套：[[深圳景点...],[洛阳景点...]]）
attractions = [
    [   # ——— 深圳（5天建议至少5个核心口碑景点） ———
        {"id":"a_sz_001","name":"世界之窗","cost":220.0,"type":"主题乐园","rating":4.7,"duration":240.0},
        {"id":"a_sz_002","name":"深圳湾公园","cost":0.0,"type":"城市公园","rating":4.8,"duration":180.0},
        {"id":"a_sz_003","name":"锦绣中华民俗村","cost":200.0,"type":"人文景区","rating":4.6,"duration":240.0},
        {"id":"a_sz_004","name":"大梅沙海滨公园","cost":0.0,"type":"海滨","rating":4.5,"duration":180.0},
        {"id":"a_sz_005","name":"华侨城欢乐海岸","cost":0.0,"type":"休闲商业区","rating":4.6,"duration":150.0}
    ],
    [   # ——— 洛阳（含指定：洛阳牡丹园） ———
        {"id":"a_ly_001","name":"洛阳牡丹园","cost":30.0,"type":"花园景区","rating":4.7,"duration":150.0},
        {"id":"a_ly_002","name":"白马寺","cost":35.0,"type":"佛教圣地","rating":4.8,"duration":180.0},
        {"id":"a_ly_003","name":"丽景门古城","cost":40.0,"type":"历史街区","rating":4.6,"duration":120.0},
        {"id":"a_ly_004","name":"洛阳博物馆","cost":0.0,"type":"博物馆","rating":4.7,"duration":150.0}
    ]
]


# 🏨 住宿（多城市嵌套：[[深圳住宿...],[洛阳住宿...]]）
accommodations = [
    [   # ——— 深圳（高舒适/服务好，价格适中以贴合总预算） ———
        {"id":"h_sz_001","name":"全季酒店（深圳湾店）","cost":480.0,"type":"三星级","rating":4.6,"feature":"近深圳湾公园，打车便利"},
        {"id":"h_sz_002","name":"维也纳国际酒店（后海店）","cost":520.0,"type":"四星级","rating":4.6,"feature":"靠近地铁与景区"},
        {"id":"h_sz_003","name":"和颐至尚酒店（会展中心店）","cost":560.0,"type":"四星级","rating":4.5,"feature":"CBD 区位，服务稳定"},
        {"id":"h_sz_004","name":"桔子水晶酒店（欢乐海岸店）","cost":600.0,"type":"四星级","rating":4.7,"feature":"临近欢乐海岸，出行便捷"}
    ],
    [   # ——— 洛阳（包含你指定的两家） ———
        {"id":"h_ly_001","name":"洛阳非凡·云联酒店","cost":360.0,"type":"舒适型","rating":4.8,"feature":"市中心，服务口碑好"},
        {"id":"h_ly_002","name":"汇丰大酒店","cost":380.0,"type":"舒适型","rating":4.6,"feature":"临近商圈，打车便利"},
        {"id":"h_ly_003","name":"全季酒店（洛阳王城公园店）","cost":420.0,"type":"舒适型","rating":4.7,"feature":"近王城公园，安静整洁"}
    ]
]


# 🍜 餐厅（多城市嵌套：[[深圳餐厅...],[洛阳餐厅...]]）
# 数量足够支撑 5 天（深圳）与 4 天（洛阳）的排餐；洛阳含你指定的两家店名
restaurants = [
    [   # ——— 深圳（高性价比+服务稳定，排队时长适中） ———
        {"id":"r_sz_001","name":"点都德（深圳湾店）","cost":120.0,"type":"早茶","rating":4.7,"recommended_food":"虾饺皇, 凤爪","queue_time":20.0,"duration":90.0},
        {"id":"r_sz_002","name":"陶陶居（欢乐海岸店）","cost":130.0,"type":"粤菜","rating":4.7,"recommended_food":"烧鹅, 叉烧","queue_time":25.0,"duration":100.0},
        {"id":"r_sz_003","name":"渔民新村（海鲜）","cost":160.0,"type":"海鲜","rating":4.6,"recommended_food":"清蒸海鲜","queue_time":15.0,"duration":100.0},
        {"id":"r_sz_004","name":"和美素食馆（福田店）","cost":85.0,"type":"素菜","rating":4.5,"recommended_food":"中式素菜","queue_time":10.0,"duration":80.0},
        {"id":"r_sz_005","name":"南门涮肉（科技园店）","cost":120.0,"type":"火锅","rating":4.5,"recommended_food":"手切鲜肉","queue_time":12.0,"duration":100.0},
        {"id":"r_sz_006","name":"老成都川菜馆","cost":95.0,"type":"川菜","rating":4.5,"recommended_food":"宫保鸡丁, 回锅肉","queue_time":10.0,"duration":90.0},
        {"id":"r_sz_007","name":"船歌鱼水饺（海岸城）","cost":65.0,"type":"北方菜","rating":4.6,"recommended_food":"水饺拼盘","queue_time":8.0,"duration":70.0},
        {"id":"r_sz_008","name":"张记腊味煲仔饭","cost":55.0,"type":"粤式快餐","rating":4.4,"recommended_food":"煲仔饭","queue_time":8.0,"duration":60.0},
        {"id":"r_sz_011","name":"点都德（深圳湾店）","cost":120.0,"type":"早茶","rating":4.7,"recommended_food":"虾饺皇, 凤爪","queue_time":20.0,"duration":90.0},
        {"id":"r_sz_022","name":"陶陶居（欢乐海岸店）","cost":130.0,"type":"粤菜","rating":4.7,"recommended_food":"烧鹅, 叉烧","queue_time":25.0,"duration":100.0},
        {"id":"r_sz_033","name":"渔民新村（海鲜）","cost":160.0,"type":"海鲜","rating":4.6,"recommended_food":"清蒸海鲜","queue_time":15.0,"duration":100.0},
        {"id":"r_sz_044","name":"和美素食馆（福田店）","cost":85.0,"type":"素菜","rating":4.5,"recommended_food":"中式素菜","queue_time":10.0,"duration":80.0},
        {"id":"r_sz_055","name":"南门涮肉（科技园店）","cost":120.0,"type":"火锅","rating":4.5,"recommended_food":"手切鲜肉","queue_time":12.0,"duration":100.0},
        {"id":"r_sz_066","name":"老成都川菜馆","cost":95.0,"type":"川菜","rating":4.5,"recommended_food":"宫保鸡丁, 回锅肉","queue_time":10.0,"duration":90.0},
        {"id":"r_sz_077","name":"船歌鱼水饺（海岸城）","cost":65.0,"type":"北方菜","rating":4.6,"recommended_food":"水饺拼盘","queue_time":8.0,"duration":70.0},
        {"id":"r_sz_088","name":"张记腊味煲仔饭","cost":55.0,"type":"粤式快餐","rating":4.4,"recommended_food":"煲仔饭","queue_time":8.0,"duration":60.0},
        {"id":"r_sz_101","name":"点都德（深圳湾店）","cost":120.0,"type":"早茶","rating":4.7,"recommended_food":"虾饺皇, 凤爪","queue_time":20.0,"duration":90.0},
        {"id":"r_sz_102","name":"陶陶居（欢乐海岸店）","cost":130.0,"type":"粤菜","rating":4.7,"recommended_food":"烧鹅, 叉烧","queue_time":25.0,"duration":100.0},
        {"id":"r_sz_103","name":"渔民新村（海鲜）","cost":160.0,"type":"海鲜","rating":4.6,"recommended_food":"清蒸海鲜","queue_time":15.0,"duration":100.0},
        {"id":"r_sz_104","name":"和美素食馆（福田店）","cost":85.0,"type":"素菜","rating":4.5,"recommended_food":"中式素菜","queue_time":10.0,"duration":80.0},
        {"id":"r_sz_105","name":"南门涮肉（科技园店）","cost":120.0,"type":"火锅","rating":4.5,"recommended_food":"手切鲜肉","queue_time":12.0,"duration":100.0},
        {"id":"r_sz_106","name":"老成都川菜馆","cost":95.0,"type":"川菜","rating":4.5,"recommended_food":"宫保鸡丁, 回锅肉","queue_time":10.0,"duration":90.0},
        {"id":"r_sz_107","name":"船歌鱼水饺（海岸城）","cost":65.0,"type":"北方菜","rating":4.6,"recommended_food":"水饺拼盘","queue_time":8.0,"duration":70.0},
        {"id":"r_sz_108","name":"张记腊味煲仔饭","cost":55.0,"type":"粤式快餐","rating":4.4,"recommended_food":"煲仔饭","queue_time":8.0,"duration":60.0}
    ],
    [   # ——— 洛阳（包含：百香园餐饮、金大吉（河科大开元校区菁园店）） ———
        {"id":"r_ly_001","name":"百香园餐饮","cost":60.0,"type":"地方菜","rating":4.6,"recommended_food":"水席拼盘","queue_time":10.0,"duration":80.0},
        {"id":"r_ly_002","name":"金大吉（河南科技大学开元校区菁园店）","cost":35.0,"type":"地方小吃","rating":4.5,"recommended_food":"酱香盖饭","queue_time":8.0,"duration":60.0},
        {"id":"r_ly_003","name":"鲁记卤肉凉菜","cost":45.0,"type":"地方菜","rating":4.7,"recommended_food":"卤味拼盘","queue_time":10.0,"duration":70.0},
        {"id":"r_ly_004","name":"鲜羊肉汤店","cost":55.0,"type":"地方菜","rating":4.7,"recommended_food":"羊肉汤, 油饼","queue_time":12.0,"duration":75.0},
        {"id":"r_ly_005","name":"真不同饭店（牡丹店）","cost":95.0,"type":"豫菜","rating":4.7,"recommended_food":"牡丹燕菜","queue_time":15.0,"duration":90.0},
        {"id":"r_ly_006","name":"老城十字街胡辣汤","cost":25.0,"type":"小吃","rating":4.6,"recommended_food":"胡辣汤","queue_time":6.0,"duration":40.0},
        {"id":"r_ly_007","name":"浆面条老店","cost":28.0,"type":"面食","rating":4.5,"recommended_food":"浆面条","queue_time":6.0,"duration":45.0},
        {"id":"r_ly_008","name":"驴肉火烧铺","cost":40.0,"type":"小吃","rating":4.6,"recommended_food":"驴肉火烧","queue_time":8.0,"duration":45.0},
        {"id":"r_ly_011","name":"百香园餐饮","cost":60.0,"type":"地方菜","rating":4.6,"recommended_food":"水席拼盘","queue_time":10.0,"duration":80.0},
        {"id":"r_ly_012","name":"金大吉（河南科技大学开元校区菁园店）","cost":35.0,"type":"地方小吃","rating":4.5,"recommended_food":"酱香盖饭","queue_time":8.0,"duration":60.0},
        {"id":"r_ly_013","name":"鲁记卤肉凉菜","cost":45.0,"type":"地方菜","rating":4.7,"recommended_food":"卤味拼盘","queue_time":10.0,"duration":70.0},
        {"id":"r_ly_014","name":"鲜羊肉汤店","cost":55.0,"type":"地方菜","rating":4.7,"recommended_food":"羊肉汤, 油饼","queue_time":12.0,"duration":75.0},
        {"id":"r_ly_015","name":"真不同饭店（牡丹店）","cost":95.0,"type":"豫菜","rating":4.7,"recommended_food":"牡丹燕菜","queue_time":15.0,"duration":90.0},
        {"id":"r_ly_016","name":"老城十字街胡辣汤","cost":25.0,"type":"小吃","rating":4.6,"recommended_food":"胡辣汤","queue_time":6.0,"duration":40.0},
        {"id":"r_ly_017","name":"浆面条老店","cost":28.0,"type":"面食","rating":4.5,"recommended_food":"浆面条","queue_time":6.0,"duration":45.0},
        {"id":"r_ly_018","name":"驴肉火烧铺","cost":40.0,"type":"小吃","rating":4.6,"recommended_food":"驴肉火烧","queue_time":8.0,"duration":45.0}
    ]
]

# 🚕 市内交通说明：
# 请用你的随机函数生成两城的市内交通(intra-city-transport)，
# 确保：酒店↔景点、景点↔景点 两两可达；偏向打车/近距离以减少通勤时间。


def expand_data(expand_num, seed = 38):
    random.seed(seed)   
    from uuid import uuid4
    aim_cross_city_train_departure = []
    aim_cross_city_train_transfer = []
    aim_cross_city_train_back = []
    for i in range(expand_num):
        tmp = [    {
            "origin_id": item["origin_id"],
            "destination_id": item["destination_id"],
            "train_number": str(uuid4()),
            "duration": str(int(item["duration"]) + random.randint(0, 100) - 50),              # 分钟，武汉→深圳 11/23 上午
            "cost": str(float(item["cost"]) + random.randint(0, 100) - 50),
            "origin_station": item["origin_station"],
            "destination_station": item["destination_station"]
        } for item in cross_city_train_departure]
        aim_cross_city_train_departure.extend(tmp)
        tmp = [    {
            "origin_id": item["origin_id"],
            "destination_id": item["destination_id"],
            "train_number": str(uuid4()),
            "duration": str(int(item["duration"]) + random.randint(0, 100) - 50),              # 分钟，武汉→深圳 11/23 上午
            "cost": str(float(item["cost"]) + random.randint(0, 100) - 50),
            "origin_station": item["origin_station"],
            "destination_station": item["destination_station"]
        } for item in cross_city_train_back]
        aim_cross_city_train_back.extend(tmp)
        tmp = [    {
            "origin_id": item["origin_id"],
            "destination_id": item["destination_id"],
            "train_number": str(uuid4()),
            "duration": str(int(item["duration"]) + random.randint(0, 100) - 50),              # 分钟，武汉→深圳 11/23 上午
            "cost": str(float(item["cost"]) + random.randint(0, 100) - 50),
            "origin_station": item["origin_station"],
            "destination_station": item["destination_station"]
        } for item in cross_city_train_transfer]
        aim_cross_city_train_transfer.extend(tmp) 
    cross_city_train_departure.extend(aim_cross_city_train_departure)  
    cross_city_train_back.extend(aim_cross_city_train_back)
    cross_city_train_transfer.extend(aim_cross_city_train_transfer) 
    for city in attractions:
        aim_tmp = []
        for i in range(expand_num):
            tmp = [{
                "id": str(uuid4()),
                "name": item["name"],
                "cost": item["cost"] + round((random.random() - 0.5) * item["cost"],1),
                "type": item["type"],
                "rating": item["rating"] + round((random.random() - 0.5) * item["rating"],1),
            } for item in city]  
            aim_tmp.extend(tmp)
        city.extend(aim_tmp)

    for city in restaurants:
        aim_tmp = []
        for i in range(expand_num):
            tmp = [{
                "id": str(uuid4()),
                "name": item["name"],
                "cost": item["cost"] + round((random.random() - 0.5) * item["cost"],1),
                "type": item["type"],
                "rating": item["rating"] + round((random.random() - 0.5) * item["rating"],1),
                "recommended_food": item["recommended_food"],
                "queue_time": item["queue_time"] + round((random.random() - 0.5) * item["queue_time"],1),
            } for item in city]  
            aim_tmp.extend(tmp)
        city.extend(aim_tmp)

    for city in accommodations:
        aim_tmp = []
        for i in range(expand_num):
            tmp = [{
                "id": str(uuid4()),
                "name": item["name"],
                "cost": item["cost"] + round((random.random() - 0.5) * item["cost"],1),
                "type": item["type"],
                "rating": item["rating"] + round((random.random() - 0.5) * item["rating"],1),
                "feature": item["feature"]
            } for item in city]  
            aim_tmp.extend(tmp)
        city.extend(aim_tmp)
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
    expand_data(100)
    if is_ground_truth:
        return cross_city_train_departure, cross_city_train_back,{'attractions': attractions[0], 'accommodations': accommodations[0], 'restaurants': restaurants[0]}, infra_transportation_info

    return cross_city_train_departure, cross_city_train_transfer, cross_city_train_back,{'attractions': attractions, 'accommodations': accommodations, 'restaurants': restaurants}, infra_transportation_info

