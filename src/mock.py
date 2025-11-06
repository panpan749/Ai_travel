import random
import json

# 🚄 跨城交通数据

# 🚄 跨城交通数据
cross_city_train_transfer = []
# 🚄 跨城交通数据（包含指定列车 D903）

cross_city_train_departure = [
    {
        "origin_id": "bj_701",
        "destination_id": "gz_701",
        "train_number": "D903",
        "duration": "600",
        "cost": "720.0",
        "origin_station": "北京西站",
        "destination_station": "广州南站"
    },
    {
        "origin_id": "bj_702",
        "destination_id": "gz_702",
        "train_number": "G71",
        "duration": "520",
        "cost": "850.0",
        "origin_station": "北京西站",
        "destination_station": "广州南站"
    }
]

cross_city_train_back = [
    {
        "origin_id": "gz_701",
        "destination_id": "bj_701",
        "train_number": "D904",
        "duration": "600",
        "cost": "720.0",
        "origin_station": "广州南站",
        "destination_station": "北京西站"
    },
    {
        "origin_id": "gz_702",
        "destination_id": "bj_702",
        "train_number": "G72",
        "duration": "520",
        "cost": "850.0",
        "origin_station": "广州南站",
        "destination_station": "北京西站"
    }
]


# 🏨 广州住宿数据 /accommodations/广州市
# 要求：经济型连锁酒店，保证舒适度、干净、独立房间（4人各一间）

accommodations = [
    {
        "id": "h701",
        "name": "如家精选酒店（广州越秀公园地铁站店）",
        "cost": 480.0,
        "type": "连锁酒店",
        "rating": 4.6,
        "feature": "含早餐, 靠近地铁, 房间干净"
    },
    {
        "id": "h702",
        "name": "汉庭优佳酒店（北京路步行街店）",
        "cost": 520.0,
        "type": "经济型连锁酒店",
        "rating": 4.7,
        "feature": "含早餐, 市中心, 靠近公交站"
    },
    {
        "id": "h703",
        "name": "全季酒店（广州海珠广场店）",
        "cost": 580.0,
        "type": "连锁酒店",
        "rating": 4.8,
        "feature": "含早餐, 安静舒适, 临近珠江"
    },
    {
        "id": "h704",
        "name": "7天优品酒店（中山纪念堂店）",
        "cost": 420.0,
        "type": "连锁酒店",
        "rating": 4.6,
        "feature": "含早餐, 交通便利"
    },
    {
        "id": "h705",
        "name": "锦江之星品尚（广州火车站店）",
        "cost": 390.0,
        "type": "经济型连锁",
        "rating": 4.5,
        "feature": "含早餐, 经济实惠"
    }
]


# 🎡 广州景点数据 /attractions/广州市
# 要求：包含邓世昌纪念馆、纯阳观等文化景点，同时补充合理可玩景点形成完整行程。

attractions = [
    {
        "id": "a701",
        "name": "邓世昌纪念馆",
        "cost": 0.0,
        "type": "历史纪念馆",
        "rating": 4.8,
        "duration": 120.0
    },
    {
        "id": "a702",
        "name": "纯阳观",
        "cost": 10.0,
        "type": "道教古迹",
        "rating": 4.7,
        "duration": 90.0
    },
    {
        "id": "a703",
        "name": "越秀公园",
        "cost": 0.0,
        "type": "自然公园",
        "rating": 4.6,
        "duration": 150.0
    },
    {
        "id": "a704",
        "name": "陈家祠（广东民间工艺博物馆）",
        "cost": 20.0,
        "type": "文化古迹",
        "rating": 4.8,
        "duration": 120.0
    },
    {
        "id": "a705",
        "name": "沙面岛",
        "cost": 0.0,
        "type": "历史街区",
        "rating": 4.7,
        "duration": 150.0
    },
    {
        "id": "a706",
        "name": "北京路步行街",
        "cost": 0.0,
        "type": "商业街区",
        "rating": 4.5,
        "duration": 120.0
    }
]


# 🍜 广州餐厅数据 /restaurants/广州市
# 控制人均 ≤150 元，餐饮品质良好，主要为粤菜馆与早茶餐厅

restaurants = [
    {"id": "r701", "name": "陶陶居（北京路店）", "cost": 120.0, "type": "粤菜馆", "rating": 4.8, "recommended_food": "早茶, 烧鹅, 虾饺", "queue_time": 20.0, "duration": 90.0},
    {"id": "r702", "name": "点都德（越秀店）", "cost": 110.0, "type": "早茶餐厅", "rating": 4.7, "recommended_food": "凤爪, 流沙包", "queue_time": 15.0, "duration": 80.0},
    {"id": "r703", "name": "银记肠粉店", "cost": 50.0, "type": "小吃", "rating": 4.6, "recommended_food": "肠粉, 牛杂", "queue_time": 8.0, "duration": 45.0},
    {"id": "r704", "name": "文记茶餐厅（海珠店）", "cost": 90.0, "type": "茶餐厅", "rating": 4.6, "recommended_food": "菠萝包, 咖喱牛腩", "queue_time": 10.0, "duration": 70.0},
    {"id": "r705", "name": "炳胜公馆（珠江新城）", "cost": 130.0, "type": "粤菜馆", "rating": 4.8, "recommended_food": "白切鸡, 烧鹅", "queue_time": 20.0, "duration": 90.0},
    {"id": "r706", "name": "泮溪酒家", "cost": 140.0, "type": "粤菜馆", "rating": 4.7, "recommended_food": "早茶, 白切鸡", "queue_time": 15.0, "duration": 90.0},
    {"id": "r707", "name": "南信甜品店（上下九）", "cost": 60.0, "type": "甜品店", "rating": 4.5, "recommended_food": "双皮奶, 杨枝甘露", "queue_time": 10.0, "duration": 50.0},
    {"id": "r708", "name": "广州酒家（文昌店）", "cost": 130.0, "type": "粤菜馆", "rating": 4.8, "recommended_food": "文昌鸡, 烧鸭", "queue_time": 25.0, "duration": 90.0},
    {"id": "r709", "name": "点心道（天河店）", "cost": 100.0, "type": "早茶餐厅", "rating": 4.6, "recommended_food": "虾饺, 凤爪", "queue_time": 12.0, "duration": 80.0},
    {"id": "r710", "name": "百味砂锅粥", "cost": 120.0, "type": "粥店", "rating": 4.7, "recommended_food": "砂锅粥, 沙姜猪手粥", "queue_time": 10.0, "duration": 90.0},

    # —— 干扰项（价格高或非粤菜） ——
    {"id": "r711", "name": "意大利餐厅La Bella Vita", "cost": 400.0, "type": "西餐", "rating": 4.8, "recommended_food": "牛排, 意面", "queue_time": 5.0, "duration": 120.0},
    {"id": "r712", "name": "川味火锅馆", "cost": 200.0, "type": "火锅", "rating": 4.5, "recommended_food": "麻辣火锅", "queue_time": 15.0, "duration": 100.0}
]





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


hotel_ids = [item['id'] for item in accommodations]
attraction_ids = [item['id'] for item in attractions]

infra_transportation_info = generate_intra_city_transport(hotel_ids, attraction_ids)


# 美观打印 JSON 格式
# print(json.dumps(infra_transportation_info, indent=4, ensure_ascii=False))

def get_mock_data():

    return cross_city_train_departure, cross_city_train_transfer, cross_city_train_back,{'attractions': [attractions], 'accommodations': [accommodations], 'restaurants': [restaurants]}, infra_transportation_info
