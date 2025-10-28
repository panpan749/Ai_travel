import random
import json

# 🚄 跨城交通数据

cross_city_train_departure = [
    {
        "origin_id": "cq_001",
        "destination_id": "gz_001",
        "train_number": "G2921",
        "duration": "480",
        "cost": "560.0",
        "origin_station": "重庆西站",
        "destination_station": "广州南站"
    },
    {
        "origin_id": "cq_002",
        "destination_id": "gz_002",
        "train_number": "G2923",
        "duration": "495",
        "cost": "580.0",
        "origin_station": "重庆北站",
        "destination_station": "广州南站"
    },
    {
        "origin_id": "cq_003",
        "destination_id": "gz_003",
        "train_number": "G2925",
        "duration": "510",
        "cost": "600.0",
        "origin_station": "重庆西站",
        "destination_station": "广州南站"
    }
]

cross_city_train_back = [
    {
        "origin_id": "gz_001",
        "destination_id": "cq_001",
        "train_number": "G2922",
        "duration": "480",
        "cost": "560.0",
        "origin_station": "广州南站",
        "destination_station": "重庆西站"
    },
    {
        "origin_id": "gz_002",
        "destination_id": "cq_002",
        "train_number": "G2924",
        "duration": "495",
        "cost": "580.0",
        "origin_station": "广州南站",
        "destination_station": "重庆北站"
    }
]


# 🏨 广州住宿数据 /accommodations/广州市

accommodations = [
    {
        "id": "h401",
        "name": "广州长隆熊猫酒店",
        "cost": 1180.0,
        "type": "亲子主题酒店",
        "rating": 4.8,
        "feature": "含早餐, 临近长隆景区, 儿童设施完善"
    },
    {
        "id": "h402",
        "name": "全季酒店（广州塔店）",
        "cost": 880.0,
        "type": "连锁酒店",
        "rating": 4.6,
        "feature": "含早餐, 可观广州塔夜景"
    },
    {
        "id": "h403",
        "name": "锦江之星品尚（北京路步行街店）",
        "cost": 720.0,
        "type": "连锁酒店",
        "rating": 4.5,
        "feature": "含早餐, 地铁直达"
    },
    {
        "id": "h404",
        "name": "粤海喜来登酒店",
        "cost": 1100.0,
        "type": "五星级酒店",
        "rating": 4.9,
        "feature": "含早餐, 临江景观, 儿童乐园"
    },
    {
        "id": "h405",
        "name": "广州花园酒店",
        "cost": 980.0,
        "type": "五星级酒店",
        "rating": 4.8,
        "feature": "含早餐, 市中心交通便利"
    },
    {
        "id": "h406",
        "name": "广州亚朵酒店（珠江新城）",
        "cost": 890.0,
        "type": "连锁酒店",
        "rating": 4.7,
        "feature": "含早餐, 安静舒适"
    },
    {
        "id": "h407",
        "name": "城市快捷酒店（评分偏低）",
        "cost": 480.0,
        "type": "快捷酒店",
        "rating": 3.8,
        "feature": "不含早餐"
    }
]


# 🎡 广州景点 /attractions/广州市

attractions = [
    {
        "id": "a401",
        "name": "长隆野生动物世界",
        "cost": 350.0,
        "type": "主题乐园",
        "rating": 4.9,
        "duration": 360.0
    },
    {
        "id": "a402",
        "name": "广州塔",
        "cost": 150.0,
        "type": "城市地标",
        "rating": 4.8,
        "duration": 120.0
    },
    {
        "id": "a403",
        "name": "沙面岛",
        "cost": 0.0,
        "type": "历史街区",
        "rating": 4.7,
        "duration": 150.0
    },
    {
        "id": "a404",
        "name": "珠江夜游",
        "cost": 180.0,
        "type": "夜景游船",
        "rating": 4.6,
        "duration": 120.0
    },
    {
        "id": "a405",
        "name": "陈家祠",
        "cost": 10.0,
        "type": "岭南建筑文化",
        "rating": 4.7,
        "duration": 90.0
    },
    {
        "id": "a406",
        "name": "北京路步行街",
        "cost": 0.0,
        "type": "购物街区",
        "rating": 4.5,
        "duration": 180.0
    },
    {
        "id": "a407",
        "name": "白云山风景区",
        "cost": 5.0,
        "type": "自然风景区",
        "rating": 4.8,
        "duration": 240.0
    },
    {
        "id": "a408",
        "name": "上下九步行街",
        "cost": 0.0,
        "type": "老城区商业街",
        "rating": 4.4,
        "duration": 180.0
    }
]


# 🍜 广州餐厅 /restaurants/广州市

restaurants = [
    # 满足用户条件的高评分粤菜馆与早茶餐厅
    {"id": "r401", "name": "陶陶居（北京路店）", "cost": 120.0, "type": "粤菜馆", "rating": 4.8, "recommended_food": "早茶, 虾饺, 烧卖", "queue_time": 20.0, "duration": 90.0},
    {"id": "r402", "name": "点都德（上下九店）", "cost": 110.0, "type": "早茶餐厅", "rating": 4.7, "recommended_food": "流沙包, 凤爪", "queue_time": 25.0, "duration": 90.0},
    {"id": "r403", "name": "银记肠粉店（荔湾店）", "cost": 45.0, "type": "小吃", "rating": 4.6, "recommended_food": "肠粉, 牛杂", "queue_time": 10.0, "duration": 45.0},
    {"id": "r404", "name": "莲香楼", "cost": 130.0, "type": "早茶餐厅", "rating": 4.8, "recommended_food": "叉烧包, 莲蓉酥", "queue_time": 20.0, "duration": 90.0},
    {"id": "r405", "name": "广州酒家（文昌店）", "cost": 140.0, "type": "粤菜馆", "rating": 4.9, "recommended_food": "文昌鸡, 烧鹅", "queue_time": 25.0, "duration": 90.0},
    {"id": "r406", "name": "点心皇（珠江新城）", "cost": 100.0, "type": "早茶餐厅", "rating": 4.7, "recommended_food": "凤爪, 烧卖, 虾饺", "queue_time": 15.0, "duration": 80.0},
    {"id": "r407", "name": "南信甜品店（上下九）", "cost": 60.0, "type": "甜品店", "rating": 4.6, "recommended_food": "双皮奶, 杨枝甘露", "queue_time": 10.0, "duration": 60.0},
    {"id": "r408", "name": "点点心（太古汇）", "cost": 120.0, "type": "早茶餐厅", "rating": 4.7, "recommended_food": "虾饺皇, 凤爪", "queue_time": 18.0, "duration": 90.0},
    {"id": "r409", "name": "文记茶餐厅", "cost": 95.0, "type": "茶餐厅", "rating": 4.5, "recommended_food": "菠萝包, 咖喱牛腩", "queue_time": 8.0, "duration": 75.0},
    {"id": "r410", "name": "点心道（天河店）", "cost": 110.0, "type": "早茶餐厅", "rating": 4.6, "recommended_food": "虾饺, 腊味萝卜糕", "queue_time": 12.0, "duration": 80.0},
    {"id": "r411", "name": "八合里海记牛肉店", "cost": 140.0, "type": "粤菜馆", "rating": 4.8, "recommended_food": "潮汕牛肉火锅", "queue_time": 25.0, "duration": 90.0},
    {"id": "r412", "name": "炳胜公馆", "cost": 130.0, "type": "粤菜馆", "rating": 4.9, "recommended_food": "白切鸡, 虾饺", "queue_time": 30.0, "duration": 90.0},
    {"id": "r413", "name": "茶点轩（海珠店）", "cost": 115.0, "type": "早茶餐厅", "rating": 4.6, "recommended_food": "烧卖, 叉烧包", "queue_time": 20.0, "duration": 80.0},
    {"id": "r414", "name": "泮溪酒家", "cost": 145.0, "type": "粤菜馆", "rating": 4.8, "recommended_food": "白切鸡, 早茶", "queue_time": 25.0, "duration": 90.0},
    {"id": "r415", "name": "陶然轩", "cost": 120.0, "type": "早茶餐厅", "rating": 4.6, "recommended_food": "虾饺, 榴莲酥", "queue_time": 15.0, "duration": 80.0},
    {"id": "r429", "name": "文记茶餐厅", "cost": 95.0, "type": "茶餐厅", "rating": 4.5, "recommended_food": "菠萝包, 咖喱牛腩", "queue_time": 8.0, "duration": 75.0},
    {"id": "r430", "name": "点心道（天河店）", "cost": 110.0, "type": "早茶餐厅", "rating": 4.6, "recommended_food": "虾饺, 腊味萝卜糕", "queue_time": 12.0, "duration": 80.0},
    {"id": "r431", "name": "八合里海记牛肉店", "cost": 140.0, "type": "粤菜馆", "rating": 4.8, "recommended_food": "潮汕牛肉火锅", "queue_time": 25.0, "duration": 90.0},
    {"id": "r432", "name": "炳胜公馆", "cost": 130.0, "type": "粤菜馆", "rating": 4.9, "recommended_food": "白切鸡, 虾饺", "queue_time": 30.0, "duration": 90.0},
    {"id": "r433", "name": "茶点轩（海珠店）", "cost": 115.0, "type": "早茶餐厅", "rating": 4.6, "recommended_food": "烧卖, 叉烧包", "queue_time": 20.0, "duration": 80.0},
    {"id": "r434", "name": "泮溪酒家", "cost": 145.0, "type": "粤菜馆", "rating": 4.8, "recommended_food": "白切鸡, 早茶", "queue_time": 25.0, "duration": 90.0},
    {"id": "r435", "name": "陶然轩", "cost": 120.0, "type": "早茶餐厅", "rating": 4.6, "recommended_food": "虾饺, 榴莲酥", "queue_time": 15.0, "duration": 80.0},
    {"id": "r436", "name": "炳胜公馆", "cost": 130.0, "type": "粤菜馆", "rating": 4.9, "recommended_food": "白切鸡, 虾饺", "queue_time": 30.0, "duration": 90.0},
    {"id": "r437", "name": "茶点轩（海珠店）", "cost": 115.0, "type": "早茶餐厅", "rating": 4.6, "recommended_food": "烧卖, 叉烧包", "queue_time": 20.0, "duration": 80.0},
    {"id": "r438", "name": "泮溪酒家", "cost": 145.0, "type": "粤菜馆", "rating": 4.8, "recommended_food": "白切鸡, 早茶", "queue_time": 25.0, "duration": 90.0},
    {"id": "r439", "name": "陶然轩", "cost": 120.0, "type": "早茶餐厅", "rating": 4.6, "recommended_food": "虾饺, 榴莲酥", "queue_time": 15.0, "duration": 80.0},
    # 干扰项（价格高/非粤菜/评分低）
    {"id": "r416", "name": "法餐厅Le Bon Goût", "cost": 480.0, "type": "法餐", "rating": 4.7, "recommended_food": "鹅肝, 牛排", "queue_time": 5.0, "duration": 120.0},
    {"id": "r417", "name": "日式料理店", "cost": 320.0, "type": "日料", "rating": 4.5, "recommended_food": "寿司刺身", "queue_time": 10.0, "duration": 120.0},
    {"id": "r418", "name": "火锅王国", "cost": 200.0, "type": "川菜火锅", "rating": 4.4, "recommended_food": "麻辣火锅", "queue_time": 10.0, "duration": 120.0},
    {"id": "r419", "name": "韩式烤肉屋", "cost": 220.0, "type": "韩餐", "rating": 4.6, "recommended_food": "烤牛肉", "queue_time": 15.0, "duration": 120.0},
    {"id": "r420", "name": "快餐简餐铺", "cost": 50.0, "type": "简餐", "rating": 3.7, "recommended_food": "盖饭", "queue_time": 2.0, "duration": 45.0}
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
print(json.dumps(infra_transportation_info, indent=4, ensure_ascii=False))

def get_mock_data():

    return cross_city_train_departure, cross_city_train_back,{'attractions': attractions, 'accommodations': accommodations, 'restaurants': restaurants}, infra_transportation_info
