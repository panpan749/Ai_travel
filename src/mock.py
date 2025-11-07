import random
import json

# 🚄 跨城交通（武汉→成都 出发；成都→洛阳 中转；洛阳→武汉 返程）
cross_city_train_departure = [
    {"origin_id":"wh_001","destination_id":"cd_001","train_number":"G348","duration":"240","cost":"410.0","origin_station":"武汉站","destination_station":"成都东站"},
    {"origin_id":"wh_002","destination_id":"cd_002","train_number":"G308","duration":"260","cost":"420.0","origin_station":"武汉站","destination_station":"成都东站"}
]

cross_city_train_transfer = [
    {"origin_id":"cd_003","destination_id":"ly_001","train_number":"G2206","duration":"420","cost":"520.0","origin_station":"成都东站","destination_station":"洛阳龙门站"},
    {"origin_id":"cd_004","destination_id":"ly_002","train_number":"G314","duration":"440","cost":"540.0","origin_station":"成都东站","destination_station":"洛阳龙门站"}
]

cross_city_train_back = [
    {"origin_id":"ly_003","destination_id":"wh_003","train_number":"Z136","duration":"420","cost":"310.0","origin_station":"洛阳站","destination_station":"武昌站"}
]


# 🎡 景点（多城市嵌套：[[成都景点...],[洛阳景点...]]）
attractions = [
    [
        {"id":"a_cd_001","name":"百花潭公园","cost":0.0,"type":"城市公园","rating":4.7,"duration":120.0},
        {"id":"a_cd_002","name":"人民公园","cost":0.0,"type":"市区休闲","rating":4.6,"duration":90.0},
        {"id":"a_cd_003","name":"宽窄巷子","cost":0.0,"type":"历史街区","rating":4.7,"duration":150.0},
        {"id":"a_cd_004","name":"武侯祠","cost":60.0,"type":"历史遗迹","rating":4.8,"duration":180.0}
    ],
    [
        {"id":"a_ly_001","name":"白马寺","cost":35.0,"type":"佛教圣地","rating":4.8,"duration":180.0},
        {"id":"a_ly_002","name":"洛阳博物馆","cost":0.0,"type":"博物馆","rating":4.7,"duration":150.0},
        {"id":"a_ly_003","name":"丽景门古城","cost":40.0,"type":"历史街区","rating":4.6,"duration":120.0},
        {"id":"a_ly_004","name":"老城十字街","cost":0.0,"type":"夜市街区","rating":4.5,"duration":150.0},
        {"id":"a_ly_005","name":"关林庙","cost":20.0,"type":"历史名胜","rating":4.6,"duration":120.0}
    ]
]


# 🏨 住宿（多城市嵌套：[[成都住宿...],[洛阳住宿...]]）
accommodations = [
    [
        {"id":"h_cd_001","name":"汉庭优佳酒店（春熙路店）","cost":280.0,"type":"经济型连锁","rating":4.6,"feature":"含早餐, 地铁直达"},
        {"id":"h_cd_002","name":"如家精选（宽窄巷子店）","cost":320.0,"type":"经济型连锁","rating":4.7,"feature":"含早餐, 交通便利"},
        {"id":"h_cd_003","name":"桔子水晶（人民公园店）","cost":400.0,"type":"舒适型","rating":4.8,"feature":"含早餐, 环境安静"}
    ],
    [
        {"id":"h_ly_001","name":"洛阳非凡·云联酒店","cost":160.0,"type":"舒适型","rating":4.8,"feature":"含早餐, 市中心, 临近白马寺"},
        {"id":"h_ly_002","name":"锦江之星（洛阳火车站店）","cost":280.0,"type":"经济型连锁","rating":4.5,"feature":"含早餐, 交通便利"},
        {"id":"h_ly_003","name":"全季酒店（洛阳王城公园店）","cost":420.0,"type":"舒适型","rating":4.7,"feature":"含早餐, 靠近景区"}
    ]
]


# 🍜 餐厅（多城市嵌套：[[成都餐厅...],[洛阳餐厅...]]）
# 成都（为期3天：提供≥9家满足 + ≥3家干扰）
# 洛阳（为期5天：提供≥15家满足 + ≥5家干扰）——本次已严格满足你的数量要求

restaurants = [
    [
        # —— 成都 满足条件（地道川味，人均亲民）9家 ——
        {"id":"r_cd_001","name":"老码头火锅","cost":120.0,"type":"火锅","rating":4.7,"recommended_food":"牛油锅, 毛肚","queue_time":15.0,"duration":90.0},
        {"id":"r_cd_002","name":"谭鸭血火锅（春熙路）","cost":130.0,"type":"火锅","rating":4.8,"recommended_food":"鸭血, 千层毛肚","queue_time":20.0,"duration":100.0},
        {"id":"r_cd_003","name":"钵钵鸡传奇","cost":60.0,"type":"川味小吃","rating":4.6,"recommended_food":"钵钵鸡, 凉粉","queue_time":10.0,"duration":60.0},
        {"id":"r_cd_004","name":"陈麻婆豆腐馆","cost":90.0,"type":"川菜","rating":4.7,"recommended_food":"麻婆豆腐, 回锅肉","queue_time":10.0,"duration":90.0},
        {"id":"r_cd_005","name":"小龙坎老火锅","cost":130.0,"type":"火锅","rating":4.8,"recommended_food":"麻辣牛肉, 黄喉","queue_time":25.0,"duration":120.0},
        {"id":"r_cd_006","name":"钢管厂五区小郡肝串串香","cost":80.0,"type":"串串香","rating":4.6,"recommended_food":"郡肝串, 藕片","queue_time":15.0,"duration":90.0},
        {"id":"r_cd_007","name":"夫妻肺片总店","cost":90.0,"type":"川菜","rating":4.8,"recommended_food":"夫妻肺片, 担担面","queue_time":10.0,"duration":80.0},
        {"id":"r_cd_008","name":"龙抄手总店","cost":50.0,"type":"小吃","rating":4.6,"recommended_food":"龙抄手, 钟水饺","queue_time":8.0,"duration":60.0},
        {"id":"r_cd_009","name":"张老坎串串香","cost":100.0,"type":"串串香","rating":4.6,"recommended_food":"牛肉串, 豆皮","queue_time":12.0,"duration":90.0},

        # —— 成都 干扰（价高/非川味/评分低）3家 ——
        {"id":"r_cd_010","name":"高端法餐Le Ciel","cost":520.0,"type":"法餐","rating":4.8,"recommended_food":"鹅肝","queue_time":5.0,"duration":120.0},
        {"id":"r_cd_011","name":"日料铁板烧","cost":360.0,"type":"日料","rating":4.5,"recommended_food":"刺身","queue_time":10.0,"duration":100.0},
        {"id":"r_cd_012","name":"清汤串串","cost":70.0,"type":"串串香","rating":3.8,"recommended_food":"清汤串串","queue_time":2.0,"duration":60.0}
    ],
    [
        # —— 洛阳 满足条件（地道/亲民，含两家指定）至少15家 ——
        {"id":"r_ly_001","name":"鲁记卤肉凉菜","cost":45.0,"type":"地方菜","rating":4.7,"recommended_food":"卤肉拼盘, 凉拌菜","queue_time":10.0,"duration":60.0},
        {"id":"r_ly_002","name":"鲜羊肉汤店","cost":55.0,"type":"地方菜","rating":4.8,"recommended_food":"羊肉汤, 油饼","queue_time":15.0,"duration":70.0},
        {"id":"r_ly_003","name":"洛阳水席馆","cost":80.0,"type":"豫菜","rating":4.6,"recommended_food":"水席全套","queue_time":20.0,"duration":90.0},
        {"id":"r_ly_004","name":"真不同饭店（牡丹店）","cost":95.0,"type":"豫菜","rating":4.7,"recommended_food":"牡丹燕菜","queue_time":15.0,"duration":90.0},
        {"id":"r_ly_005","name":"老城十字街胡辣汤","cost":25.0,"type":"小吃","rating":4.6,"recommended_food":"胡辣汤, 油条","queue_time":8.0,"duration":40.0},
        {"id":"r_ly_006","name":"不翻汤老店","cost":35.0,"type":"小吃","rating":4.6,"recommended_food":"不翻汤","queue_time":10.0,"duration":50.0},
        {"id":"r_ly_007","name":"牛肉汤老字号","cost":45.0,"type":"地方菜","rating":4.7,"recommended_food":"牛肉汤, 肉夹馍","queue_time":12.0,"duration":60.0},
        {"id":"r_ly_008","name":"十字街面馆","cost":30.0,"type":"面食","rating":4.5,"recommended_food":"烩面, 羊杂汤","queue_time":8.0,"duration":50.0},
        {"id":"r_ly_009","name":"浆面条馆","cost":28.0,"type":"面食","rating":4.5,"recommended_food":"浆面条","queue_time":6.0,"duration":45.0},
        {"id":"r_ly_010","name":"驴肉火烧铺","cost":40.0,"type":"小吃","rating":4.6,"recommended_food":"驴肉火烧","queue_time":10.0,"duration":45.0},
        {"id":"r_ly_011","name":"羊肉烩面坊","cost":36.0,"type":"面食","rating":4.6,"recommended_food":"羊肉烩面","queue_time":10.0,"duration":60.0},
        {"id":"r_ly_012","name":"洛阳焖饼馆","cost":32.0,"type":"面食","rating":4.5,"recommended_food":"焖饼","queue_time":8.0,"duration":50.0},
        {"id":"r_ly_013","name":"烧鸡店（卤味）","cost":48.0,"type":"卤味","rating":4.6,"recommended_food":"道口烧鸡","queue_time":12.0,"duration":60.0},
        {"id":"r_ly_014","name":"小碗卤肉饭","cost":22.0,"type":"快餐","rating":4.5,"recommended_food":"卤肉饭","queue_time":5.0,"duration":40.0},
        {"id":"r_ly_015","name":"老城锅贴铺","cost":24.0,"type":"小吃","rating":4.6,"recommended_food":"锅贴, 酸辣汤","queue_time":6.0,"duration":45.0},

        # —— 洛阳 干扰项（≥5：价高/非地道/评分低等）——
        {"id":"r_ly_016","name":"西餐牛排馆","cost":280.0,"type":"西餐","rating":4.8,"recommended_food":"牛排","queue_time":5.0,"duration":120.0},
        {"id":"r_ly_017","name":"日式寿司屋","cost":210.0,"type":"日料","rating":4.5,"recommended_food":"刺身","queue_time":8.0,"duration":90.0},
        {"id":"r_ly_018","name":"高端融合餐厅","cost":260.0,"type":"融合菜","rating":4.6,"recommended_food":"黑松露意面","queue_time":10.0,"duration":120.0},
        {"id":"r_ly_019","name":"清汤羊汤铺","cost":28.0,"type":"地方菜","rating":3.7,"recommended_food":"羊汤","queue_time":2.0,"duration":40.0},
        {"id":"r_ly_020","name":"重油快餐店","cost":26.0,"type":"快餐","rating":3.6,"recommended_food":"盖饭","queue_time":3.0,"duration":40.0}
    ]
]


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
def get_mock_data():

    return cross_city_train_departure, cross_city_train_transfer, cross_city_train_back,{'attractions': attractions, 'accommodations': accommodations, 'restaurants': restaurants}, infra_transportation_info
