import random
import json
from mock_data import cross_city_train_departure,cross_city_train_transfer,cross_city_train_back, attractions, restaurants,accommodations

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
                "duration": item["duration"] + round((random.random() - 0.5) * item["duration"], 1),
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
                "duration": item["duration"] + round((random.random() - 0.5) * item["duration"], 1),
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
                "feature": item["feature"],
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

def get_mock_data(is_ground_truth = False ,days = None):
    expand_data(10)
    if days:
        generate_stage(days)
    if is_ground_truth:
        return cross_city_train_departure, cross_city_train_back,{'attractions': attractions[0], 'accommodations': accommodations[0], 'restaurants': restaurants[0]}, infra_transportation_info

    return cross_city_train_departure, cross_city_train_transfer, cross_city_train_back,{'attractions': attractions, 'accommodations': accommodations, 'restaurants': restaurants}, infra_transportation_info

