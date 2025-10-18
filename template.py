from IR import *
import pyomo.environ as pyo 
import requests
from datetime import time, timedelta, datetime

ir  = IR()

origin_city = ir.original_city
destination_city = ir.destinate_city 
def fetch_data():
    url = "http://localhost:12457"
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

def rough_rank():
    pass
    # return cross_city_train_departure, cross_city_train_back, poi_data, intra_city_trans

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
    

class template:

    model: pyo.Model
    ir: IR
    cfg: dynamic_constraint
    cross_city_train_departure: dict
    cross_city_train_back: dict
    poi_data: dict
    intra_city_trans: dict

    def __init__(self,cross_city_train_departure, cross_city_train_back,poi_data,intra_city_trans,ir,model = None):
        if not model:
            self.model = pyo.ConcreteModel()
        else: self.model = model

        self.cross_city_train_departure = cross_city_train_departure
        self.cross_city_train_back = cross_city_train_back
        self.poi_data = poi_data
        self.intra_city_trans = intra_city_trans
        self.ir = ir



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
        return transport_cost
    def make(self, cfg: dynamic_constraint):
        outer_self = self
        self.cfg = cfg
        pois = [a['id'] for a in self.poi_data['attractions']] + [h['id'] for h in self.poi_data['accommodations']]
        def field_extract_adapter(self:FieldNode,context: dict):
            self.field = self.field.lower()
            if self.field == 'num_attractions_per_day':
                day = context.get('day', 1)
                return sum(outer_self.model.select_attr[day,a] for a in outer_self.model.attractions)
            elif self.field == 'num_restaurants_per_day':
                day = context.get('day', 1)
                return sum(outer_self.model.select_rest[day,r] for r in outer_self.model.restaurants)
            elif self.field == 'num_hotels_per_day':
                day = context.get('day', 1)
                return sum(outer_self.model.select_hotel[day,h] for h in outer_self.model.accommodations)
            elif self.field == 'daily_total_time':
                day = context.get('day', 1)
                return outer_self.get_daily_total_time(day)
                 
            elif self.field == 'daily_queue_time':
                day = context.get('day', 1)
                return outer_self.get_daily_queue_time(day)
            
            elif self.field == 'daily_total_restaurant_time':
                day = context.get('day', 1)
                return outer_self.get_daily_total_restaurant_time(day)
            
            elif self.field == 'daily_transportation_time':
                day = context.get('day', 1)
                return outer_self.get_daily_total_transportation_time(day)
            
            elif self.field == 'total_active_time':
                sum_time = 0
                for day in range(ir.travel_days):
                    sum_time += outer_self.get_daily_total_time(day + 1) 
                return sum_time  
            elif self.field == 'total_transportation_time':
                sum_time = 0
                for day in range(ir.travel_days):
                    sum_time += outer_self.get_daily_total_transportation_time(day + 1) 
                return sum_time
            elif self.field == 'total_queue_time':
                sum_time = 0
                for day in range(ir.travel_days):
                    sum_time += outer_self.get_daily_queue_time(day + 1) 
                return sum_time
            elif self.field == 'total_restaurant_time':
                sum_time = 0
                for day in range(ir.travel_days):
                    sum_time += outer_self.get_daily_total_restaurant_time(day + 1) 
                return sum_time
            elif self.field == 'total_cost': 
                return sum(outer_self.get_daily_total_cost(day) for day in ir.travel_days)
            elif self.field == 'total_hotel_cost':
                return sum(outer_self.get_daily_total_hotel_cost(day) for day in ir.travel_days)
            elif self.field == 'total_attraction_cost':
                return sum(outer_self.get_daily_total_attraction_cost(day) for day in ir.travel_days)
            elif self.field == 'total_restaurant_cost':
                return sum(outer_self.get_daily_total_restaurant_cost(day) for day in ir.travel_days)
            elif self.field == 'total_transportation_cost':
                return sum(outer_self.get_daily_total_transportation_cost(day) for day in ir.travel_days)
            elif self.field == 'daily_total_cost':
                day = context.get('day', 1)
                return outer_self.get_daily_total_cost(day)
            elif self.field == 'daily_total_attraction_cost':
                day = context.get('day', 1)
                return outer_self.get_daily_total_attraction_cost(day)
            elif self.field == 'daily_total_restaurant_cost':
                day = context.get('day', 1)
                return outer_self.get_daily_total_restaurant_cost(day)
            elif self.field == 'daily_total_hotel_cost':
                day = context.get('day', 1)
                return outer_self.get_daily_total_hotel_cost(day)
            elif self.field == 'daily_total_transportation_cost':
                day = context.get('day', 1)
                return outer_self.get_daily_total_transportation_cost(day)
            
        FieldNode.eval = field_extract_adapter

        attraction_dict = self.poi_data['attractions'] ## {'attractions':{'id_1':{...},'id_2':{...},...}}
        hotel_dict = self.poi_data['accommodations']
        restaurant_dict = self.poi_data['restaurants']
        
        days = range(1,ir.travel_days + 1)
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
            }
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
            }
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
            }
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
            }
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
            }
        )

        ## variables
        self.model.select_hotel = pyo.Var(self.model.days, self.model.accommodations, domain=pyo.Binary)
        self.model.select_attr = pyo.Var(self.model.days, self.model.attractions, domain=pyo.Binary)
        self.model.select_rest = pyo.Var(self.model.days, self.model.restaurants, domain=pyo.Binary)
        self.model.trans_mode = pyo.Var(self.model.days, domain=pyo.Binary) # 1为公交 0为打车
        self.model.select_train_departure = pyo.Var(self.model.train_departure, domain=pyo.Binary)
        self.model.select_train_back = pyo.Var(self.model.train_back, domain=pyo.Binary)

        ## last day hotel constraint
        if ir.travel_days > 1:
            def last_day_hotel_constraint(model,h):
                N = ir.travel_days
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

        def u_rule(m,d,p):
            M = len(self.model.attractions)
            return m.select_attr[d,p] <= m.u[d,p] <= M * m.select_attr[d, p]
        
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
        self.model.u_rule = pyo.Constraint(
            self.model.days, self.model.attractions,
            rule=u_rule
        )
        
        self.model.unique_attr = pyo.Constraint(
            self.model.attractions,
            rule=lambda m, a: sum(m.select_attr[d, a] for d in m.days) <= 1
        )

        self.model.unique_rest = pyo.Constraint(
            self.model.restaurants,
            rule=lambda m, r: sum(m.select_rest[d, r] for d in m.days) <= 1
        )
        ##约束1
        self.model.attr_num = pyo.Constraint(
            self.model.days,
            rule=lambda m, d: cfg.num_attractions_per_day.eval({'day': d})
        )

        self.model.rest_num = pyo.Constraint(
            self.model.days,
            rule=lambda m, d: cfg.meal_frequency.eval({'day': d})
        )

        self.model.hotel_num = pyo.Constraint(
            self.model.days,
            rule=lambda m, d:  cfg.hotel_frequency.eval({'day': d})
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
            self.model.daily_time_rule = pyo.Constraint(
                self.model.days,
                rule=lambda m, d: cfg.daily_total_time.eval({'day': d})
            )
        
        ## 用户约束
        if cfg.daily_queue_time :
            self.model.daily_queue_time_rule = pyo.Constraint(
                self.model.days,
                rule=lambda m, d: cfg.daily_queue_time.eval({'day': d})
            )
        
        if cfg.daily_total_meal_time :
            self.model.daily_total_meal_time_rule = pyo.Constraint(
                self.model.days,
                rule=lambda m, d: cfg.daily_total_meal_time.eval({'day': d})
            )
        
        if cfg.daily_transportation_time :
            self.model.daily_transportation_time_rule = pyo.Constraint(
                self.model.days,
                rule=lambda m, d: cfg.daily_transportation_time.eval({'day': d})
            )

        if cfg.total_active_time :
            self.model.total_active_time_rule = pyo.Constraint(
                rule=lambda m: cfg.total_active_time.eval({})
            )

        if cfg.total_resturant_time:
            self.model.total_resturant_time_rule = pyo.Constraint(
                rule=lambda m: cfg.total_resturant_time.eval({})
            )

        if cfg.total_queue_time :
            self.model.total_queue_time_rule = pyo.Constraint(
                rule=lambda m: cfg.total_queue_time.eval({})
            )
        
        if cfg.total_transportation_time :
            self.model.total_transportation_time_rule = pyo.Constraint(
                rule=lambda m: cfg.total_transportation_time.eval({})
            )
        
        if cfg.total_budget :
            self.model.total_budget_rule = pyo.Constraint(
                rule=lambda m: cfg.total_budget.eval({})
            )
        
        if cfg.total_meal_budget :
            self.model.total_meal_budget_rule = pyo.Constraint(
                rule=lambda m: cfg.total_meal_budget.eval({})
            )
        
        if cfg.total_attraction_ticket_budget :
            self.model.total_attraction_ticket_budget_rule = pyo.Constraint(
                rule=lambda m: cfg.total_attraction_ticket_budget.eval({})
            )
        
        if cfg.total_hotel_budget :
            self.model.total_hotel_budget_rule = pyo.Constraint(
                rule=lambda m: cfg.total_hotel_budget.eval({})
            )
        
        if cfg.total_transportation_budget :
            self.model.total_transportation_budget_rule = pyo.Constraint(
                rule=lambda m: cfg.total_transportation_budget.eval({})
            )
        
        if cfg.daily_total_budget:
            self.model.daily_total_budget_rule = pyo.Constraint(
                self.model.days,
                rule=lambda m, d: cfg.daily_total_budget.eval({'day': d})
            )
        
        if cfg.daily_total_meal_budget:
            self.model.daily_total_meal_budget_rule = pyo.Constraint(
                self.model.days,
                rule=lambda m, d: cfg.daily_total_meal_budget.eval({'day': d})
            )
        
        if cfg.daily_total_attraction_ticket_budget:
            self.model.daily_total_attraction_ticket_budget_rule = pyo.Constraint(
                self.model.days,
                rule=lambda m, d: cfg.daily_total_attraction_ticket_budget.eval({'day': d})
            )
        
        if cfg.daily_total_hotel_budget:
            self.model.daily_total_hotel_budget_rule = pyo.Constraint(
                self.model.days,
                rule=lambda m, d: cfg.daily_total_hotel_budget.eval({'day': d})
            )
        
        if cfg.daily_total_transportation_budget:
            self.model.daily_total_transportation_budget_rule = pyo.Constraint(
                self.model.days,
                rule=lambda m, d: cfg.daily_total_transportation_budget.eval({'day': d})
            )

        eval(cfg.extra)

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
        return selected_train[0]


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
        if day <= 0 or day >= self.ir.travel_days:
            return 'null'
        model = self.model
        selected_hotel = [
            model.hotel_data[t]
            for t in model.accommodations
            if pyo.value(model.select_hotel[day,t]) > 0.9
        ]
        return selected_hotel[0]


    def get_time(self, selected_attr, selected_rest, selected_hotel, day, intra_city_trans):
        model = self.model
        daily_time = 0
        daily_time += selected_attr['duration']
        for r in selected_rest:
            daily_time += r['queue_time'] +r['duration']

        if pyo.value(model.trans_mode[day]) > 0.9:
            transport_time = get_trans_params(
                intra_city_trans,
                selected_hotel['id'],
                selected_attr['id'],
                'bus_duration'
            ) + get_trans_params(
                intra_city_trans,
                selected_attr['id'],
                selected_hotel['id'],
                'bus_duration'
            )
        else:
            transport_time = get_trans_params(
                intra_city_trans,
                selected_hotel['id'],
                selected_attr['id'],
                'taxi_duration'
            ) + get_trans_params(
                intra_city_trans,
                selected_attr['id'],
                selected_hotel['id'],
                'taxi_duration'
            )

        return daily_time + transport_time, transport_time


    def get_cost(self, model, selected_attr, selected_rest, departure_trains, back_trains, selected_hotel, day, intra_city_trans):
        daily_cost = 0
        peoples = self.ir.peoples
        daily_cost += peoples * selected_attr['cost']
        travel_days = self.ir.travel_days
        for r in selected_rest:
            daily_cost += peoples * r['cost']

        if pyo.value(model.trans_mode[day]) > 0.9:
            transport_cost = peoples * get_trans_params(
                intra_city_trans,
                selected_hotel['id'],
                selected_attr['id'],
                'bus_cost'
            ) + peoples * get_trans_params(
                intra_city_trans,
                selected_attr['id'],
                selected_hotel['id'],
                'bus_cost'
            )
        else:
            transport_cost = get_trans_params(
                intra_city_trans,
                selected_hotel['id'],
                selected_attr['id'],
                'taxi_cost'
            ) + get_trans_params(
                intra_city_trans,
                selected_attr['id'],
                selected_hotel['id'],
                'taxi_cost'
            )

        if day != travel_days:
            daily_cost += selected_hotel['cost']
        if day == 1:
            daily_cost += peoples * departure_trains['cost']
        if day == travel_days:
            daily_cost += peoples * back_trains['cost']
        return daily_cost + transport_cost, transport_cost

    def generate_daily_plan(self):
        model = self.model
        intra_city_trans = self.intra_city_trans
        departure_trains = self.get_selected_train(model, 'departure')
        back_trains = self.get_selected_train(model, 'back')
        selected_hotel = self.get_selected_hotel(model)
        total_cost = 0
        daily_plans = []
        select_at = []
        select_re = []
        travel_days = self.ir.travel_days
        date = self.generate_date_range(self.ir.start_date) ##todo
        for day in sorted(model.days):
            attr_details = []
            attr_details = self.get_selected_poi(model, 'attraction', day, select_at)[0]
            select_at.append(attr_details['id'])
            rest_details = []
            rest_details = self.get_selected_poi(model, 'restaurant', day, select_re)
            for r in rest_details:
                select_re.append(r['id'])
            meal_allocation = {
                'breakfast': rest_details[0],
                'lunch': rest_details[1],
                'dinner': rest_details[2]
            }

            daily_time, transport_time = self.get_time(model, attr_details, rest_details, departure_trains, back_trains, selected_hotel, day, intra_city_trans)
            daily_cost, transport_cost = self.get_cost(model, attr_details, rest_details, departure_trains, back_trains, selected_hotel, day, intra_city_trans)
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
    def get_solution(self,sovle_result):
        pass


