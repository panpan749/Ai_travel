from IR import *
import pyomo.environ as pyo 
import requests
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

    def field_extract_adapter(self):
        model = self.model




    def make(self, cfg: dynamic_constraint):
        
        FieldNode.eval = self.field_extract_adapter
        
        attraction_dict = self.poi_data['attractions']
        hotel_dict = self.poi_data['accommodations']
        restaurant_dict = self.poi_data['restaurants']
        
        days = range(1,ir.travel_days + 1)
        self.model.days = pyo.Set(initialize=days)
        self.model.attractions = pyo.Set(initialize=attraction_dict.keys())
        self.model.accommodations = pyo.Set(initialize=hotel_dict.keys())
        self.model.restaurants = pyo.Set(initialize=restaurant_dict.keys())
        self.model.train_departure = pyo.Set(initialize=self.cross_city_train_departure.keys())
        self.model.train_back = pyo.Set(initialize=self.cross_city_train_back.keys())

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

        self.model.attr_hotel = pyo.Var(
            self.model.days, self.model.attractions, self.model.accommodations,
            domain=pyo.Binary,
            initialize=0,
            bounds=(0, 1)
        )   
        ## 一致性约束
        def link_attr_hotel_rule1(model, d, a, h):
            return model.attr_hotel[d, a, h] <= model.select_attr[d, a]

        def link_attr_hotel_rule2(model, d, a, h):
            return model.attr_hotel[d, a, h] <= model.select_hotel[h]

        def link_attr_hotel_rule3(model, d, a, h):
            return model.attr_hotel[d, a, h] >= model.select_attr[d, a] + model.select_hotel[h] - 1

        self.model.link_attr_hotel1 = pyo.Constraint(
            self.model.days, self.model.attractions, self.model.accommodations,
            rule=link_attr_hotel_rule1
        )
        self.model.link_attr_hotel2 = pyo.Constraint(
            self.model.days, self.model.attractions, self.model.accommodations,
            rule=link_attr_hotel_rule2
        )
        self.model.link_attr_hotel3 = pyo.Constraint(
            self.model.days, self.model.attractions, self.model.accommodations,
            rule=link_attr_hotel_rule3
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
            rule=lambda m, d:  cfg.hotel_frequency.eval({'day': d}) if d < self.ir.travel_days else sum(m.select_hotel[d, h] for h in m.accommodations) == 0
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
        solver.options = {
            'limits/time': 300,
            'limits/gap': 0,
        }
        return solver
    def solve(self):
        solver = self.configure_solver()
        results = solver.solve(self.model, tee=True)
        return results
    
    def get_solution(self,sovle_result):
        pass


