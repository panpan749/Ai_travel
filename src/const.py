from mock import get_mock_data,generate_stage
generate_stage([item.travel_days for item in ir.stages])
cross_city_train_departure,cross_city_train_transfer,cross_city_train_back, poi_data, intra_city_trans = get_mock_data()