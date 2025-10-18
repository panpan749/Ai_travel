SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

-- ----------------------------
-- Table structure for poi_master
-- ----------------------------
DROP TABLE IF EXISTS `poi_master`;
CREATE TABLE `poi_master`  (
  `poi_id` varchar(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `poi_type` enum('starting','accommodation','transport','restaurant','attraction') CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL,
  PRIMARY KEY (`poi_id`) USING BTREE
) ENGINE = InnoDB CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci COMMENT = 'POI统一入口表' ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Table structure for city_info
-- ----------------------------
DROP TABLE IF EXISTS `city_info`;
CREATE TABLE `city_info`  (
  `city_id` varchar(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `up_ad_name` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL COMMENT '省级区域名称',
  `up_ad_code` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL COMMENT '省级区域代码',
  `cityname` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `adcode` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL COMMENT '具体区域代码',
  `citycode` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL COMMENT '市级区域代码',
  PRIMARY KEY (`city_id`) USING BTREE
) ENGINE = InnoDB CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Table structure for path_planning_cross_city
-- ----------------------------
DROP TABLE IF EXISTS `path_planning_cross_city`;
CREATE TABLE `path_planning_cross_city`  (
  `origin_id` varchar(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL COMMENT '起点ID（关联交通表） 关联poi_master.poi_id',
  `destination_id` varchar(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL COMMENT '终点ID（关联交通表） 关联poi_master.poi_id',
  `train_plan` json NULL COMMENT '火车方案',
  `airplane_plan` json NULL COMMENT '飞机方案',
  `origin_type` enum('火车站','机场','汽车站') CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `destination_type` enum('火车站','机场','汽车站') CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  PRIMARY KEY (`origin_id`, `destination_id`) USING BTREE,
  INDEX `destination_id`(`destination_id` ASC) USING BTREE,
  CONSTRAINT `path_planning_cross_city_ibfk_1` FOREIGN KEY (`origin_id`) REFERENCES `poi_master` (`poi_id`) ON DELETE RESTRICT ON UPDATE RESTRICT,
  CONSTRAINT `path_planning_cross_city_ibfk_2` FOREIGN KEY (`destination_id`) REFERENCES `poi_master` (`poi_id`) ON DELETE RESTRICT ON UPDATE RESTRICT
) ENGINE = InnoDB CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci COMMENT = '跨市路径规划表' ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Table structure for path_planning_in_city
-- ----------------------------
DROP TABLE IF EXISTS `path_planning_in_city`;
CREATE TABLE `path_planning_in_city`  (
  `origin_id` varchar(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL COMMENT '起点ID（关联起点/住宿/交通/美食/景点表） 关联poi_master.poi_id',
  `destination_id` varchar(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL COMMENT '终点ID（关联起点/住宿/交通/美食/景点表） 关联poi_master.poi_id',
  `distance` varchar(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '距离(m)',
  `taxi_duration` varchar(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '打车时间花费(min)',
  `taxi_cost` varchar(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '打车花费(rmb)',
  `bus_duration` varchar(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '公交时间花费',
  `bus_cost` varchar(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '公交花费',
  `walk_duration` varchar(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '步行时间花费',
  `origin_type` enum('起点','景点','住宿','饭店') CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `destination_type` enum('起点','景点','住宿','饭店') CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `city_name` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '城市名',
  `citycode` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL,
  PRIMARY KEY (`origin_id`, `destination_id`) USING BTREE,
  INDEX `destination_id`(`destination_id` ASC) USING BTREE,
  CONSTRAINT `path_planning_in_city_ibfk_1` FOREIGN KEY (`origin_id`) REFERENCES `poi_master` (`poi_id`) ON DELETE RESTRICT ON UPDATE RESTRICT,
  CONSTRAINT `path_planning_in_city_ibfk_2` FOREIGN KEY (`destination_id`) REFERENCES `poi_master` (`poi_id`) ON DELETE RESTRICT ON UPDATE RESTRICT
) ENGINE = InnoDB CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci COMMENT = '市内路径规划表' ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Table structure for poi_accommodation
-- ----------------------------
DROP TABLE IF EXISTS `poi_accommodation`;
CREATE TABLE `poi_accommodation`  (
  `accommodation_id` varchar(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `accommodation_name` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `accommodation_address` varchar(500) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `accommodation_type` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `city_name` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '城市名',
  `avg_price` decimal(8, 2) NULL DEFAULT NULL COMMENT '平均价格',
  `rating` decimal(3, 2) NULL DEFAULT NULL,
  `feature_hotel_type` text CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL COMMENT '酒店特色',
  `longitude` text CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL COMMENT '经度',
  `latitude` text CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL COMMENT '纬度',
  `citycode` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL,
  PRIMARY KEY (`accommodation_id`) USING BTREE,
  CONSTRAINT `poi_accommodation_chk_1` CHECK (`rating` between 0 and 5)
) ENGINE = InnoDB CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Table structure for poi_attraction
-- ----------------------------
DROP TABLE IF EXISTS `poi_attraction`;
CREATE TABLE `poi_attraction`  (
  `attraction_id` varchar(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `attraction_name` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL COMMENT '景点名称',
  `attraction_address` varchar(500) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL COMMENT '详细地址',
  `avg_consumption` decimal(10, 2) NULL DEFAULT NULL COMMENT '平均消费（单位：元）',
  `attraction_type` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '景点类型',
  `rating` decimal(3, 2) NULL DEFAULT NULL COMMENT '评分（0-5分）',
  `city_name` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '城市名',
  `longitude` text CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL COMMENT '经度',
  `latitude` text CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL COMMENT '纬度',
  `open_time` time NULL DEFAULT NULL COMMENT '开店时间',
  `close_time` time NULL DEFAULT NULL COMMENT '关店时间',
  `suggested_duration` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '建立游玩时长(单位分钟)',
  `citycode` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL,
  PRIMARY KEY (`attraction_id`) USING BTREE,
  CONSTRAINT `poi_attraction_chk_1` CHECK (`rating` between 0 and 5)
) ENGINE = InnoDB CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Table structure for poi_city
-- ----------------------------
DROP TABLE IF EXISTS `poi_city`;
CREATE TABLE `poi_city`  (
  `city_code` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL COMMENT '城市代码',
  `city_name` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL COMMENT '城市名称',
  PRIMARY KEY (`city_code`) USING BTREE
) ENGINE = InnoDB CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci ROW_FORMAT = Dynamic;


-- ----------------------------
-- Table structure for poi_restaurant
-- ----------------------------
DROP TABLE IF EXISTS `poi_restaurant`;
CREATE TABLE `poi_restaurant`  (
  `restaurant_id` varchar(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `restaurant_name` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `restaurant_address` varchar(500) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `restaurant_type` varchar(500) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `city_name` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '城市名',
  `avg_price` decimal(8, 2) NULL DEFAULT NULL COMMENT '人均消费',
  `rating` decimal(3, 2) NULL DEFAULT NULL,
  `business_hours` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '营业时间段',
  `recommended_food` text CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL COMMENT '推荐菜品',
  `queue_time` int NULL DEFAULT NULL COMMENT '平均排队时间（分钟）',
  `consumption_time` int NULL DEFAULT NULL COMMENT '平均消费时间（分钟）',
  `longitude` text CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL COMMENT '经度',
  `latitude` text CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL COMMENT '纬度',
  `citycode` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL,
  PRIMARY KEY (`restaurant_id`) USING BTREE,
  CONSTRAINT `poi_restaurant_chk_1` CHECK (`rating` between 0 and 5)
) ENGINE = InnoDB CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Table structure for poi_starting_point
-- ----------------------------
DROP TABLE IF EXISTS `poi_starting_point`;
CREATE TABLE `poi_starting_point`  (
  `starting_point_id` varchar(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `point_name` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `point_address` varchar(500) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `city_name` varchar(500) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `longitude` text CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL COMMENT '经度',
  `latitude` text CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL COMMENT '纬度',
  `citycode` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL,
  PRIMARY KEY (`starting_point_id`) USING BTREE
) ENGINE = InnoDB CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Table structure for poi_transport
-- ----------------------------
DROP TABLE IF EXISTS `poi_transport`;
CREATE TABLE `poi_transport`  (
  `transport_id` varchar(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `transport_name` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL COMMENT '站点名称',
  `transport_address` varchar(500) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `transport_type` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `city_name` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '城市名',
  `open_time` time NULL DEFAULT NULL COMMENT '开放时间',
  `close_time` time NULL DEFAULT NULL COMMENT '关闭时间',
  `longitude` text CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL COMMENT '经度',
  `latitude` text CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL COMMENT '纬度',
  `citycode` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL,
  `distance_type` enum('长途','短途') CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL,
  PRIMARY KEY (`transport_id`) USING BTREE
) ENGINE = InnoDB CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Table structure for poi_type
-- ----------------------------
DROP TABLE IF EXISTS `poi_type`;
CREATE TABLE `poi_type`  (
  `type_id` varchar(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `type_code` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `big_category` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `mid_category` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `sub_category` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  PRIMARY KEY (`type_id`) USING BTREE
) ENGINE = InnoDB CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Table structure for train_info
-- ----------------------------
DROP TABLE IF EXISTS `train_info`;
CREATE TABLE `train_info`  (
  `origin_id` varchar(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL COMMENT '起点ID（关联交通表） 关联poi_master.poi_id',
  `destination_id` varchar(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL COMMENT '终点ID（关联交通表） 关联poi_master.poi_id',
  `train_number` varchar(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '车次',
  `duration` varchar(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '乘坐时间',
  `price` varchar(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '票价',
  `origin_station` varchar(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '出发站',
  `origin_city` varchar(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '出发城市',
  `origin_city_code` varchar(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '出发城市',
  `destination_station` varchar(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '目的站',
  `destination_city` varchar(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '目的城市',
  `destination_city_code` varchar(16) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '目的城市',
  PRIMARY KEY (`origin_id`, `destination_id`) USING BTREE,
  INDEX `destination_id`(`destination_id` ASC) USING BTREE,
  CONSTRAINT `train_info_ibfk_1` FOREIGN KEY (`origin_id`) REFERENCES `poi_master` (`poi_id`) ON DELETE RESTRICT ON UPDATE RESTRICT,
  CONSTRAINT `train_info_ibfk_2` FOREIGN KEY (`destination_id`) REFERENCES `poi_master` (`poi_id`) ON DELETE RESTRICT ON UPDATE RESTRICT
) ENGINE = InnoDB CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci COMMENT = '火车信息表' ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Triggers structure for table poi_accommodation
-- ----------------------------
DROP TRIGGER IF EXISTS `trig_accommodation`;
delimiter ;;
CREATE TRIGGER `trig_accommodation` AFTER INSERT ON `poi_accommodation` FOR EACH ROW BEGIN
    INSERT INTO poi_master(poi_id, poi_type) 
    VALUES (NEW.accommodation_id, 'accommodation');
END
;;
delimiter ;

-- ----------------------------
-- Triggers structure for table poi_attraction
-- ----------------------------
DROP TRIGGER IF EXISTS `trig_attraction`;
delimiter ;;
CREATE TRIGGER `trig_attraction` AFTER INSERT ON `poi_attraction` FOR EACH ROW BEGIN
    INSERT INTO poi_master(poi_id, poi_type) 
    VALUES (NEW.attraction_id, 'attraction');
END
;;
delimiter ;

-- ----------------------------
-- Triggers structure for table poi_restaurant
-- ----------------------------
DROP TRIGGER IF EXISTS `trig_restaurant`;
delimiter ;;
CREATE TRIGGER `trig_restaurant` AFTER INSERT ON `poi_restaurant` FOR EACH ROW BEGIN
    INSERT INTO poi_master(poi_id, poi_type) 
    VALUES (NEW.restaurant_id, 'restaurant');
END
;;
delimiter ;

-- ----------------------------
-- Triggers structure for table poi_starting_point
-- ----------------------------
DROP TRIGGER IF EXISTS `trig_starting_point`;
delimiter ;;
CREATE TRIGGER `trig_starting_point` AFTER INSERT ON `poi_starting_point` FOR EACH ROW BEGIN
    INSERT INTO poi_master(poi_id, poi_type) 
    VALUES (NEW.starting_point_id, 'starting');
END
;;
delimiter ;

-- ----------------------------
-- Triggers structure for table poi_transport
-- ----------------------------
DROP TRIGGER IF EXISTS `trig_transport`;
delimiter ;;
CREATE TRIGGER `trig_transport` AFTER INSERT ON `poi_transport` FOR EACH ROW BEGIN
    INSERT INTO poi_master(poi_id, poi_type) 
    VALUES (NEW.transport_id, 'transport');
END
;;
delimiter ;

SET FOREIGN_KEY_CHECKS = 1;

-- ----------------------------
-- Records of poi_transport
-- ----------------------------
INSERT INTO `poi_transport` VALUES ('03b4588ee2694c67', '贵阳北站', '观山街道创新东路与西二环交叉口东北侧', '火车站', '贵阳市', '00:00:00', '24:00:00', '106.674451', '26.619442', '0851', '长途');
INSERT INTO `poi_transport` VALUES ('2e8d724048d24f78', '苏州北站', '城通路', '火车站', '苏州市', '00:00:00', '24:00:00', '120.644301', '31.421548', '0512', '长途');
INSERT INTO `poi_transport` VALUES ('52712064787a4566', '杭州东站', '天城路1号', '火车站', '杭州市', '00:00:00', '24:00:00', '120.212600', '30.290851', '0571', '长途');
INSERT INTO `poi_transport` VALUES ('615cd5f1022b41f4', '青岛西站', '铁山街道海西三路', '火车站', '青岛市', '00:00:00', '24:00:00', '119.956267', '35.901455', '0532', '长途');

-- ----------------------------
-- Records of poi_accommodation
-- ----------------------------
INSERT INTO `poi_accommodation` VALUES ('02c1a8fc77dc41e5', '杭州九里云松度假酒店', '灵隐路18-8号', '五星级宾馆', '杭州市', 1763.00, 4.70, '松鼠桂鱼,自助早餐,点心,双人餐,下午茶', '120.113209', '30.244454', '0571');
INSERT INTO `poi_accommodation` VALUES ('02e6a7a285e24e48', '杭州滨江开元名都大酒店', '火炬大道岩大房巷59号', '宾馆酒店', '杭州市', 670.00, 4.70, '自助餐', '120.168237', '30.169134', '0571');
INSERT INTO `poi_accommodation` VALUES ('0315e00908b54fac', '安和隐世酒店', '越王路2077号湘湖越界X-LIVIN内街', '宾馆酒店', '杭州市', 528.00, 4.30, '无', '120.209771', '30.137105', '0571');

-- ----------------------------
-- Records of poi_attraction
-- ----------------------------
INSERT INTO `poi_attraction` VALUES ('060cb7c1ecb54fd9', '思澄堂', '解放路132号(近中山中路)', 0.00, '教堂', 4.50, '杭州市', '120.172971', '30.251002', '08:30:00', '11:30:00', '90', '0571');
INSERT INTO `poi_attraction` VALUES ('11a543aed7f14b98', '西溪国家湿地公园洪园', '五常大道龙舌嘴入口(西溪湿地西区)', 80.00, '国家级景点', 4.60, '杭州市', '120.050588', '30.261234', '08:30:00', '17:30:00', '30', '0571');
INSERT INTO `poi_attraction` VALUES ('3922a00eb0674ead', '杭州西湖风景名胜区-断桥残雪', '龙井路1号杭州西湖风景名胜区内(东北角)', 0.00, '国家级景点', 4.80, '杭州市', '120.151299', '30.258106', '00:00:00', '24:00:00', '30', '0571');

-- ----------------------------
-- Records of poi_restaurant
-- ----------------------------
INSERT INTO `poi_restaurant` VALUES ('14cc238b3c934fb2', '麦田咖啡·见山小院', '塘栖镇超丁村小龙线西南侧', '餐饮相关', '杭州市', 129.00, 4.60, '10:00-19:00', '无', 30, 90, '120.203069', '30.448232', '0571');
INSERT INTO `poi_restaurant` VALUES ('169e50321b9b42d7', '西湖·塔宴', '南山路15号', '餐饮相关', '杭州市', 158.00, 4.70, '11:00-21:00', '西湖醋鱼,蟹黄豆腐,东坡肉,龙井虾仁,白切鸡', 30, 120, '120.147840', '30.230281', '0571');
INSERT INTO `poi_restaurant` VALUES ('19f964c89bf146a6', '中国兰州拉面(半山路店)', '半山路153号', '清真菜馆', '杭州市', 955.00, 3.90, '09:00-22:00', '羊肉泡馍', 0, 120, '120.178600', '30.350170', '0571');

-- ----------------------------
-- Records of train_info
-- ----------------------------
INSERT INTO `train_info` VALUES ('03b4588ee2694c67', '2e8d724048d24f78', 'G1378', '591', '825.5', '贵阳北站', '贵阳市', '0851', '苏州北站', '苏州市', '0512');
INSERT INTO `train_info` VALUES ('03b4588ee2694c67', '52712064787a4566', 'G1332', '468', '702.5', '贵阳北站', '贵阳市', '0851', '杭州东站', '杭州市', '0571');
INSERT INTO `train_info` VALUES ('03b4588ee2694c67', '615cd5f1022b41f4', 'G1324', '841', '978.5', '贵阳北站', '贵阳市', '0851', '青岛西站', '青岛市', '0532');

-- ----------------------------
-- Records of path_planning_in_city
-- ----------------------------
INSERT INTO `path_planning_in_city` VALUES ('02c1a8fc77dc41e5', '060cb7c1ecb54fd9', '3916', '16', '13.0', '72', '6', '53', '住宿', '景点', '杭州市', '0571');
INSERT INTO `path_planning_in_city` VALUES ('02c1a8fc77dc41e5', '11a543aed7f14b98', '1391', '24', '13.0', '38', '5', '113', '住宿', '景点', '杭州市', '0571');
INSERT INTO `path_planning_in_city` VALUES ('02c1a8fc77dc41e5', '3922a00eb0674ead', '4732', '23', '13.0', '35', '3', '70', '住宿', '景点', '杭州市', '0571');
