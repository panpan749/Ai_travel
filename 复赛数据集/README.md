# 复赛说明

本文件说明复赛阶段的文件结构、新增预定义规则以及提交要求。请选手严格遵守以下格式与命名规范。

---

## 文件结构
---

- **prompts/**
包含 10 个标准示例。

- **dataset/question.json**
数据集文件，包含所有自然语言问题。

## 新增预定义规则
---
1. 对于多人旅行问题，若问题描述中明确提及有一名孩童，则该孩童不计任何开销，不计入房间入住人数和打车人数。
2. 对于多阶段旅行问题，中转当日游览的景点、饭店、市内通勤等活动在前一个城市进行，在下一个城市进行当晚的住宿，城际交通通勤时间不计入当日活动时间。

## 规划代码输出格式：
---
单阶段问题,代码输出结果应包含以下字段：
```json
{
    "budget": ,
    "peoples": ,
    "travel_days": ,
    "origin_city": ,
    "destination_city": ,
    "start_date": ,
    "end_date": ,
    "daily_plans": ,
    "departure_trains": ,
    "back_trains": ,
    "total_cost": ,
    "objective_value": 
}
```
多阶段问题,代码输出结果应包含以下字段：（新增"intermediate_city"， "intermediate_city"，"intermediate_trains"字段）
```json
{
    "budget": ,
    "peoples": ,
    "travel_days": ,
    "origin_city": ,
    "intermediate_city": ,
    "destination_city": ,
    "start_date": ,
    "intermediate_date": ,
    "end_date": ,
    "departure_trains": ,
    "intermediate_trains": ,
    "back_trains": ,         
    "daily_plans": ,
    "total_cost": ,
    "objective_value": 
}
```

# 复赛提交文件：

选手需提交一个 `commit.zip` 压缩包,zip压缩包内文件结构如下，格式同初赛一致，需要包含生成的规划代码与问题描述文件

```Bash
.
├── code
│   ├── id_1.py
│   ├── ...
│   └── id_800.py
└── predict.json
```
