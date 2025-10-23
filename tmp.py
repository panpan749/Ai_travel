# 导入OpenAI库，用于调用OpenAI API
from openai import OpenAI

# 配置API密钥和基础URL
# 创建OpenAI客户端实例
client = OpenAI(
    api_key="sk-60JytTArnn5XhePWShYgcRzEpcBQLyJoi60KYptcimqzj11W",  # 替换为你的实际令牌
    base_url="https://www.dmxapi.com/v1"  # 设置中转地址，添加 /v1 端点
)

# 调用对话接口示例
# 使用chat.completions.create方法创建对话请求
response = client.chat.completions.create(
    model="gpt-3.5-turbo",  # 指定使用的模型
    messages=[{"role": "user", "content": "你好"}]  # 发送消息，role为用户角色，content为消息内容
)

# 打印AI的回复内容
print(response.choices[0].message.content)