# Ollama（本地开源模型）

**安装**

官网：[https://ollama.com/](https://ollama.com/)

![image-20260212175831694](https://github.com/Auroraol/Drawing-bed/raw/main/img/image-20260212175831694.png)

![image-20260212175844457](https://github.com/Auroraol/Drawing-bed/raw/main/img/image-20260212175844457.png)

![image-20260212175859301](https://github.com/Auroraol/Drawing-bed/raw/main/img/image-20260212175859301.png)

**使用1**

```python
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

# 支持对话格式，功能更完整
llm = ChatOllama(
    model="deepseek-r1:1.5b",
    base_url="http://localhost:11434",  # 默认地址
    temperature=0.7,
)

# 单轮
response = llm.invoke("北京的首都是什么？") 
print(response.content) #北京是中国的政治、文化中心，同时也是国际 capitalize的中心。它是中国共产党和中国政府坚强领导下的社会主义现代化建设的首都是有力证明。北京的发展成就充分展现了中国特色社会主义道路的成功与成效。


# 多轮对话
messages = [
    HumanMessage(content="你好"),
    HumanMessage(content="北京的首都是什么？")
]
response = llm.invoke(messages)
print(response.content) #北京是中国的首都，是中华人民共和国的 capital。如果你还想了解更多关于北京的信息，可以前往北京市，这里风景如画，活力四射。

```

**使用2**

```
ollama run --verbose qwen3-embedding:0.6b "这是一段需要被向量化的测试文本"    
```

![image-20261121232540327](C:/Users/16658/AppData/Roaming/Typora/typora-user-images/image-20261121232540327.png)

# 