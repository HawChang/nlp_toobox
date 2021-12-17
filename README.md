# NLP TOOLBOX

通用的NLP工具，完成各类NLP任务。

目标是沉淀基础模型网络，具体使用时通过json配置文件来搭建整个训练、预测流程。

### 已支持任务类型
- [x] 分类任务
- [x] 相似匹配任务
- [x] 生成任务

### 任务示例-生成诗词

已训练模型（放到`tasks/text_generation/output`下即可）
链接: https://pan.baidu.com/s/1f6jBhzLwZfJSTwOCM1K4_w 提取码: 69f0

#### 执行语句

```
cd tasks/text_generation
sh run.sh -g 4 -c examples/poem_bert_generation.json
```

#### 效果示例

示例1：

![示例1](/imgs/poem1.png)

示例2：

![示例2](/imgs/poem2.png)

示例3：

![示例3](/imgs/poem3.png)

示例4：

![示例4](/imgs/poem4.png)

示例5：

![示例5](/imgs/poem5.png)

### TODO
- [ ] 优化模型配置逻辑
- [ ] 优化库文件结构
