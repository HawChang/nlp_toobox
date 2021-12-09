# NLP TOOLBOX

通用的NLP工具，完成各类NLP任务。

目标是沉淀基础模型网络，具体使用时通过json配置文件来搭建整个训练、预测流程。

### 已支持任务类型
- [x] 分类任务
- [x] 相似匹配任务
- [x] 生成任务

### 任务示例-生成诗词

#### 执行语句

```
cd tasks/text_generation
sh run.sh -g 4 -c examples/poem_bert_generation.json
```

#### 效果示例






### TODO
- [ ] 优化模型配置逻辑
- [ ] 优化库文件结构
