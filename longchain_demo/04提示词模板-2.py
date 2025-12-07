from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate

from longchain_demo.my_llm import llm

# ICL ： In Context Learning
# 问题： 巴伦·特朗普的父亲是谁？

# 步骤一： 提供示例
examples = [
    {
        "question": "穆罕默德·阿里和艾伦·图灵谁活得更久？",
        "answer": """
是否需要后续问题：是。
后续问题：穆罕默德·阿里去世时多大？
中间答案：穆罕默德·阿里去世时74岁。
后续问题：艾伦·图灵去世时多大？
中间答案：艾伦·图灵去世时41岁。
所以最终答案是：穆罕默德·阿里
""",
    },
    {
        "question": "乔治·华盛顿的外祖父是谁？",
        "answer": """
是否需要后续问题：是。
后续问题：乔治·华盛顿的母亲是谁？
中间答案：乔治·华盛顿的母亲是玛丽·鲍尔·华盛顿。
后续问题：玛丽·鲍尔·华盛顿的父亲是谁？
中间答案：玛丽·鲍尔·华盛顿的父亲是约瑟夫·鲍尔。
所以最终答案是：约瑟夫·鲍尔
""",
    },
    {
        "question": "《大白鲨》和《007：大战皇家赌场》的导演是否来自同一个国家？",
        "answer": """
是否需要后续问题：是。
后续问题：《大白鲨》的导演是谁？
中间答案：《大白鲨》的导演是史蒂文·斯皮尔伯格。
后续问题：史蒂文·斯皮尔伯格来自哪里？
中间答案：美国。
后续问题：《007：大战皇家赌场》的导演是谁？
中间答案：《007：大战皇家赌场》的导演是马丁·坎贝尔。
后续问题：马丁·坎贝尔来自哪里？
中间答案：新西兰。
所以最终答案是：否
""",
    },
]


base_template = PromptTemplate.from_template("问题: {question}\n{answer}")

# 步骤二：创建FewShotPromptTemplate实例
final_template = FewShotPromptTemplate(
    examples=examples,  # 传入示例列表
    example_prompt=base_template,  # 指定单个示例的提示模板
    suffix="问题: {input}",  # 最后追加的问题模板
    input_variables=["input"],  # 指定输入变量
)


#
chain = final_template | llm
# resp = chain.invoke({"input": "巴伦·特朗普的父亲是谁？"})
resp = chain.invoke({"input": "擎天柱和威震天谁厉害"})
print(resp)