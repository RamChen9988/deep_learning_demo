import pandas as pd
import random
import numpy as np

# ---------------------- 1. 定义8类情绪的文本模板与关键词 ----------------------
# 每个情绪对应5个核心场景模板，搭配替换关键词库
emotion_config = {
    "平静": {
        "templates": [
            "今天{time}在家{action}，感觉一切都很{state}。",
            "路上遇到{person}，简单聊了几句，氛围很{state}。",
            "周末{action}了一会儿，没有特别的事情，心情很{state}。",
            "看完{thing}后，坐在沙发上发呆，内心很{state}。",
            "整理完{thing}，房间变整洁了，感觉很{state}。"
        ],
        "keywords": {
            "time": ["早上", "下午", "晚上", "午后", "傍晚"],
            "person": ["邻居", "同事", "同学", "朋友", "家人"],
            "action": ["看书", "喝茶", "听音乐", "练字", "养花"],
            "state": ["平和", "安稳", "平静", "淡然", "舒缓"],
            "thing": ["一本书", "一部纪录片", "整理衣柜", "书桌", "阳台"]
        }
    },
    "开心": {
        "templates": [
            "收到{person}送的{gift}，真的太{feeling}了！",
            "{event}成功了，终于实现了目标，特别{feeling}！",
            "吃到了想吃很久的{food}，味道超棒，心情很{feeling}。",
            "和{person}去{place}玩，拍了很多照片，特别{feeling}。",
            "今天{good_news}，瞬间觉得一切都很{feeling}。"
        ],
        "keywords": {
            "person": ["朋友", "家人", "对象", "闺蜜", "兄弟"],
            "gift": ["生日礼物", "手写信", "喜欢的周边", "零食大礼包", "鲜花"],
            "event": ["考试", "项目", "比赛", "面试", "减肥计划"],
            "feeling": ["开心", "高兴", "快乐", "兴奋", "愉悦"],
            "food": ["火锅", "烤肉", "奶茶", "蛋糕", "日料"],
            "place": ["公园", "游乐园", "海边", "古镇", "演唱会现场"],
            "good_news": ["被夸奖了", "涨工资了", "中奖了", "收到录取通知", "买到限量款"]
        }
    },
    "关心": {
        "templates": [
            "听说{person}最近{condition}，记得多{care_action}。",
            "天气变冷了，{person}要注意{care_thing}，别感冒了。",
            "{person}今天{tired_reason}，早点休息，别太累了。",
            "给{person}准备了{care_thing}，记得按时用。",
            "知道{person}在{busy_with}，有需要帮忙的随时说。"
        ],
        "keywords": {
            "person": ["妈妈", "爸爸", "朋友", "同事", "室友"],
            "condition": ["感冒了", "加班多", "压力大", "不舒服", "没休息好"],
            "care_action": ["喝热水", "穿厚点", "吃点热的", "放松一下", "补充睡眠"],
            "care_thing": ["暖宝宝", "感冒药", "热奶茶", "围巾", "润喉糖"],
            "tired_reason": ["加班到很晚", "跑了一天业务", "照顾生病的家人", "赶项目", "考试复习"],
            "busy_with": ["准备考试", "处理工作", "照顾孩子", "装修房子", "筹备婚礼"]
        }
    },
    "生气": {
        "templates": [
            "{person}居然{bad_behavior}，真的太让人{angry_feeling}了！",
            "等了{time}，{thing}还是没到，越想越{angry_feeling}。",
            "明明是{person}的错，还不承认，真的很{angry_feeling}！",
            "好好的{plan}被{person}搞砸了，特别{angry_feeling}。",
            "看到{person}在{bad_action}，忍不住想发火，太{angry_feeling}了。"
        ],
        "keywords": {
            "person": ["同事", "商家", "外卖员", "司机", "陌生人"],
            "bad_behavior": ["迟到半小时", "弄丢我的东西", "敷衍工作", "乱插队", "说脏话"],
            "angry_feeling": ["生气", "气愤", "恼火", "愤怒", "烦躁"],
            "time": ["半小时", "一小时", "两小时", "一下午", "一整晚"],
            "thing": ["外卖", "快递", "预约的服务", "买的东西", "维修师傅"],
            "plan": ["旅行计划", "聚会安排", "工作方案", "学习计划", "约会"],
            "bad_action": ["破坏公共设施", "欺负别人", "乱扔垃圾", "大声喧哗", "占小便宜"]
        }
    },
    "惊讶": {
        "templates": [
            "没想到{person}居然{surprise_thing}，太让人{surprise_feeling}了！",
            "打开{thing}，发现里面有{surprise_content}，真的很{surprise_feeling}！",
            "今天在{place}遇到了{person}，太{surprise_feeling}了，好久没见！",
            "{news}的消息出来，大家都很{surprise_feeling}，完全没预料到。",
            "试了一下{thing}，居然{surprise_result}，太{surprise_feeling}了！"
        ],
        "keywords": {
            "person": ["小学同学", "以前的老师", "很久没见的朋友", "偶像", "邻居"],
            "surprise_thing": ["考上了名校", "创业成功了", "搬到了同一个小区", "结婚了", "获奖了"],
            "surprise_feeling": ["惊讶", "吃惊", "意外", "震惊", "没想到"],
            "thing": ["快递盒", "礼物袋", "旧相册", "抽屉", "手机"],
            "surprise_content": ["小时候的玩具", "一张老照片", "意外的红包", "手写的卡片", "喜欢的礼物"],
            "place": ["超市", "地铁站", "演唱会", "公园", "咖啡馆"],
            "news": ["公司要加薪", "学校放假通知", "比赛结果", "政策变化", "朋友怀孕"],
            "surprise_result": ["很好用", "成功了", "味道超赞", "效果明显", "比想象中好"]
        }
    },
    "伤心": {
        "templates": [
            "{person}离开了{place}，以后很难见面了，心里很{sad_feeling}。",
            "养了很久的{pet}走了，忍不住{sad_action}，特别{sad_feeling}。",
            "努力了很久的{thing}还是失败了，感觉很{sad_feeling}。",
            "看到{person}因为{reason}难过，自己也跟着{sad_feeling}。",
            "不小心弄丢了{important_thing}，找了很久没找到，很{sad_feeling}。"
        ],
        "keywords": {
            "person": ["朋友", "家人", "同事", "室友", "老师"],
            "place": ["这座城市", "公司", "学校", "小区", "家乡"],
            "sad_feeling": ["伤心", "难过", "难受", "失落", "沮丧"],
            "pet": ["猫咪", "狗狗", "仓鼠", "兔子", "鹦鹉"],
            "sad_action": ["哭了", "睡不着", "不想说话", "情绪低落", "发呆"],
            "thing": ["考试", "项目", "比赛", "面试", "减肥计划"],
            "reason": ["失恋了", "家人生病", "工作丢了", "考试失利", "宠物离开"],
            "important_thing": ["妈妈送的项链", "童年的相册", "重要的文件", "朋友的礼物", "常用的钢笔"]
        }
    },
    "厌恶": {
        "templates": [
            "{person}总是{annoying_behavior}，真的很让人{disgust_feeling}。",
            "闻到{smell}的味道，忍不住想躲开，太{disgust_feeling}了。",
            "看到{person}在{bad_habit}，觉得很{disgust_feeling}，无法理解。",
            "吃了一口{food}，味道很{bad_taste}，特别{disgust_feeling}。",
            "{thing}上有{dirty_thing}，瞬间没了兴趣，很{disgust_feeling}。"
        ],
        "keywords": {
            "person": ["同事", "邻居", "陌生人", "室友", "同学"],
            "annoying_behavior": ["随地吐痰", "大声嚼东西", "说八卦", "抖腿", "抠鼻子"],
            "disgust_feeling": ["厌恶", "反感", "恶心", "讨厌", "不舒服"],
            "smell": ["垃圾", "变质的食物", "油烟", "香水味", "汗味"],
            "bad_habit": ["浪费食物", "破坏环境", "说脏话", "占小便宜", "背后议论别人"],
            "food": ["变质的牛奶", "生虫子的面包", "过咸的菜", "有怪味的零食", "没熟的肉"],
            "bad_taste": ["酸败", "苦涩", "油腻", "刺鼻", "发臭"],
            "thing": ["外卖盒", "衣服", "桌子", "鞋子", "餐具"],
            "dirty_thing": ["污渍", "虫子", "灰尘", "毛发", "脏水"]
        }
    },
    "疑问": {
        "templates": [
            "不知道{thing}在哪里，有人知道{question}吗？",
            "{person}为什么要{action}呢？真的很{confuse_feeling}。",
            "这个{thing}该怎么{use_method}啊？有点{confuse_feeling}。",
            "听说{news}，是真的吗？有点{confuse_feeling}，想确认一下。",
            "{place}的{thing}怎么突然{change}了？有点{confuse_feeling}。"
        ],
        "keywords": {
            "thing": ["我的钥匙", "手机充电器", "会议资料", "快递", "雨伞"],
            "question": ["它可能在什么地方", "怎么找比较快", "谁见过", "有没有人捡到", "该联系谁"],
            "person": ["他", "她", "老师", "商家", "主办方"],
            "action": ["这么做", "取消活动", "改变计划", "说这样的话", "拒绝帮忙"],
            "confuse_feeling": ["疑问", "困惑", "不解", "不确定", "想知道"],
            "use_method": ["操作", "设置", "使用", "安装", "调试"],
            "news": ["明天要放假", "公司要搬家", "这个产品要下架", "价格要上涨", "活动取消"],
            "place": ["超市", "公司", "学校", "小区", "地铁站"],
            "change": ["涨价了", "关门了", "换负责人了", "改规则了", "断货了"]
        }
    }
}

# ---------------------- 2. 批量生成10万条数据 ----------------------
def generate_emotion_text(emotion, template, keywords):
    """根据模板和关键词生成单条情感文本"""
    filled_text = template
    for key, options in keywords.items():
        if key in filled_text:
            filled_text = filled_text.replace(f"{{{key}}}", random.choice(options))
    return filled_text

# 初始化总数据集列表
total_data = []
# 每个情绪生成12500条数据（8类×12500=10万）
per_emotion_count = 12500

for emotion, config in emotion_config.items():
    templates = config["templates"]
    keywords = config["keywords"]
    # 循环生成当前情绪的所有数据
    for _ in range(per_emotion_count):
        # 随机选一个模板
        template = random.choice(templates)
        # 生成文本
        text = generate_emotion_text(emotion, template, keywords)
        # 添加到总数据
        total_data.append({"text": text, "label": emotion})

# ---------------------- 3. 转换为DataFrame并导出CSV ----------------------
# 转换为DataFrame
df = pd.DataFrame(total_data)
# 打乱数据顺序（避免同类情绪集中）
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
# 导出为CSV（编码用utf-8-sig，避免中文乱码）
df.to_csv("data/chinese_emotion_analysis_100k.csv", index=False, encoding="utf-8-sig")

# 验证数据分布（可选）
print("数据量验证：", len(df))  # 应输出100000
print("\n情绪分布验证：")
print(df["label"].value_counts())  # 每类应输出12500