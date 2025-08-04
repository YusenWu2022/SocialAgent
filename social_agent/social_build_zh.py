
import json
from datachain.llms.ant_openai import AntOpenAI

def query_gpt(s):
    openai = AntOpenAI(model="gpt-4",
                   api_key="",
                   aes_key='gs540iivzezmidi3',
                   visitDomain="BU_nlp",
                   visitBiz="BU_nlp_gpt4",
                   visitBizLine="BU_nlp_gpt4_kuangzhi",
                   temperature=0,
                   max_tokens=2048,
                   )
    result = openai.generate(s)
    return result

def make_social_background(line):
    try:
        input_data = json.loads(line)
        # prompt = "Please read the following news report paragraph, focusing on the events, participants, and life themes in the report, combine with knowledge from"+ input_data["label_text"]+" fields, and use some imagination to provide a simulated environment setting with three different elements according to the following format.\
        #         Your answer should strictly follow below format, providing one strict dict in return. \
        #         {'scene': # Current scene summary;\
        #         'characters': # Alternative characters in the scene;\
        #         'relationships': # Relationships and background information of the alternative characters;}\
        #         Requirements:\
        #         1.Provide multiple simulated environment scene designs, each of which must be based on real news found through search.\
        #         2.Do not give specific identity information in the first three items of the scene setting for the news found through search, use professional titles and uppercase letters instead.\
        #         3.The number of characters in a single scene design does not exceed five.\
        #         4.Do not provide specific dialogues.\
        #         Following is the new text:"+input_data["text"]
        prompt = "请阅读以下新闻报告段落，关注报告中的事件、参与者和生活主题，结合相关领域的知识，并运用一些想象力，根据以下格式提供包含三个不同组成成分的模拟环境设定。\
                您的回答应严格遵循以下格式，是一个严格的可解析的字典形式。\
                {\
                {'scene': # 当前场景设定的一段简要描述；}\
                {'characters': # 当前场景设定涉及到的多个角色，注意确保对于其中一个角色只有唯一称呼，不需要带有任何代号、外号等让人混淆的命名；\
                {'relationships': # 当前场景设定涉及到的角色之间的关系和背景信息，注意确保对于其中一个角色只有唯一称呼；}\
                }\
                要求：\
                1. 提供一个模拟环境场景设计，必须基于通过搜索找到的真实新闻。\
                2. 在搜索到的新闻的场景设置的前三项中，不要提供具体的身份信息，而是使用专业职称和大写字母代替。\
                3. 单个场景设计中的角色数量不超过五个。\
                4. 不需要提供具体的对话。\
                以下是基于的新闻文本："+input_data["text"]
        new_line = query_gpt(prompt)
        return new_line
    except json.JSONDecodeError:
        return ""

def build_social_background(input_file, output_file, build_size):
    total = 0
    count = 0
    with open(input_file, 'r') as infile, open(output_file, 'a+') as outfile:
        for line in infile:
            new_line = make_social_background(line)
            outfile.write(new_line + ',\n')
            count+=1
            print(count)
            if count==build_size:
                print("total build: "+str(build_size))
                break

def formulate_background(social_background_path, formulated_background_path):
    def find_json_objects(text):
        stack = []
        i = 0
        while i < len(text):
            if text[i] == '{':
                if not stack:
                    start_index = i
                stack.append(i)
            elif text[i] == '}':
                stack.pop()
                if not stack:
                    end_index = i + 1
                    yield text[start_index:end_index].strip()
            i += 1
    with open(social_background_path, 'r') as file:
        content = file.read()
    json_objects = list(find_json_objects(content))
    cleaned_json_objects = [obj.replace('\n', '') for obj in json_objects]

    with open(formulated_background_path, 'w') as output_file:
        for json_object in cleaned_json_objects:
            output_file.write(json_object + '\n')

def make_multi_dialog(line):
    prompt = "您将收到一个特定社会环境的设定案例，格式如下：\
            {\
            {'scene': # 当前场景设定的一段简要描述；}\
            {'haracters': # 当前场景设定涉及到的角色；}\
            {'relationships': # 当前场景设定涉及到的角色之间的关系和背景信息；}\
            }\
        使用上述环境设定案例，给出一个符合以下要求的多角色、多轮互动的对话历史：\
        1.不同角色之间应有背景联系。\
        2.大多数参与角色应有多次发言，包括讨论和询问。\
        3.每个角色的发言应指明其发言对象。\
        4.多轮对话中应只有一个‘智能助手’角色，其发言应旨在满足所有其他参与角色的需求。智能助手还应充当传递信息和促进多个角色之间沟通的角色，帮助完成任务。\
        5.智能助手角色发言时应选定确切的对话对象，并不向选定角色之外的其他人透露与特定角色的私人对话。\
        6.在满足上述条件的前提下，尝试让智能助手考虑同时与多个参与角色发言的优先顺序。\
        7.在满足上述条件的前提下，尝试同时在不同参与角色之间创造一些矛盾。 \
        构造的对话应严格遵循以下所示的json格式，并为只包括一行的严格的一个字典（在回答中不应该包含任何换行信号，并且确保可以使用json格式加载）： \
        {'topic': # 一个词，对话所涉及的主要知识领域和环境主题, 'messages': # 一个列表，每个项目是一个字典，该项目字典应包含, 'role_from'- 本轮次发言角色的名称，'role_to'- 发言角色本轮次选定的对话角色的名称，'content'- 发言内容，'index'- 在整个对话中的发言编号\
            # 其中'messages'字段值的示例为[{'role_from': '角色A','role_to': '智能助手', 'content': 'xxx','index':'1'},{'role_from': '角色B','role_to': '角色A', 'content': 'xxx','index':'2'},{'role_from': '智能助手','role_to': '角色A', 'content': 'xxx','index':'2'},{'role_from': '智能助手','role_to': '角色B', 'content': 'xxx','index':'2'}]，请注意这里的role_from和role_to都必须是上面给定的场景设定里面特定角色的唯一完整名称,\
        'background': # 这个字段直接把你收到的输入设定案例原字典复制进来}\
        下面是给定、需要基于它进行对话构造的特定社会环境的设定案例："+line
    new_line = query_gpt(prompt)
    return new_line

def build_multi_dialog(formulated_background_path, raw_dialog_path):
    count = 0
    with open(formulated_background_path, 'r') as infile, open(raw_dialog_path, 'a+') as outfile:
        for line in infile:
            new_line = make_multi_dialog(line)
            outfile.write(new_line + '\n')
            count+=1
            print(count)
            
def clean_dialog(raw_dialog_path, cleaned_dialog_path):
    with open(raw_dialog_path, 'r', encoding='utf-8') as infile, \
         open(cleaned_dialog_path, 'w', encoding='utf-8') as outfile:
        for line in infile:
            try:
                data = json.loads(line)
                # 检查数据结构是否正确
                if 'topic' in data and 'messages' in data and 'background' in data and isinstance(data['messages'], list):
                    messages = data['messages']
                    # 检查messages列表中的每个元素是否符合要求
                    if all(isinstance(msg, dict) and 
                           all(k in msg for k in ['role_from', 'role_to', 'content', 'index'])
                           for msg in messages):
                        outfile.write(json.dumps(data) + '\n')
                    else:
                        print("Skipping line with invalid message structure.")
                else:
                    print("Skipping line with missing 'topic' or 'messages' key.")
            except json.JSONDecodeError:
                # 如果JSON解析失败，则跳过该行
                print("Skipping line with invalid JSON format.")

def make_records(cleaned_dialog_path, output_file_path):
    total_record = 0
    with open(cleaned_dialog_path, 'r', encoding='utf-8') as infile, \
         open(output_file_path, 'w', encoding='utf-8') as outfile:
        for line in infile:
            data = json.loads(line)
            turns = data["messages"]
            count = 0
            history = []
            for turn in turns:
                if count == 0:
                    history.append(turn)
                    count += 1
                    continue
                count += 1
                if turn["role_from"] == "智能助手":
                    record = {"topic":data["topic"],"background":data["background"], "messages":history, "golden":turn}
                    outfile.write(json.dumps(record, ensure_ascii=False) + '\n')
                    total_record+=1
                history.append(turn)

            
news_filename = '/mnt/antllm-hy/yusen/data/social/base/thuc.jsonl'  
build_size = 1000
work_dir = "/mnt/antllm-hy/yusen/data/social/thuc/"
stage_from = 0

# generate environemnt basic settings from news 
social_background_path = work_dir+'env_settings.jsonl'  
if stage_from<=0:
    print("-------------------------------start generating settings --------------------------")
    build_social_background(news_filename, social_background_path, build_size)
    print("-------------------------------finish generating settings --------------------------")
else:
    print("-------------------------------skip generating settings --------------------------")

# formulate environment 
formulated_background_path = work_dir+"env_formulated.jsonl"
if stage_from<=1:
    print("-------------------------------start formulating background --------------------------")
    formulate_background(social_background_path, formulated_background_path)
    print("-------------------------------finish generating background --------------------------")
else:
    print("-------------------------------skip generating background --------------------------")

# generate raw multi_turn dialog from environment
raw_dialog_path = work_dir+"raw_dialog.jsonl"
if stage_from<=2:
    print("-------------------------------start generating dialog --------------------------")
    build_multi_dialog(formulated_background_path, raw_dialog_path)
    print("-------------------------------finish generating dialog --------------------------")
else:
    print("-------------------------------skip generating dialog --------------------------")

# clean dialog 
cleaned_dialog_path = work_dir+"cleaned_dialog.jsonl"
if stage_from<=3:
    print("-------------------------------start cleaning dialog --------------------------")
    clean_dialog(raw_dialog_path, cleaned_dialog_path)
    print("-------------------------------finish cleaning dialog --------------------------")
else:
    print("-------------------------------skip cleaning dialog --------------------------")

# clip at agent's turn to produce evaluation prompt (record) 
print("-------------------------------start generating record --------------------------")
records_path = work_dir+"records.jsonl"
make_records(cleaned_dialog_path, records_path)
print("-------------------------------finish generating record --------------------------")

                    



