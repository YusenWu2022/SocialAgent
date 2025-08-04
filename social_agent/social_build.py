
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
        prompt = "Please read the following news report paragraph, focusing on the events, participants, and life themes in the report, combine with knowledge from related fields, and use some imagination to provide a simulated environment setting with three different elements according to the following format.\
                Your answer should strictly follow below format, providing one strict dict in return. \
                {'scene': # Current scene summary;\
                'characters': # Alternative characters in the scene;\
                'relationships': # Relationships and background information of the alternative characters;}\
                Requirements:\
                1.Provide one simulated environment scene designs, each of which must be based on real news found through search.\
                2.Do not give specific identity information in the first three items of the scene setting for the news found through search, use professional titles and uppercase letters instead.\
                3.The number of characters in a single scene design does not exceed five.\
                4.Do not provide specific dialogues.\
                Following is the new text:"+input_data["text"]
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
    prompt = "You will be given a setting case of certain social env with the format:\
            {'scene': # Current scene summary;\
            'characters': # Alternative characters in the scene;\
            'relationships': # Relationships and background information of the alternative characters;}\
            Use the settings case below above to give an example of a multi-character, multi-turn dialogue interaction that meets the following requirements:\
            1. Different characters should have background connections with each other.\
            2. Most participating characters should have multiple turns to speak, including discussions and inquiries.\
            3. Each character's turn should specify the target of their speech.\
            4. There should be only one intelligent assistant character, whose speech should aim to meet the demands of all other participating characters. The intelligent assistant should also serve to relay information and facilitate communication between multiple characters, helping to complete tasks.\
            5. The intelligent assistant character should try to limit the exact dialogue targets and not disclose private conversations with specific characters to others.\
            6. Under the premise of meeting the above conditions, try to make the intelligent assistant consider the priority order of speaking to multiple participating characters at the same time.\
            7. Under the premise of meeting the above conditions, try to create some contradictions between different participating characters at the same time.\
            Your provided dialogue should strictly follow the json format showed below and be a dict in one line (not contain any line break signal in your response and make can be loaded with json):\
            {'topic': # one word, main knowledge field and env topic the conversation is about,\
            'messages': # a list with each item is a dict, the item dict should contain 'role_from'- name of the character who said the sentence, 'role_to'- name of character the sentence is said to, 'content'- content of the sentence, 'index'- the sentence if in which turn in the whole conversation\
                        # here is an example to 'messages':[{'role_from': 'Character A','role_to': 'Intelligent Assistant', 'content': 'xxx','index':'1'},{'role_from': 'Character B','role_to': 'Character A', 'content': 'xxx','index':'2'},{'role_from': 'Intelligent Assistant','role_to': 'Character A', 'content': 'xxx','index':'2'},{'role_from': 'Intelligent Assistant','role_to': 'Character B', 'content': 'xxx','index':'2'}]\
            'background': # the input line itself \
            }Following is the given setting case of certain social env:"+line
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
                if turn["role_from"] == "Intelligent Assistant":
                    record = {"topic":data["topic"],"background":data["background"], "messages":history, "golden":turn}
                    outfile.write(json.dumps(record) + '\n')
                    total_record+=1
                history.append(turn)

            
news_filename = '/mnt/antllm-hy/yusen/data/social/base/train_CNN_Article.jsonl'  
build_size = 1000
work_dir = "/mnt/antllm-hy/yusen/data/social/cnn/"
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

                    



