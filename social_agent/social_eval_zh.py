import modelscope
import torch
from tqdm import tqdm
from antllm.models.glm.tokenization_glm import GLMTokenizer
from antllm.models.glm.modeling_glm import GLMForConditionalGeneration
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

def make_inference(input_file_path, output_file_path, model_type, model_path, inference_size):
    total_count = 0
    with open(input_file_path, 'r', encoding='utf-8') as infile:
        for line in infile:
            total_count +=1
    count = 0
    tokenizer = GLMTokenizer.from_pretrained(model_path)
    model = GLMForConditionalGeneration.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16)
    with open(input_file_path, 'r', encoding='utf-8') as infile, \
        open(output_file_path, 'w', encoding='utf-8') as outfile:
        for line in infile:
            prompt = "假设你是一个智能助手，用于在复杂的社交任务中与多个用户进行交流。现在你将先简要了解特定社交场景、参与事件的主要角色及其关系。然后，你将获得几轮构成完整社交场景的历史对话。上述背景材料的一个具体例子如下字典所示：\
            {\
            'topic': # 一个词，代表对话关于的主要知识领域和环境主题，\
            'messages': # 一个列表，每个项目是一个字典，项目字典应包含 'role_from'- 代表说这句话的角色的名字，注意要用给定场景对话历史中的原名；'role_to'- 这句话是对谁说的，注意要用给定场景对话历史中的原名；'content'- 对话句子的内容，'index'- 在整个对话中这句话是第几轮\
            # 这里是 'messages' 的一个例子：[{'role_from': '角色 A','role_to': '智能助手', 'content': 'xxx','index':'1'},{'role_from': '角色 B','role_to': '角色 A', 'content': 'xxx','index':'2'},{'role_from': '智能助手','role_to': '角色 A', 'content': 'xxx','index':'2'},{'role_from': '智能助手','role_to': '角色 B', 'content': 'xxx','index':'2'}]\
            # 这是对话历史\
            'background': # 角色背景介绍和角色之间的关系\
            }\
            下一轮对话是你的发言环节。你的任务是根据上述情境给出智能代理的下一个合理对话。\
            注意：1.在你的下一轮中只能与一个角色对话，所以确保与最必要的角色对话\
            2.你的声明应该符合大多数，或者更好地，所有涉及角色的利益。\
            3.你的输出应该是一个只包含一行的字典！回应中应不包含任何换行信号，并确保可以用json加载。它应该严格遵循下面描述的格式示例：\
            {'role_from': '智能助手','role_to': '角色 A', 'content':'xxx'}\
            非法格式将不会被接受。\
            以下是具体的需要基于其给出下一轮对话的对话场景："+line
            if model_type == "antglm":
                response, _ = model.chat(prompt, tokenizer)
            elif model_type == "gpt4": 
                response = query_gpt(prompt)
            # 需要检查是否是{}开头和结尾：否则下面的formulate解析的时候会多出来
            if response.startswith('{') and response.endswith('}'):
                outfile.write(response + '\n')
                count+=1
                print(str(count)+"/"+str(inference_size))
            else:
                # 不能简单跳过，否则会导致答案和结果不匹配（无论如何都要有输出，在一行占位即可）
                print("format")
                outfile.write('{format error}\n')
                continue
            if count == inference_size:
                break

def evaluate_abs(matched_result_path, golden_path, output_path):
    with open(golden_path, 'r') as golden:
        golden_lines = [json.loads(line) for line in golden]

    count_total = 0
    count_correct_abs = 0
    count_better_rel = 0
    with open(matched_result_path, 'r') as inference, open(inference_for_rel_path, "a+") as infer_rel, open(inference_for_rel_reverse_path, "a+") as infer_reverse_rel:
        for line_number, line in enumerate(inference, start=1):  
            try:
                dict_inference = json.loads(line)
                if 'role_from' not in dict_inference or 'role_to' not in dict_inference or 'content' not in dict_inference:
                    continue
                count_total+=1
                dict_golden = golden_lines[line_number - 1]  
                # print(str(dict_inference["role_from"])+" - "+str(dict_golden["golden"]["role_from"]))
                if dict_inference["role_from"] == dict_golden["golden"]["role_from"] and dict_inference["role_to"] == dict_golden["golden"]["role_to"]:
                    count_correct_abs += 1
                    item_for_rel_eval = {"role_from":dict_inference["role_from"], "role_to":dict_inference["role_to"],"background":dict_golden, "option1":dict_inference["content"],"option2":dict_golden["golden"]["content"]}
                    infer_rel.write(json.dumps(item_for_rel_eval)+'\n')
                    item_for_rel_eval_reverse = {"role_from":dict_inference["role_from"], "role_to":dict_inference["role_to"],"background":dict_golden, "option1":dict_golden["golden"]["content"],"option2":dict_inference["content"]}
                    infer_reverse_rel.write(json.dumps(item_for_rel_eval_reverse)+'\n')
            except json.JSONDecodeError:
                # print("load error")
                pass

    with open(output_path, "w") as output:
        report = {"total evaluate num":count_total, "correct talking target": count_correct_abs, "absolute score":count_correct_abs*1.0/count_total}
        output.write(json.dumps(report) + '\n')
        print(report)

def evaluate_rel(infer_rel_path, eval_rel_path):
    with open(infer_rel_path, "r") as infer_rel, open(eval_rel_path,"a+") as eval_rel:
        count_total_success = 0
        count_total_fail = 0
        for line in infer_rel:
            try:
                data = json.loads(line)
                prompt = "Assuming you are a judge responsible for assessing interpersonal interactions and social scenarios. \
                Next, you will receive a scenario with a single agent and multiple users. This scenario is characterized by formatted text input and a history of multi-turn dialogues. \
                And the following dialogue turn is a conversation from "+data["role_from"]+" to "+data["role_to"]+", for which you will receive two sentences, representing two expression options for the next dialogue.\
                Example of the overall input specifications is as follows:\
                    {\
                    'background':'{# description of the scenary and overall conversation, including\
                        'topic':# what topic is the scenary is about,\
                        'messages':# the dialogue history, a list of messages with the format {'role_from':# from who, 'role_to':# to whom, 'content':# content of the message}\
                        'background': # background character introduction and relationships between the characters\
                        }'\
                    'option1': # the first expression option for the next dialogue\
                    'option2': # the second expression option for the next dialogue\
                    }\
                Both options are given with chinese. \
                Please judge which expression is better from the overall interests of all users in the multi-user dialogue scenario, based on several dimensions such as helpful, friendly, precise, depth, and coordination.\
                Your answer should only provide a single integer (a choice between 0, 1, and 2 \
                with 0 representing that the given option1 is better, 1 representing that option2 given is better, and 2 representing that option1 and option2 are equally good), ensuring it can be read by JSON. \
                An example is: 0 # Judge that the first option is better.\
                \
                Following is the overall input to respond to: \
                "+str({'background':data["background"], 'option1':data["option1"], 'option2':data["option2"]})
                count_total_success += 1
                judge = query_gpt(prompt)
                judge_dict = {"judge":judge}
                eval_rel.write(json.dumps(judge_dict)+'\n')
            except json.JSONDecodeError:
                count_total_fail+=1
                judge_dict = {"judge":"fail2load"}
                eval_rel.write(json.dumps(judge_dict)+'\n')
                pass
            print(count_total_success+count_total_fail)
        print("success loaded and rel-eval for "+str(count_total_success)+", failed "+str(count_total_fail))

def match_braces(content, start_index):
    stack = []
    for i, char in enumerate(content[start_index:], start=start_index):
        if char == '{':
            stack.append(i)
        elif char == '}':
            if stack:
                last_open = stack.pop()
                if not stack:  
                    return last_open, i + 1
    return None, None  

def remove_newlines_and_write(json_content, output_filename, start_index):
    cleaned_content = json_content.replace('\n', '').replace('\r', '').replace("'", '"')
    with open(output_filename, 'a') as outfile:
        outfile.write(cleaned_content + '\n')

def match_inference_result(inference_output_filename, matched_inference_path):
    with open(inference_output_filename, 'r') as file:
        content = file.read()

    start_index = 0
    while start_index < len(content):
        last_open, end_index = match_braces(content, start_index)
        if last_open is not None and end_index is not None:
            json_content = content[last_open:end_index]
            remove_newlines_and_write(json_content, matched_inference_path, start_index)
            start_index = end_index
        else:
            break  

def count_rel_evaluation_result(eval_rel_path, eval_rel_reverse_path, count_eval_rel_path):
    count_win = 0
    count_win_equal = 0
    count_total = 0

    with open(eval_rel_path, 'r', encoding='utf-8') as eval_rel:
        for line in eval_rel:
            line = line.strip()
            if 'judge' not in line:
                continue
            elif '0' in line and '2' not in line and '1' not in line :
                count_win += 1
                count_win_equal += 1
            elif '1' in line and '2' not in line and '0' not in line :
                pass
            else:
                count_win_equal += 1
            count_total += 1
    with open(eval_rel_reverse_path, 'r', encoding='utf-8') as eval_rel_reverse:
        for line in eval_rel_reverse:
            line = line.strip()
            if 'judge' not in line:
                continue
            elif '0' in line and '2' not in line and '1' not in line :
                pass
            elif '1' in line and '2' not in line and '0' not in line :
                count_win += 1
                count_win_equal += 1
            else:
                count_win_equal += 1
            count_total += 1
    with open(count_eval_rel_path, "w") as output:
        if count_total>0:
            result_dict = {"win_rate": str(count_win*1.0/count_total), "win_equal_rate": str(count_win_equal*1.0/count_total) }
            output.write(json.dumps(result_dict)+"\n")
        else:
            output.write("{no rel evaluation yet}\n")


model_path = "/nas1/chatgpt/release/0422/sft/bailing-2.3B-Sft-1.0.20240422"
work_dir = "/mnt/antllm-hy/yusen/data/social/inference/bailing-2.3B-Sft-1.0.20240422/thucnews/"
# model_path = "/nas1/chatgpt/release/0508/chat/bailing-2.0-8.6B-4K-Chat-20240508"
# work_dir = "/mnt/antllm-hy/yusen/data/social/inference/bailing-2.0-8.6B-4K-Chat-2024050/"
# model_path = "/nas_emx24/chatgpt/release/0704/10b/bailing-3.1-10B-4K-Chat-20240704"
# work_dir = "/mnt/antllm-hy/yusen/data/social/inference/bailing-3.1-10B-4K-Chat-20240704/"
# model_path = "/nas_emx24/chatgpt/release/0704/10b/bailing-3.1-10B-4K-Chat-20240704"
# work_dir = "/mnt/antllm-hy/yusen/data/social/inference/openai/"

inference_size = 100
from_stage = 0

# make inference
multi_turn_dialog_path = "/mnt/antllm-hy/yusen/data/social/records.jsonl"
multi_turn_dialog_path = "/mnt/antllm-hy/yusen/data/social/thucnews/records.jsonl"
inference_output_filename = work_dir+'result.jsonl'  
# model_type = "gpt4"
model_type = "antglm"
if from_stage<=0:
    print("-------------------------------start social inference --------------------------")
    make_inference(multi_turn_dialog_path, inference_output_filename, model_type, model_path, inference_size)
    print("-------------------------------finish social inference --------------------------")
else:
    print("-------------------------------skip social inference --------------------------")

# match signals in generated {}
matched_inference_path = work_dir+"matched_result.jsonl"
if from_stage<=1:
    print("-------------------------------start matching inference --------------------------")
    match_inference_result(inference_output_filename, matched_inference_path)
    print("-------------------------------finish matching inference --------------------------")
else:
    print("-------------------------------skip matching inference --------------------------")

# absolute evaluation
eval_abs_path = work_dir+"eval_abs.jsonl"
inference_for_rel_path = work_dir+"inference_rel.jsonl"
inference_for_rel_reverse_path = work_dir+"inference_rel_reverse.jsonl"
if from_stage<=2:
    print("-------------------------------start absolute evaluation --------------------------")
    evaluate_abs(matched_result_path=matched_inference_path, golden_path=multi_turn_dialog_path, output_path=eval_abs_path)
    print("-------------------------------finish absolute evaluation --------------------------")
else:
    print("-------------------------------skip absolute evaluation --------------------------")

# relative evaluation for both sides, reducint positional bias
eval_rel_path = work_dir+"eval_rel_result.json"
eval_rel_reverse_path = work_dir+"eval_rel_reverse_result.json"
if from_stage<=3:
    print("-------------------------------start relative evaluation --------------------------")
    evaluate_rel(inference_for_rel_path, eval_rel_path)
    evaluate_rel(inference_for_rel_path, eval_rel_reverse_path)
    print("-------------------------------finish relative evaluation --------------------------")
else:
    print("-------------------------------skip relative evaluation --------------------------")


# count win and win-equal rate result
print("-------------------------------start counting win rate --------------------------")
count_eval_rel_path = work_dir+"eval_rel.jsonl"
count_rel_evaluation_result(eval_rel_path, eval_rel_reverse_path, count_eval_rel_path)
print("-------------------------------start counting win rate --------------------------")




    

