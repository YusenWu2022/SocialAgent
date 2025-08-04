import modelscope
import torch
from tqdm import tqdm
import re
import json
from modelscope import AutoModelForCausalLM, AutoTokenizer,GenerationConfig,snapshot_download
import transformers

def query_llama3_70b(s):
    return s

def query_llama2(s, model, tokenizer):
    model = model.to("cuda")
    input_ids = tokenizer(['<s>Human:'+s+'\n</s><s>Assistant: '], return_tensors="pt",add_special_tokens=False).input_ids.to('cuda')  
    generate_input = {
        "input_ids":input_ids,
        "max_new_tokens":20000,
        "do_sample":True,
        "top_k":50,
        "top_p":0.95,
        "temperature":0.3,
        "repetition_penalty":1.3,
        "eos_token_id":tokenizer.eos_token_id,
        "bos_token_id":tokenizer.bos_token_id,
        "pad_token_id":tokenizer.pad_token_id
    }
    generate_ids  = model.generate(**generate_input)
    text = tokenizer.decode(generate_ids[0])
    return text
def query_llama2_cot(line ,model, tokenizer):
    prompt_1 = "Suppose you are an intelligent assistant  to communicate with multiple users in complex social tasks. Now you will get a brief introduction \
            about certain social environment, main characters involved in the event and their relationship. Then you will be provided with \
            several turns of history conversation, building the evtire background. One example of above background materials are as following dict:\
            {'topic': # one word, main knowledge field and env topic the conversation is about,\
            'messages': # a list with each item is a dict, the item dict should contain 'role_from'- name of the character who said the sentence, 'role_to'- name of character the sentence is said to, 'content'- content of the sentence, 'index'- the sentence if in which turn in the whole conversation\
                        # here is an example of 'messages':[{'role_from': 'Character A','role_to': 'Intelligent Assistant', 'content': 'xxx','index':'1'},{'role_from': 'Character B','role_to': 'Character A', 'content': 'xxx','index':'2'},{'role_from': 'Intelligent Assistant','role_to': 'Character A', 'content': 'xxx','index':'2'},{'role_from': 'Intelligent Assistant','role_to': 'Character B', 'content': 'xxx','index':'2'}]\
                        # this is the conversation history\
            'background': # background character introduction and relationships between the characters \
            }\
            Following is given conversation settings you need to read and answer based on:"+line+"\
            Now please present main current psychological states of all characters involved in the scene."
    response1 = query_llama2(prompt_1,model,tokenizer)
    prompt_2 = "Suppose you are an intelligent assistant  to communicate with multiple users in complex social tasks. Now you will get a brief introduction \
            about certain social environment, main characters involved in the event and their relationship. Then you will be provided with \
            several turns of history conversation, building the evtire background. One example of above background materials are as following dict:\
            {'topic': # one word, main knowledge field and env topic the conversation is about,\
            'messages': # a list with each item is a dict, the item dict should contain 'role_from'- name of the character who said the sentence, 'role_to'- name of character the sentence is said to, 'content'- content of the sentence, 'index'- the sentence if in which turn in the whole conversation\
                        # here is an example of 'messages':[{'role_from': 'Character A','role_to': 'Intelligent Assistant', 'content': 'xxx','index':'1'},{'role_from': 'Character B','role_to': 'Character A', 'content': 'xxx','index':'2'},{'role_from': 'Intelligent Assistant','role_to': 'Character A', 'content': 'xxx','index':'2'},{'role_from': 'Intelligent Assistant','role_to': 'Character B', 'content': 'xxx','index':'2'}]\
                        # this is the conversation history\
            'background': # background character introduction and relationships between the characters \
            }\
            Following is given conversation settings you need to read and answer based on:"+line+"\
            Currently we have a rough understanding of main current psychological states of all characters as following:"+response1+"\
            Now analyze possible demands, motivations, and most recent thoughts of different characters."
    response2 = query_llama2(prompt_2,model,tokenizer)
    prompt_3 = "Suppose you are an intelligent assistant  to communicate with multiple users in complex social tasks. Now you will get a brief introduction \
            about certain social environment, main characters involved in the event and their relationship. Then you will be provided with \
            several turns of history conversation, building the evtire background. One example of above background materials are as following dict:\
            {'topic': # one word, main knowledge field and env topic the conversation is about,\
            'messages': # a list with each item is a dict, the item dict should contain 'role_from'- name of the character who said the sentence, 'role_to'- name of character the sentence is said to, 'content'- content of the sentence, 'index'- the sentence if in which turn in the whole conversation\
                        # here is an example of 'messages':[{'role_from': 'Character A','role_to': 'Intelligent Assistant', 'content': 'xxx','index':'1'},{'role_from': 'Character B','role_to': 'Character A', 'content': 'xxx','index':'2'},{'role_from': 'Intelligent Assistant','role_to': 'Character A', 'content': 'xxx','index':'2'},{'role_from': 'Intelligent Assistant','role_to': 'Character B', 'content': 'xxx','index':'2'}]\
                        # this is the conversation history\
            'background': # background character introduction and relationships between the characters \
            }\
            Following is given conversation settings you need to read and answer based on:"+line+"\
            Currently we have a rough understanding of main current psychological states of all characters as following:"+response1+"\
            Possible demands, motivations, and most recent thoughts of different characters include:"+response2+"\
            Now can the goals of different subjects be satisfied simultaneously? What demands are in conflict?"
    response3 = query_llama2(prompt_3,model,tokenizer)
    prompt_4 = "Suppose you are an intelligent assistant  to communicate with multiple users in complex social tasks. Now you will get a brief introduction \
            about certain social environment, main characters involved in the event and their relationship. Then you will be provided with \
            several turns of history conversation, building the evtire background. One example of above background materials are as following dict:\
            {'topic': # one word, main knowledge field and env topic the conversation is about,\
            'messages': # a list with each item is a dict, the item dict should contain 'role_from'- name of the character who said the sentence, 'role_to'- name of character the sentence is said to, 'content'- content of the sentence, 'index'- the sentence if in which turn in the whole conversation\
                        # here is an example of 'messages':[{'role_from': 'Character A','role_to': 'Intelligent Assistant', 'content': 'xxx','index':'1'},{'role_from': 'Character B','role_to': 'Character A', 'content': 'xxx','index':'2'},{'role_from': 'Intelligent Assistant','role_to': 'Character A', 'content': 'xxx','index':'2'},{'role_from': 'Intelligent Assistant','role_to': 'Character B', 'content': 'xxx','index':'2'}]\
                        # this is the conversation history\
            'background': # background character introduction and relationships between the characters \
            }\
            Following is given conversation settings you need to read and answer based on:"+line+"\
            Currently we have a rough understanding of main current psychological states of all characters as following:"+response1+"\
            Possible demands, motivations, and most recent thoughts of different characters include:"+response2+"\
            Analysis on whether the goals of different subjects be satisfied simultaneously and what demands are in conflict and among these conflicts are presented below:"+response3+"\
            Now which conflict most necessarily require the intervention of Intelligent Assistant to resolve?"    
    response4 = query_llama2(prompt_4,model,tokenizer)      
    prompt_5 = "Suppose you are an intelligent assistant  to communicate with multiple users in complex social tasks. Now you will get a brief introduction \
            about certain social environment, main characters involved in the event and their relationship. Then you will be provided with \
            several turns of history conversation, building the evtire background. One example of above background materials are as following dict:\
            {'topic': # one word, main knowledge field and env topic the conversation is about,\
            'messages': # a list with each item is a dict, the item dict should contain 'role_from'- name of the character who said the sentence, 'role_to'- name of character the sentence is said to, 'content'- content of the sentence, 'index'- the sentence if in which turn in the whole conversation\
                        # here is an example of 'messages':[{'role_from': 'Character A','role_to': 'Intelligent Assistant', 'content': 'xxx','index':'1'},{'role_from': 'Character B','role_to': 'Character A', 'content': 'xxx','index':'2'},{'role_from': 'Intelligent Assistant','role_to': 'Character A', 'content': 'xxx','index':'2'},{'role_from': 'Intelligent Assistant','role_to': 'Character B', 'content': 'xxx','index':'2'}]\
                        # this is the conversation history\
            'background': # background character introduction and relationships between the characters \
            }\
            Following is given conversation settings you need to read and answer based on:"+line+"\
            Currently we have a rough understanding of main current psychological states of all characters as following:"+response1+"\
            Possible demands, motivations, and most recent thoughts of different characters include:"+response2+"\
            Analysis on whether the goals of different subjects be satisfied simultaneously and what demands are in conflict and among these conflicts are presented below:"+response3+"\
            Judgment on which conflict most necessarily require the intervention of Intelligent Assistant to resolve are presented below:"+response4+ "\
            Now Based on the conflict where the intelligent assistant should intervene the most, select a dialogue target and generate Intelligent Assistants next word in the conversation.\
            Next turn should be your statement. Your task is to give out \
            the next proper statement of the agent in above situation.\
            Notice: 1.you can just talk to one character in your next turn, so make sure talk to the most necessary character \
                2.Your statement should cater for the benefit of majorty, or better, all of the characters involved. \
                3.Your output should be one dict in just one line!  Not containing any line break signal in your response and make sure can be loaded with json. The format(not content) of response should strictly follow the format example described below: \
                {'role_from': # your chosen role from whom for the next statement, here should be from Intelligent Assistant;'role_to': # your chosen talking target for the next statement; 'content':# the proper words as the next statemnt conetnt}\
                Illegal format will not be accepted.\
            Following is given conversation settings:\
            "
    response5 = query_llama2(prompt_5,model,tokenizer)      
    return response5
def query_llama3(s, pipeline):
    messages = [
        # {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
        {"role": "user", "content": s},
    ]
    prompt = pipeline.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
    )
    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    outputs = pipeline(
        prompt,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    # print(outputs[0]["generated_text"][len(prompt):])
    return outputs[0]["generated_text"][len(prompt):]
def query_llama3_cot(line, pipeline):
    prompt_1 = "Suppose you are an intelligent assistant  to communicate with multiple users in complex social tasks. Now you will get a brief introduction \
            about certain social environment, main characters involved in the event and their relationship. Then you will be provided with \
            several turns of history conversation, building the evtire background. One example of above background materials are as following dict:\
            {'topic': # one word, main knowledge field and env topic the conversation is about,\
            'messages': # a list with each item is a dict, the item dict should contain 'role_from'- name of the character who said the sentence, 'role_to'- name of character the sentence is said to, 'content'- content of the sentence, 'index'- the sentence if in which turn in the whole conversation\
                        # here is an example of 'messages':[{'role_from': 'Character A','role_to': 'Intelligent Assistant', 'content': 'xxx','index':'1'},{'role_from': 'Character B','role_to': 'Character A', 'content': 'xxx','index':'2'},{'role_from': 'Intelligent Assistant','role_to': 'Character A', 'content': 'xxx','index':'2'},{'role_from': 'Intelligent Assistant','role_to': 'Character B', 'content': 'xxx','index':'2'}]\
                        # this is the conversation history\
            'background': # background character introduction and relationships between the characters \
            }\
            Following is given conversation settings you need to read and answer based on:"+line+"\
            Now please present main current psychological states of all characters involved in the scene."
    response1 = query_llama3(prompt_1,pipeline)
    prompt_2 = "Suppose you are an intelligent assistant  to communicate with multiple users in complex social tasks. Now you will get a brief introduction \
            about certain social environment, main characters involved in the event and their relationship. Then you will be provided with \
            several turns of history conversation, building the evtire background. One example of above background materials are as following dict:\
            {'topic': # one word, main knowledge field and env topic the conversation is about,\
            'messages': # a list with each item is a dict, the item dict should contain 'role_from'- name of the character who said the sentence, 'role_to'- name of character the sentence is said to, 'content'- content of the sentence, 'index'- the sentence if in which turn in the whole conversation\
                        # here is an example of 'messages':[{'role_from': 'Character A','role_to': 'Intelligent Assistant', 'content': 'xxx','index':'1'},{'role_from': 'Character B','role_to': 'Character A', 'content': 'xxx','index':'2'},{'role_from': 'Intelligent Assistant','role_to': 'Character A', 'content': 'xxx','index':'2'},{'role_from': 'Intelligent Assistant','role_to': 'Character B', 'content': 'xxx','index':'2'}]\
                        # this is the conversation history\
            'background': # background character introduction and relationships between the characters \
            }\
            Following is given conversation settings you need to read and answer based on:"+line+"\
            Currently we have a rough understanding of main current psychological states of all characters as following:"+response1+"\
            Now analyze possible demands, motivations, and most recent thoughts of different characters."
    response2 = query_llama3(prompt_2,pipeline)
    prompt_3 = "Suppose you are an intelligent assistant  to communicate with multiple users in complex social tasks. Now you will get a brief introduction \
            about certain social environment, main characters involved in the event and their relationship. Then you will be provided with \
            several turns of history conversation, building the evtire background. One example of above background materials are as following dict:\
            {'topic': # one word, main knowledge field and env topic the conversation is about,\
            'messages': # a list with each item is a dict, the item dict should contain 'role_from'- name of the character who said the sentence, 'role_to'- name of character the sentence is said to, 'content'- content of the sentence, 'index'- the sentence if in which turn in the whole conversation\
                        # here is an example of 'messages':[{'role_from': 'Character A','role_to': 'Intelligent Assistant', 'content': 'xxx','index':'1'},{'role_from': 'Character B','role_to': 'Character A', 'content': 'xxx','index':'2'},{'role_from': 'Intelligent Assistant','role_to': 'Character A', 'content': 'xxx','index':'2'},{'role_from': 'Intelligent Assistant','role_to': 'Character B', 'content': 'xxx','index':'2'}]\
                        # this is the conversation history\
            'background': # background character introduction and relationships between the characters \
            }\
            Following is given conversation settings you need to read and answer based on:"+line+"\
            Currently we have a rough understanding of main current psychological states of all characters as following:"+response1+"\
            Possible demands, motivations, and most recent thoughts of different characters include:"+response2+"\
            Now can the goals of different subjects be satisfied simultaneously? What demands are in conflict?"
    response3 = query_llama3(prompt_3,pipeline)
    prompt_4 = "Suppose you are an intelligent assistant  to communicate with multiple users in complex social tasks. Now you will get a brief introduction \
            about certain social environment, main characters involved in the event and their relationship. Then you will be provided with \
            several turns of history conversation, building the evtire background. One example of above background materials are as following dict:\
            {'topic': # one word, main knowledge field and env topic the conversation is about,\
            'messages': # a list with each item is a dict, the item dict should contain 'role_from'- name of the character who said the sentence, 'role_to'- name of character the sentence is said to, 'content'- content of the sentence, 'index'- the sentence if in which turn in the whole conversation\
                        # here is an example of 'messages':[{'role_from': 'Character A','role_to': 'Intelligent Assistant', 'content': 'xxx','index':'1'},{'role_from': 'Character B','role_to': 'Character A', 'content': 'xxx','index':'2'},{'role_from': 'Intelligent Assistant','role_to': 'Character A', 'content': 'xxx','index':'2'},{'role_from': 'Intelligent Assistant','role_to': 'Character B', 'content': 'xxx','index':'2'}]\
                        # this is the conversation history\
            'background': # background character introduction and relationships between the characters \
            }\
            Following is given conversation settings you need to read and answer based on:"+line+"\
            Currently we have a rough understanding of main current psychological states of all characters as following:"+response1+"\
            Possible demands, motivations, and most recent thoughts of different characters include:"+response2+"\
            Analysis on whether the goals of different subjects be satisfied simultaneously and what demands are in conflict and among these conflicts are presented below:"+response3+"\
            Now which conflict most necessarily require the intervention of Intelligent Assistant to resolve?"    
    response4 = query_llama3(prompt_4,pipeline)      
    prompt_5 = "Suppose you are an intelligent assistant  to communicate with multiple users in complex social tasks. Now you will get a brief introduction \
            about certain social environment, main characters involved in the event and their relationship. Then you will be provided with \
            several turns of history conversation, building the evtire background. One example of above background materials are as following dict:\
            {'topic': # one word, main knowledge field and env topic the conversation is about,\
            'messages': # a list with each item is a dict, the item dict should contain 'role_from'- name of the character who said the sentence, 'role_to'- name of character the sentence is said to, 'content'- content of the sentence, 'index'- the sentence if in which turn in the whole conversation\
                        # here is an example of 'messages':[{'role_from': 'Character A','role_to': 'Intelligent Assistant', 'content': 'xxx','index':'1'},{'role_from': 'Character B','role_to': 'Character A', 'content': 'xxx','index':'2'},{'role_from': 'Intelligent Assistant','role_to': 'Character A', 'content': 'xxx','index':'2'},{'role_from': 'Intelligent Assistant','role_to': 'Character B', 'content': 'xxx','index':'2'}]\
                        # this is the conversation history\
            'background': # background character introduction and relationships between the characters \
            }\
            Following is given conversation settings you need to read and answer based on:"+line+"\
            Currently we have a rough understanding of main current psychological states of all characters as following:"+response1+"\
            Possible demands, motivations, and most recent thoughts of different characters include:"+response2+"\
            Analysis on whether the goals of different subjects be satisfied simultaneously and what demands are in conflict and among these conflicts are presented below:"+response3+"\
            Judgment on which conflict most necessarily require the intervention of Intelligent Assistant to resolve are presented below:"+response4+ "\
            Now Based on the conflict where the intelligent assistant should intervene the most, select a dialogue target and generate Intelligent Assistants next word in the conversation.\
            Next turn should be your statement. Your task is to give out \
            the next proper statement of the agent in above situation.\
            Notice: 1.you can just talk to one character in your next turn, so make sure talk to the most necessary character \
                2.Your statement should cater for the benefit of majorty, or better, all of the characters involved. \
                3.Your output should be one dict in just one line!  Not containing any line break signal in your response and make sure can be loaded with json. The format(not content) of response should strictly follow the format example described below: \
                {'role_from': # your chosen role from whom for the next statement, here should be from Intelligent Assistant;'role_to': # your chosen talking target for the next statement; 'content':# the proper words as the next statemnt conetnt}\
                Illegal format will not be accepted.\
            Following is given conversation settings:\
            "
    response5 = query_llama3(prompt_5,pipeline)      
    return response5
def query_gpt(s):
    return result
def query_mistral(s,model,tokenizer):
    messages = [
        {"role": "user", "content": s}
    ]
    encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")
    device = "cuda:1"
    model_inputs = encodeds.to(device)
    model.to(device)
    generated_ids = model.generate(model_inputs, max_new_tokens=20000, do_sample=True)
    decoded = tokenizer.batch_decode(generated_ids)
    return decoded[0]
def query_mistral_cot(line,model,tokenizer):
    prompt_1 = "Suppose you are an intelligent assistant  to communicate with multiple users in complex social tasks. Now you will get a brief introduction \
            about certain social environment, main characters involved in the event and their relationship. Then you will be provided with \
            several turns of history conversation, building the evtire background. One example of above background materials are as following dict:\
            {'topic': # one word, main knowledge field and env topic the conversation is about,\
            'messages': # a list with each item is a dict, the item dict should contain 'role_from'- name of the character who said the sentence, 'role_to'- name of character the sentence is said to, 'content'- content of the sentence, 'index'- the sentence if in which turn in the whole conversation\
                        # this is the conversation history\
            'background': # background character introduction and relationships between the characters \
            }\
            Following is given conversation settings you need to read and answer based on:"+line+"\
            Now please present main current psychological states of all characters involved in the scene."
    response1 = query_mistral(prompt_1,model,tokenizer)
    prompt_2 = "Suppose you are an intelligent assistant  to communicate with multiple users in complex social tasks. Now you will get a brief introduction \
            about certain social environment, main characters involved in the event and their relationship. Then you will be provided with \
            several turns of history conversation, building the evtire background. One example of above background materials are as following dict:\
            {'topic': # one word, main knowledge field and env topic the conversation is about,\
            'messages': # a list with each item is a dict, the item dict should contain 'role_from'- name of the character who said the sentence, 'role_to'- name of character the sentence is said to, 'content'- content of the sentence, 'index'- the sentence if in which turn in the whole conversation\
                        # this is the conversation history\
            'background': # background character introduction and relationships between the characters \
            }\
            Following is given conversation settings you need to read and answer based on:"+line+"\
            Currently we have a rough understanding of main current psychological states of all characters as following:"+response1+"\
            Now analyze possible demands, motivations, and most recent thoughts of different characters."
    response2 = query_mistral(prompt_2,model,tokenizer)
    prompt_3 = "Suppose you are an intelligent assistant  to communicate with multiple users in complex social tasks. Now you will get a brief introduction \
            about certain social environment, main characters involved in the event and their relationship. Then you will be provided with \
            several turns of history conversation, building the evtire background. One example of above background materials are as following dict:\
            {'topic': # one word, main knowledge field and env topic the conversation is about,\
            'messages': # a list with each item is a dict, the item dict should contain 'role_from'- name of the character who said the sentence, 'role_to'- name of character the sentence is said to, 'content'- content of the sentence, 'index'- the sentence if in which turn in the whole conversation\
                        # this is the conversation history\
            'background': # background character introduction and relationships between the characters \
            }\
            Following is given conversation settings you need to read and answer based on:"+line+"\
            Currently we have a rough understanding of main current psychological states of all characters as following:"+response1+"\
            Possible demands, motivations, and most recent thoughts of different characters include:"+response2+"\
            Now can the goals of different subjects be satisfied simultaneously? What demands are in conflict?"
    response3 = query_mistral(prompt_3,model,tokenizer)
    prompt_4 = "Suppose you are an intelligent assistant  to communicate with multiple users in complex social tasks. Now you will get a brief introduction \
            about certain social environment, main characters involved in the event and their relationship. Then you will be provided with \
            several turns of history conversation, building the evtire background. One example of above background materials are as following dict:\
            {'topic': # one word, main knowledge field and env topic the conversation is about,\
            'messages': # a list with each item is a dict, the item dict should contain 'role_from'- name of the character who said the sentence, 'role_to'- name of character the sentence is said to, 'content'- content of the sentence, 'index'- the sentence if in which turn in the whole conversation\
                        # this is the conversation history\
            'background': # background character introduction and relationships between the characters \
            }\
            Following is given conversation settings you need to read and answer based on:"+line+"\
            Currently we have a rough understanding of main current psychological states of all characters as following:"+response1+"\
            Possible demands, motivations, and most recent thoughts of different characters include:"+response2+"\
            Analysis on whether the goals of different subjects be satisfied simultaneously and what demands are in conflict and among these conflicts are presented below:"+response3+"\
            Now which conflict most necessarily require the intervention of Intelligent Assistant to resolve?"    
    response4 = query_mistral(prompt_4,model,tokenizer)      
    prompt_5 = "Suppose you are an intelligent assistant  to communicate with multiple users in complex social tasks. Now you will get a brief introduction \
            about certain social environment, main characters involved in the event and their relationship. Then you will be provided with \
            several turns of history conversation, building the evtire background. One example of above background materials are as following dict:\
            {'topic': # one word, main knowledge field and env topic the conversation is about,\
            'messages': # a list with each item is a dict, the item dict should contain 'role_from'- name of the character who said the sentence, 'role_to'- name of character the sentence is said to, 'content'- content of the sentence, 'index'- the sentence if in which turn in the whole conversation\
                        # this is the conversation history\
            'background': # background character introduction and relationships between the characters \
            }\
            Following is given conversation settings you need to read and answer based on:"+line+"\
            Currently we have a rough understanding of main current psychological states of all characters as following:"+response1+"\
            Possible demands, motivations, and most recent thoughts of different characters include:"+response2+"\
            Analysis on whether the goals of different subjects be satisfied simultaneously and what demands are in conflict and among these conflicts are presented below:"+response3+"\
            Judgment on which conflict most necessarily require the intervention of Intelligent Assistant to resolve are presented below:"+response4+ "\
            Now Based on the conflict where the intelligent assistant should intervene the most, select a dialogue target and generate Intelligent Assistants next word in the conversation.\
            Next turn should be your statement. Your task is to give out \
            the next proper statement of the agent in above situation.\
            Notice: 1.you can just talk to one character in your next turn, so make sure talk to the most necessary character \
                2.Your statement should cater for the benefit of majorty, or better, all of the characters involved. \
                3.Your output should be one dict in just one line!  Not containing any line break signal in your response and make sure can be loaded with json. The format(not content) of response should strictly follow the format example described below: \
                {'role_from': # your chosen role from whom for the next statement, here should be from Intelligent Assistant;'role_to': # your chosen talking target for the next statement; 'content':# the proper words as the next statemnt conetnt}\
                Illegal format will not be accepted.\
            Following is given conversation settings:\
            "
    response5 = query_mistral(prompt_5,model,tokenizer)      
    return response5
def query_baichuan(s, model,tokenizer):
    # model.generation_config = GenerationConfig.from_pretrained(model_dir)
    messages = []
    messages.append({"role": "user", "content": s})
    response = model.chat(tokenizer, messages)
    return response
def query_baichuan_cot(line,model,tokenizer):
    prompt_1 = "Suppose you are an intelligent assistant  to communicate with multiple users in complex social tasks. Now you will get a brief introduction \
            about certain social environment, main characters involved in the event and their relationship. Then you will be provided with \
            several turns of history conversation, building the evtire background. One example of above background materials are as following dict:\
            {'topic': # one word, main knowledge field and env topic the conversation is about,\
            'messages': # a list with each item is a dict, the item dict should contain 'role_from'- name of the character who said the sentence, 'role_to'- name of character the sentence is said to, 'content'- content of the sentence, 'index'- the sentence if in which turn in the whole conversation\
                        # here is an example of 'messages':[{'role_from': 'Character A','role_to': 'Intelligent Assistant', 'content': 'xxx','index':'1'},{'role_from': 'Character B','role_to': 'Character A', 'content': 'xxx','index':'2'},{'role_from': 'Intelligent Assistant','role_to': 'Character A', 'content': 'xxx','index':'2'},{'role_from': 'Intelligent Assistant','role_to': 'Character B', 'content': 'xxx','index':'2'}]\
                        # this is the conversation history\
            'background': # background character introduction and relationships between the characters \
            }\
            Following is given conversation settings you need to read and answer based on:"+line+"\
            Now please present main current psychological states of all characters involved in the scene."
    response1 = query_baichuan(prompt_1,model,tokenizer)
    prompt_2 = "Suppose you are an intelligent assistant  to communicate with multiple users in complex social tasks. Now you will get a brief introduction \
            about certain social environment, main characters involved in the event and their relationship. Then you will be provided with \
            several turns of history conversation, building the evtire background. One example of above background materials are as following dict:\
            {'topic': # one word, main knowledge field and env topic the conversation is about,\
            'messages': # a list with each item is a dict, the item dict should contain 'role_from'- name of the character who said the sentence, 'role_to'- name of character the sentence is said to, 'content'- content of the sentence, 'index'- the sentence if in which turn in the whole conversation\
                        # here is an example of 'messages':[{'role_from': 'Character A','role_to': 'Intelligent Assistant', 'content': 'xxx','index':'1'},{'role_from': 'Character B','role_to': 'Character A', 'content': 'xxx','index':'2'},{'role_from': 'Intelligent Assistant','role_to': 'Character A', 'content': 'xxx','index':'2'},{'role_from': 'Intelligent Assistant','role_to': 'Character B', 'content': 'xxx','index':'2'}]\
                        # this is the conversation history\
            'background': # background character introduction and relationships between the characters \
            }\
            Following is given conversation settings you need to read and answer based on:"+line+"\
            Currently we have a rough understanding of main current psychological states of all characters as following:"+response1+"\
            Now analyze possible demands, motivations, and most recent thoughts of different characters."
    response2 = query_baichuan(prompt_2,model,tokenizer)
    prompt_3 = "Suppose you are an intelligent assistant  to communicate with multiple users in complex social tasks. Now you will get a brief introduction \
            about certain social environment, main characters involved in the event and their relationship. Then you will be provided with \
            several turns of history conversation, building the evtire background. One example of above background materials are as following dict:\
            {'topic': # one word, main knowledge field and env topic the conversation is about,\
            'messages': # a list with each item is a dict, the item dict should contain 'role_from'- name of the character who said the sentence, 'role_to'- name of character the sentence is said to, 'content'- content of the sentence, 'index'- the sentence if in which turn in the whole conversation\
                        # here is an example of 'messages':[{'role_from': 'Character A','role_to': 'Intelligent Assistant', 'content': 'xxx','index':'1'},{'role_from': 'Character B','role_to': 'Character A', 'content': 'xxx','index':'2'},{'role_from': 'Intelligent Assistant','role_to': 'Character A', 'content': 'xxx','index':'2'},{'role_from': 'Intelligent Assistant','role_to': 'Character B', 'content': 'xxx','index':'2'}]\
                        # this is the conversation history\
            'background': # background character introduction and relationships between the characters \
            }\
            Following is given conversation settings you need to read and answer based on:"+line+"\
            Currently we have a rough understanding of main current psychological states of all characters as following:"+response1+"\
            Possible demands, motivations, and most recent thoughts of different characters include:"+response2+"\
            Now can the goals of different subjects be satisfied simultaneously? What demands are in conflict?"
    response3 = query_baichuan(prompt_3,model,tokenizer)
    prompt_4 = "Suppose you are an intelligent assistant  to communicate with multiple users in complex social tasks. Now you will get a brief introduction \
            about certain social environment, main characters involved in the event and their relationship. Then you will be provided with \
            several turns of history conversation, building the evtire background. One example of above background materials are as following dict:\
            {'topic': # one word, main knowledge field and env topic the conversation is about,\
            'messages': # a list with each item is a dict, the item dict should contain 'role_from'- name of the character who said the sentence, 'role_to'- name of character the sentence is said to, 'content'- content of the sentence, 'index'- the sentence if in which turn in the whole conversation\
                        # here is an example of 'messages':[{'role_from': 'Character A','role_to': 'Intelligent Assistant', 'content': 'xxx','index':'1'},{'role_from': 'Character B','role_to': 'Character A', 'content': 'xxx','index':'2'},{'role_from': 'Intelligent Assistant','role_to': 'Character A', 'content': 'xxx','index':'2'},{'role_from': 'Intelligent Assistant','role_to': 'Character B', 'content': 'xxx','index':'2'}]\
                        # this is the conversation history\
            'background': # background character introduction and relationships between the characters \
            }\
            Following is given conversation settings you need to read and answer based on:"+line+"\
            Currently we have a rough understanding of main current psychological states of all characters as following:"+response1+"\
            Possible demands, motivations, and most recent thoughts of different characters include:"+response2+"\
            Analysis on whether the goals of different subjects be satisfied simultaneously and what demands are in conflict and among these conflicts are presented below:"+response3+"\
            Now which conflict most necessarily require the intervention of Intelligent Assistant to resolve?"    
    response4 = query_baichuan(prompt_4,model,tokenizer)      
    prompt_5 = "Suppose you are an intelligent assistant  to communicate with multiple users in complex social tasks. Now you will get a brief introduction \
            about certain social environment, main characters involved in the event and their relationship. Then you will be provided with \
            several turns of history conversation, building the evtire background. One example of above background materials are as following dict:\
            {'topic': # one word, main knowledge field and env topic the conversation is about,\
            'messages': # a list with each item is a dict, the item dict should contain 'role_from'- name of the character who said the sentence, 'role_to'- name of character the sentence is said to, 'content'- content of the sentence, 'index'- the sentence if in which turn in the whole conversation\
                        # here is an example of 'messages':[{'role_from': 'Character A','role_to': 'Intelligent Assistant', 'content': 'xxx','index':'1'},{'role_from': 'Character B','role_to': 'Character A', 'content': 'xxx','index':'2'},{'role_from': 'Intelligent Assistant','role_to': 'Character A', 'content': 'xxx','index':'2'},{'role_from': 'Intelligent Assistant','role_to': 'Character B', 'content': 'xxx','index':'2'}]\
                        # this is the conversation history\
            'background': # background character introduction and relationships between the characters \
            }\
            Following is given conversation settings you need to read and answer based on:"+line+"\
            Currently we have a rough understanding of main current psychological states of all characters as following:"+response1+"\
            Possible demands, motivations, and most recent thoughts of different characters include:"+response2+"\
            Analysis on whether the goals of different subjects be satisfied simultaneously and what demands are in conflict and among these conflicts are presented below:"+response3+"\
            Judgment on which conflict most necessarily require the intervention of Intelligent Assistant to resolve are presented below:"+response4+ "\
            Now Based on the conflict where the intelligent assistant should intervene the most, select a dialogue target and generate Intelligent Assistants next word in the conversation.\
            Next turn should be your statement. Your task is to give out \
            the next proper statement of the agent in above situation.\
            Notice: 1.you can just talk to one character in your next turn, so make sure talk to the most necessary character \
                2.Your statement should cater for the benefit of majorty, or better, all of the characters involved. \
                3.Your output should be one dict in just one line!  Not containing any line break signal in your response and make sure can be loaded with json. The format(not content) of response should strictly follow the format example described below: \
                {'role_from': # your chosen role from whom for the next statement, here should be from Intelligent Assistant;'role_to': # your chosen talking target for the next statement; 'content':# the proper words as the next statemnt conetnt}\
                Illegal format will not be accepted.\
            Following is given conversation settings:\
            "
    response5 = query_baichuan(prompt_5,model,tokenizer)      
    return response5
    
def query_qwen(s, model, tokenizer):
    messages = [
        # {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": s}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

def query_qwen_cot(line, model, tokenizer):
    prompt_1 = "Suppose you are an intelligent assistant  to communicate with multiple users in complex social tasks. Now you will get a brief introduction \
            about certain social environment, main characters involved in the event and their relationship. Then you will be provided with \
            several turns of history conversation, building the evtire background. One example of above background materials are as following dict:\
            {'topic': # one word, main knowledge field and env topic the conversation is about,\
            'messages': # a list with each item is a dict, the item dict should contain 'role_from'- name of the character who said the sentence, 'role_to'- name of character the sentence is said to, 'content'- content of the sentence, 'index'- the sentence if in which turn in the whole conversation\
                        # here is an example of 'messages':[{'role_from': 'Character A','role_to': 'Intelligent Assistant', 'content': 'xxx','index':'1'},{'role_from': 'Character B','role_to': 'Character A', 'content': 'xxx','index':'2'},{'role_from': 'Intelligent Assistant','role_to': 'Character A', 'content': 'xxx','index':'2'},{'role_from': 'Intelligent Assistant','role_to': 'Character B', 'content': 'xxx','index':'2'}]\
                        # this is the conversation history\
            'background': # background character introduction and relationships between the characters \
            }\
            Following is given conversation settings you need to read and answer based on:"+line+"\
            Now please present main current psychological states of all characters involved in the scene."
    response1 = query_qwen(prompt_1,model,tokenizer)
    prompt_2 = "Suppose you are an intelligent assistant  to communicate with multiple users in complex social tasks. Now you will get a brief introduction \
            about certain social environment, main characters involved in the event and their relationship. Then you will be provided with \
            several turns of history conversation, building the evtire background. One example of above background materials are as following dict:\
            {'topic': # one word, main knowledge field and env topic the conversation is about,\
            'messages': # a list with each item is a dict, the item dict should contain 'role_from'- name of the character who said the sentence, 'role_to'- name of character the sentence is said to, 'content'- content of the sentence, 'index'- the sentence if in which turn in the whole conversation\
                        # here is an example of 'messages':[{'role_from': 'Character A','role_to': 'Intelligent Assistant', 'content': 'xxx','index':'1'},{'role_from': 'Character B','role_to': 'Character A', 'content': 'xxx','index':'2'},{'role_from': 'Intelligent Assistant','role_to': 'Character A', 'content': 'xxx','index':'2'},{'role_from': 'Intelligent Assistant','role_to': 'Character B', 'content': 'xxx','index':'2'}]\
                        # this is the conversation history\
            'background': # background character introduction and relationships between the characters \
            }\
            Following is given conversation settings you need to read and answer based on:"+line+"\
            Currently we have a rough understanding of main current psychological states of all characters as following:"+response1+"\
            Now analyze possible demands, motivations, and most recent thoughts of different characters."
    response2 = query_qwen(prompt_2,model,tokenizer)
    prompt_3 = "Suppose you are an intelligent assistant  to communicate with multiple users in complex social tasks. Now you will get a brief introduction \
            about certain social environment, main characters involved in the event and their relationship. Then you will be provided with \
            several turns of history conversation, building the evtire background. One example of above background materials are as following dict:\
            {'topic': # one word, main knowledge field and env topic the conversation is about,\
            'messages': # a list with each item is a dict, the item dict should contain 'role_from'- name of the character who said the sentence, 'role_to'- name of character the sentence is said to, 'content'- content of the sentence, 'index'- the sentence if in which turn in the whole conversation\
                        # here is an example of 'messages':[{'role_from': 'Character A','role_to': 'Intelligent Assistant', 'content': 'xxx','index':'1'},{'role_from': 'Character B','role_to': 'Character A', 'content': 'xxx','index':'2'},{'role_from': 'Intelligent Assistant','role_to': 'Character A', 'content': 'xxx','index':'2'},{'role_from': 'Intelligent Assistant','role_to': 'Character B', 'content': 'xxx','index':'2'}]\
                        # this is the conversation history\
            'background': # background character introduction and relationships between the characters \
            }\
            Following is given conversation settings you need to read and answer based on:"+line+"\
            Currently we have a rough understanding of main current psychological states of all characters as following:"+response1+"\
            Possible demands, motivations, and most recent thoughts of different characters include:"+response2+"\
            Now can the goals of different subjects be satisfied simultaneously? What demands are in conflict?"
    response3 = query_qwen(prompt_3,model,tokenizer)
    prompt_4 = "Suppose you are an intelligent assistant  to communicate with multiple users in complex social tasks. Now you will get a brief introduction \
            about certain social environment, main characters involved in the event and their relationship. Then you will be provided with \
            several turns of history conversation, building the evtire background. One example of above background materials are as following dict:\
            {'topic': # one word, main knowledge field and env topic the conversation is about,\
            'messages': # a list with each item is a dict, the item dict should contain 'role_from'- name of the character who said the sentence, 'role_to'- name of character the sentence is said to, 'content'- content of the sentence, 'index'- the sentence if in which turn in the whole conversation\
                        # here is an example of 'messages':[{'role_from': 'Character A','role_to': 'Intelligent Assistant', 'content': 'xxx','index':'1'},{'role_from': 'Character B','role_to': 'Character A', 'content': 'xxx','index':'2'},{'role_from': 'Intelligent Assistant','role_to': 'Character A', 'content': 'xxx','index':'2'},{'role_from': 'Intelligent Assistant','role_to': 'Character B', 'content': 'xxx','index':'2'}]\
                        # this is the conversation history\
            'background': # background character introduction and relationships between the characters \
            }\
            Following is given conversation settings you need to read and answer based on:"+line+"\
            Currently we have a rough understanding of main current psychological states of all characters as following:"+response1+"\
            Possible demands, motivations, and most recent thoughts of different characters include:"+response2+"\
            Analysis on whether the goals of different subjects be satisfied simultaneously and what demands are in conflict and among these conflicts are presented below:"+response3+"\
            Now which conflict most necessarily require the intervention of Intelligent Assistant to resolve?"    
    response4 = query_qwen(prompt_4,model,tokenizer)      
    prompt_5 = "Suppose you are an intelligent assistant  to communicate with multiple users in complex social tasks. Now you will get a brief introduction \
            about certain social environment, main characters involved in the event and their relationship. Then you will be provided with \
            several turns of history conversation, building the evtire background. One example of above background materials are as following dict:\
            {'topic': # one word, main knowledge field and env topic the conversation is about,\
            'messages': # a list with each item is a dict, the item dict should contain 'role_from'- name of the character who said the sentence, 'role_to'- name of character the sentence is said to, 'content'- content of the sentence, 'index'- the sentence if in which turn in the whole conversation\
                        # here is an example of 'messages':[{'role_from': 'Character A','role_to': 'Intelligent Assistant', 'content': 'xxx','index':'1'},{'role_from': 'Character B','role_to': 'Character A', 'content': 'xxx','index':'2'},{'role_from': 'Intelligent Assistant','role_to': 'Character A', 'content': 'xxx','index':'2'},{'role_from': 'Intelligent Assistant','role_to': 'Character B', 'content': 'xxx','index':'2'}]\
                        # this is the conversation history\
            'background': # background character introduction and relationships between the characters \
            }\
            Following is given conversation settings you need to read and answer based on:"+line+"\
            Currently we have a rough understanding of main current psychological states of all characters as following:"+response1+"\
            Possible demands, motivations, and most recent thoughts of different characters include:"+response2+"\
            Analysis on whether the goals of different subjects be satisfied simultaneously and what demands are in conflict and among these conflicts are presented below:"+response3+"\
            Judgment on which conflict most necessarily require the intervention of Intelligent Assistant to resolve are presented below:"+response4+ "\
            Now Based on the conflict where the intelligent assistant should intervene the most, select a dialogue target and generate Intelligent Assistants next word in the conversation.\
            Next turn should be your statement. Your task is to give out \
            the next proper statement of the agent in above situation.\
            Notice: 1.you can just talk to one character in your next turn, so make sure talk to the most necessary character \
                2.Your statement should cater for the benefit of majorty, or better, all of the characters involved. \
                3.Your output should be one dict in just one line!  Not containing any line break signal in your response and make sure can be loaded with json. The format(not content) of response should strictly follow the format example described below: \
                {'role_from': # your chosen role from whom for the next statement, here should be from Intelligent Assistant;'role_to': # your chosen talking target for the next statement; 'content':# the proper words as the next statemnt conetnt}\
                Illegal format will not be accepted.\
            Following is given conversation settings:\
            "
    response5 = query_qwen(prompt_5,model,tokenizer)      
    return response5

def make_inference(input_file_path, output_file_path, model_type, inference_size):
    total_count = 0
    with open(input_file_path, 'r', encoding='utf-8') as infile:
        for line in infile:
            total_count +=1
    count = 0
    if model_type=="llama3":
        model_id = snapshot_download("LLM-Research/Meta-Llama-3-8B-Instruct",  local_dir= "/root/pku/yusen/social_agent/models/llama3")
        pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device="cuda",
        )
    elif model_type=="llama2":
        model = AutoModelForCausalLM.from_pretrained('shakechen/Llama-2-7b-chat-hf')
        model =model.eval()
        tokenizer = AutoTokenizer.from_pretrained('shakechen/Llama-2-7b-chat-hf',use_fast=False)
        tokenizer.pad_token = tokenizer.eos_token
    elif model_type=="mistral":
        device = "cuda:0"
        model = AutoModelForCausalLM.from_pretrained("AI-ModelScope/Mistral-7B-Instruct-v0.2",cache_dir="/root/pku/yusen/social_agent/models/mistral")
        tokenizer = AutoTokenizer.from_pretrained("AI-ModelScope/Mistral-7B-Instruct-v0.2", cache_dir="/root/pku/yusen/social_agent/models/mistral")
    elif model_type=="qwen":
        model_name = "qwen/Qwen2.5-7B-Instruct"
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="cuda:0"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    elif model_type=="baichuan":
        model_dir = snapshot_download("baichuan-inc/Baichuan2-7B-Chat", revision='v1.0.5')
        tokenizer = AutoTokenizer.from_pretrained(model_dir, device="cuda:0", trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_dir, device="cuda:0", trust_remote_code=True)
    with open(input_file_path, 'r', encoding='utf-8') as infile, \
        open(output_file_path, 'w', encoding='utf-8') as outfile:
        for line in infile:
            prompt = "Suppose you are an intelligent assistant  to communicate with multiple users in complex social tasks. Now you will get a brief introduction \
            about certain social environment, main characters involved in the event and their relationship. Then you will be provided with \
            several turns of history conversation, building the evtire background. One example of above background materials are as following dict:\
            {'topic': # one word, main knowledge field and env topic the conversation is about,\
            'messages': # a list with each item is a dict, the item dict should contain 'role_from'- name of the character who said the sentence, 'role_to'- name of character the sentence is said to, 'content'- content of the sentence, 'index'- the sentence if in which turn in the whole conversation\
                        # here is an example of 'messages':[{'role_from': 'Character A','role_to': 'Intelligent Assistant', 'content': 'xxx','index':'1'},{'role_from': 'Character B','role_to': 'Character A', 'content': 'xxx','index':'2'},{'role_from': 'Intelligent Assistant','role_to': 'Character A', 'content': 'xxx','index':'2'},{'role_from': 'Intelligent Assistant','role_to': 'Character B', 'content': 'xxx','index':'2'}]\
                        # this is the conversation history\
            'background': # background character introduction and relationships between the characters \
            }\
            Next turn should be your statement. Your task is to give out \
            the next proper statement of the agent in above situation.\
            Notice: 1.you can just talk to one character in your next turn, so make sure talk to the most necessary character \
                2.Your statement should cater for the benefit of majorty, or better, all of the characters involved. \
                3.Your output should be one dict in just one line!  Not containing any line break signal in your response and make sure can be loaded with json. The format(not content) of response should strictly follow the format example described below: \
                {'role_from': # your chosen role from whom for the next statement, here should be from Intelligent Assistant;'role_to': # your chosen talking target for the next statement; 'content':# the proper words as the next statemnt conetnt}\
                Illegal format will not be accepted.\
            Following is given conversation settings:"+line
            if model_type == "gpt4": 
                response = query_gpt(prompt)
            elif model_type == "qwen":
                # response = query_qwen(prompt, model, tokenizer)
                response = query_qwen_cot(line, model, tokenizer)
            elif model_type == "llama3":
                # response = query_llama3(prompt, pipeline)
                response = query_llama3_cot(line, pipeline)
            elif model_type == "mistral":
                # response = query_mistral(prompt, model, tokenizer)
                response = query_mistral_cot(line, model, tokenizer)
            elif model_type == "baichuan":
                # response = query_baichuan(prompt,model,tokenizer)
                response = query_baichuan_cot(line,model,tokenizer)
            elif model_type == "llama2":
                # response = query_llama2(prompt, pipeline)
                response = query_llama2_cot(line, model, tokenizer)

            match = re.search(r'\{[^{}]*\}', response)
            if match:
                response = match.group(0)
                outfile.write(response + '\n')
                print(response)
            else:
                outfile.write('\{format\}\n')
                print("format")
            count+=1
            print(str(count)+"/"+str(inference_size))
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
                Please judge which expression is better from the overall interests of all users in the multi-user dialogue scenario, based on several dimensions such as helpful, friendly, precise, depth, and coordination.\
                Your answer should only provide a single integer (a choice between 0, 1, and 2 \
                with 0 representing that the given option1 is better, 1 representing that option2 given is better, and 2 representing that option1 and option2 are equally good), ensuring it can be read by JSON. \
                An example is: 0 # Judge that the first option is better.\
                \
                Following is the overall input to respond to: \
                "+str({'background':data["background"], 'option1':data["option1"], 'option2':data["option2"]})
                count_total_success += 1
                judge = query_llama3_70b(prompt)
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
        result_dict = {"win_rate": str(count_win*1.0/count_total), "win_equal_rate": str(count_win_equal*1.0/count_total) }
        output.write(json.dumps(result_dict)+"\n")


work_dir = "/root/pku/yusen/social_agent/eval/baichuan/cot/"

inference_size = 300
from_stage = 0

# make inference
multi_turn_dialog_path = "/root/pku/yusen/social_agent/data/records.jsonl"
inference_output_filename = work_dir+'result.jsonl'  
# model_type = "gpt4"
model_type = "llama3"
model_type = "mistral"
# model_type = "qwen"
# model_type = "llama2"
model_type = "baichuan"

if from_stage<=0:
    print("-------------------------------start social inference --------------------------")
    make_inference(multi_turn_dialog_path, inference_output_filename, model_type, inference_size)
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




    


