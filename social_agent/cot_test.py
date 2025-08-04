


def test_cot_success(s):
    prompt_1 = "present the current psychological states of all subjects."
    prompt_2 = "Analyze the demands, motivations, and most recent thoughts of different subjects."
    prompt_3 = "Can the goals of different subjects be satisfied simultaneously? What demands are in conflict?"
    prompt_4 = "Among these conflicts, which are easier to resolve and which require the intervention of Intelligent Assistant to resolve?"
    prompt_5 = "Based on the conflict where the intelligent assistant should intervene the most, select a dialogue target and generate Intelligent Assistant's next word in the conversation."
    ans1 = get_ans(prompt1)
    ans2 = get_ans(prompt2)
    ans3 = get_ans(prompt3)
    # following all the same
    
    