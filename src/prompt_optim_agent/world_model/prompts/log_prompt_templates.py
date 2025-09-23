# log prompts

forward_log_tempelate = """---------------\tforward begin\t----------------
cur_prompt:\n{cur_prompt}
labels:  {labels}
preds:   {preds}
correct: {correct}
acc:     {acc}
---------------\tforward finished\t----------------
"""



optimize_log_tempelate_1st = """-------------\toptimize begin\t---------------
optimize_prompt:\n{optimize_prompt}
-------------\t\t---------------
response:\n{response}
-------------\toptimize finished\t---------------
"""

