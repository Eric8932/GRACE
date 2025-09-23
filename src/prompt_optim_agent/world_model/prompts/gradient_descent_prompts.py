
gradient_prompt_tempelate = ""


example_template = """
<Example {index}> 
The model's input is:
{question}

The model's response (solution) is: 
{response}

The correct label is: {label}
The model's final prediction is: {prediction}.
"""


optimize_prompt_template_single_1st = """
Your task is to optimize the current prompt for a language model performing a specific task. The goal is to correct previously failed predictions while preserving the model’s correct behavior on already successful examples.

The current prompt is:
"{cur_prompt}"

This prompt was evaluated on a batch of examples.

It successfully handled the following examples:
{correct_string}


It failed on the following examples:
{error_string}


Please analyze both the successful and failed examples. 
Based on the example analysis, please optimize the current prompt under the following principles:

1. **Preserve Correctness**  
Ensure the model, instructed by the optimized prompt, continues to predict correct answers for all successful examples. In addition to prediction correctness, maintain the model’s original correct solutions and response for these cases as much as possible.

2. **Refine to Fix Errors**  
For failed examples, attempt to correct them by refining the prompt’s instructions — for example, by adding clearer or more complete guidance. 
Any new content should integrate naturally with the current prompt and form a coherent task instruction. Avoid special-case logic, examples, or instructions targeted at individual cases.


Additional guidelines:
- Prompt modifications should always aim to preserve model's correct behavior on successful examples.
- All changes should be minimal, necessary, and stable across iterations.
- The optimized prompt should be generalizable across different cases, rather than focusing on specific vocabulary or phrasing
- Only optimize the current prompt. Do not include input formats, verbalizers, or other fixed components.
- Provide the final optimized prompt within <START> and </START>.

""".strip()









optimize_prompt_template_single_2nd_simp = """
Your task is to reconstruct a cleaner, more concise version of the current prompt for a language model.
    
The current prompt is:
"{cur_prompt}"

The prompt may have accumulated redundant, overly specific, or ineffective wording across previous iterations. Your goal is to simplify and restructure it into a more effective and streamlined form — one that retains its core guidance while leaving room for future refinement. 

Guidelines:
- Eliminate instructions that are verbose, ambiguous, or unlikely to generalize. Preserve the core intent and task framing, but express it as clearly and simply as possible. 
- The new prompt should be self-contained, compact, and easy to iterate on in later optimization rounds.
- Provide the final optimized prompt within <START> and </START>.  

""".strip()