<div align="center">
  <h1>
    No Loss, No Gain: Gated Refinement and Adaptive Compression for Prompt Optimization
    <br><br>
    <b>NIPS 2025</b>
    <br><br>
    <a href="https://arxiv.org/abs/ARXIV_ID" target="_blank">
      <img src="https://img.shields.io/badge/Paper%20ArXiv-GRACE-b31b1b.svg" alt="Paper ArXiv: GRACE">
    </a>
  </h1>
</div>


## Introduction

Unlike other methods that generate a large number of candidate updates in each iteration to ensure prompt improvement—often at the cost of efficiency, our GRACE method performs more targeted and effective prompt updates. Additionally, when optimization stagnates, GRACE activates adaptive compression to help escape local optima.

<p align="center">
<img src="./images/method.png" alt="Method Comparison" width="700" title="Method Comparison"/>
</p>

## Data
In the **datasets** folder, we provide the data for 5 challenging BBH tasks. All other datasets can be downloaded from HuggingFace.
 

## Quick Start

The following command run GRACE to optimize the prompt for a BIG-bench task, [snarks](https://github.com/google/BIG-bench/tree/main/bigbench/benchmark_tasks/snarks). 

**Note**: Before running this command, please add your (OpenAI) api key to the config/_.yaml file (base_model_setting: api_key and optim_model_setting: api_key). You can also check all the other auguments in the yaml file.
```bash
python src/main.py --config_dir config/snarks.yaml 
```

`snarks` is a classification task that asks which of two given statements is sarcastic. An example from the original dataset looks like this:
```
Determine which of two sentences is sarcastic.

Which statement is sarcastic?

Options:
(A) You're not allowed to walk in the park, you can only do it on your own property
(B) You're not allowed to smoke in the park, you can only do it on your own property
```
Then, the expected result is `A`.

The initial prompt from the BIG-bench dataset is `Determine which of two sentences is sarcastic.` Starting from this simple prompt, GRACE performs two-stage gated refinement based on both successful and failed training examples.
Furthermore, if optimization stagnates during the iterative process, GRACE applies adaptive compression, distilling core concepts from the prompt to open new directions for further optimization. The optimized prompt for `snarks` will look like this:

```
Identify sarcastic sentences by detecting contradictions between literal meaning and 
contextual intent. Analyze irony, exaggerated/dismissive language, rhetorical questions, 
and incongruities with common knowledge, situational context, or the speaker’s expected 
perspective (prioritizing typical assumptions about the speaker if unspecified). 
Prioritize mismatches between stated sentiment (positive/negative) and contextual 
plausibility, including mock endorsement of implausible perspectives, obvious falsehoods, 
trivialization of significant issues, or alignment with viewpoints the speaker would 
obviously oppose given the context. Consider both overt contradictions and subtle 
incongruities, particularly when tone or intensity is disproportionately exaggerated 
relative to the situation’s practical reality or the speaker’s implied stance. 
Additionally, evaluate whether the statement critiques or mockingly endorses widely 
recognized frustrations, overhyped trends (regardless of their actual merit), 
or common societal critiques, as sarcasm often arises from these contexts. 
Pay special attention to rhetorical questions that ironically affirm or deny propositions 
based on prevailing attitudes, and to statements where the speaker’s true stance is 
evident through contextual cues that contradict the literal message.
```

After optimization, all intermediate prompts and their corresponding validation scores are stored in a JSON file. We then evaluate the best nodes (based on validation score) on the test set.
Additionally, we log the number of API calls and input/output token usage for both the base LLM and the optimizer LLM throughout the process.


## Add new models

You can add a new.py file in [Models folder](https://github.com/Eric8932/GRACE/tree/main/src/prompt_optim_agent/language_model) to register a new model (e.g., a Hugging Face text-generation model or a vLLM model). The model class must implement two methods:
batch_forward_func: input a batch of prompts, output a batch of model's responses.

```bash
def batch_forward_func(self, batch_prompts: List(str)):
  ...
  return List(str)
```

generate: input one prompt, output one response
```bash
def generate(self, input: str):
  ...
  return str
```



## Citations
If you find this work useful, please kindly star the repo and and cite the paper below. For questions, contact wenhangshi@ruc.edu.cn, or open an issue. Thank you!

```bibtex
@misc{shi2025lossgaingatedrefinement,
      title={No Loss, No Gain: Gated Refinement and Adaptive Compression for Prompt Optimization}, 
      author={Wenhang Shi and Yiren Chen and Shuqing Bian and Xinyi Zhang and Kai Tang and Pengfei Hu and Zhe Zhao and Wei Lu and Xiaoyong Du},
      year={2025},
      eprint={2509.23387},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2509.23387}, 
}
```

## Acknowledgments

Our code is based in part on [PromptAgent](https://github.com/XinyuanWangCS/PromptAgent). Thanks for their great works.
