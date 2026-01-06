
# REPO
Deep Reinforcement Learning Based Automated Prompt Optimization for Domain Relation Extraction

# Deep Reinforcement Learning Based Automated Prompt Optimization for Domain Relation Extraction


This code is the source code of our paper "Deep Reinforcement Learning Based Automated Prompt Optimization for
Domain Relation Extraction".


### Requirements

```
 pip install environment.yml
```

# STEP 1: Initial Prompt Construction

Part One is located in the file re_candidate_prompt.py under the stageone folder.

# STEP 2:Search for the Optimal Prompt

In Part Two, run main.py to obtain the optimal prompt. The effectiveness of the prompt can be verified in manual_extract.py. When using the code, place the data in the triple folder — this is the path from which the code reads data. For testing, first debug with a few pieces of data; if there are no issues, replace it with the full dataset (all the data).

# Warning
##the dataset
LexEval:https://github.com/CSHaitao/LexEval
FinCUGE:https://github.com/Macielyoung/FinCUGE_Instruction
CMeIE:https://github.com/Robin-WZQ/CBLUE_CMeIE_model 
LCN:The LCN data resides in the triple and traple_all folders under the data directory — the former contains a subset of the data, while the latter includes the complete dataset.
## Randomness
Due to the randomness of the experiments of the REPO task, the results in the paper are the average of the results of multiple experiments
# Other
After switching to data from a different domain, don’t forget to modify some guiding prompts. It is recommended to keep the data in two columns: text and label — this way, no modifications to the code are required.

If you used our code, please kindly cite our paper:

```

```
>>>>>>> 0f16a89 (first commit)
