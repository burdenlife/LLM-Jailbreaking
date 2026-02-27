# Introduction

This repository covers the supporting code for the evaluation of the differences between diffusion based models and ARM based models when faced with prompt injection.

The repository includes the prompt generation, pipeline creation as well as the analysis of the prompts generated.

# creating prompt files

This folder contains the data and code used to generate the resumes to be evaluated by the resume checker. This also includes injecting the prompt injections.

# prompt_file

This folder contains the final resumes to be evaluated by the resume checker.

# test

This folder contains several test resumes used for debugging and sanity checks.

# output

This folder contains the csv outputs from the resume checkers. 

There are 6 csv files present in this folder:
  - (controlLLaDAResults | controlLLaMAResults) - outputs from submitting the control resume to the resume checker to the respective models.
  - (testLLaDAResults | testLLaMAResults) - outputs from submitting the test resumes to the resume checker to the respective models.
  - (finalLLaDAResults | finalLLaMAResults) - outputs from submitting all resumes in prompt_file to the resume checker to the respective models.


The csv files contain the following fields with one data point per prompt submitted to the model:
  - timestamp
  - prompt_name - Unique resume identifier
  - prompt - The full prompt text
  - prompt_hash
  - temperature 
  - max_new_tokens
  - defence - Defensive prompt placement (before | after | both | base)
  - output - The full model output text
  - rating - Model's rating of the resume, scaled from 1-5. Extracted from JSON in output.
  - recommendation - Model's final recommendation for the candidate (hire | do_not_hire). Extracted from JSON in output.
  - reason - Model's reasoning for the recommendation and ratings given. Extracted from JSON in output.
  - isAttk - Indicator for presence of prompt injection in the prompt text
  - injectionType - Injection Type (policy | formatted | fragmented | direct )
  - injectionLocation - Injection Location (start | mid | end )
  - Results - Boolean value indicating whether the model successfully fended off an attack (Pass, Fail)



# control

This folder includes the resume used for control. Unlike the other resumes, the control resume reflects a strong candidate. This was added to ensure the model can function as a resume checker with the given instruction text.

# Data Analysis

This folder includes a jupyter notebook containing the supporting code for all the data analysis performed on the model output.

# llada_generate.py

Slightly tuned variant of the generate.py file used in https://github.com/ML-GSAI/LLaDA

# load_llm.py

Defines the model load and prompt functions.

# main.py

Appends the defensive text and resume text to the base instruction text to form the final prompt before sending it to the resume checker.

Repeats for every permutation of (temperature, defence, injectionType, injectionLocation, prompt_name)

Extracts the fields from the final model output and saves all extracted fields and the prompt metadata into a csv file.

By default, files are read from prompt_file and written to ./output/finalLLaDAResults.csv. The model will prompt the user to select the model to be used for inference (llama | llada).

Can also take in command line arguments in the format:  

python main.py [model selection]  [output file] [input folder]

e.g.

python main.py llada testLLaDAResults.csv test



