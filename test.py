import pandas as pd
import re
from prompt_model import extract_fields
import load_llm

def main():
    new_df = pd.DataFrame(columns=["timestamp", "model", "prompt", "prompt_hash", "output", "temperature", "max_new_tokens", "execution_time" , "rating", "recommendation", "reason"])
    results = pd.read_csv("resumeResults.csv")
    
    for _ , row in results.iterrows():
        output = extract_fields(row)
        new_df.loc[len(new_df)] = output
        #print(output)
    load_llm.save_results_csv(new_df, "appendedResults.csv")
if __name__ == '__main__':
    main()
