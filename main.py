# use ollama and langchain to create clinical notes
# before running this, run 'ollama serve' on local machine


from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import Ollama
import sys, time

# selected_model="llama2:13b"
selected_model="mistral"
# selected_model="gemma:7b"
fl_drug = "drug_applied.txt"
fl_no_drug = "drug_not_applied.txt"


def main():
    # the number of case notes to generate    
    num_cases = 1
    print(f" Creating Case Notes for {num_cases} patients")
    llm = Ollama(
    model=selected_model, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]), temperature=1
    )
    prompt_with_drug = """You will generate a Clinical Note for an epilepsy patient based on the usage of drug X. Each note should include
        the patient's name, id, date of visit, current medications, chief complaint, general observations, assessment and a followup plan""" 
    # prompt_without_drug = """You will generate a Clinical Note for an epilepsy patient who does not use drug X. Each note should include
        # the patient's name, id, date of visit, current medications, chief complaint, general observations, assessment and a followup plan"""

    # tbd: A general instruction
    # llm("""I shall ask you to generate clinnical notes. Make each note about an unique patient with a different name and id.
    # Wait for further instructions before generating responses""")

    create_notes(llm, num_cases, prompt_with_drug, fl_drug)
    # create_notes(llm, num_cases, prompt_without_drug, fl_no_drug)
    print()
    print(f"Created notes at the file {fl_drug}")
 


def create_notes(llm, num_cases, prompt, filename):
    # llm = Ollama(
    #     model=selected_model, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
    # )

    with open(filename, 'w') as f:
        sys.stdout = f
        print()
        print("=========================================================================================================")
        print("===================================Notes for Patients that got Drug X====================================")
        for i in range(num_cases):
            # prompt = """You will generate a Clinical Note for an epilepsy patient based on the usage of drug X; the note shall
            # include an assessment of whether there was progress. Each note should include
            # the patient's name, id, date of visit, current medications, chief complaint, general observations, assessment and a followup plan"""        
            res = llm(prompt)
            print(res)
            print("=========================================================================================================")
            time.sleep(1)
        f.flush()
        sys.stdout = sys.__stdout__



if __name__ == "__main__":
    main()
