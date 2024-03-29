# use ollama and langchain to create clinical notes
# before running this, run 'ollama serve' on local machine


from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import Ollama


inp = input("Please enter the number of cases to generate: ")

try:
    num_cases = int(inp)
    # selected_model="llama2"
    selected_model="mistral"

    llm = Ollama(
        model=selected_model, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
    )

    print("=========================================================================================================")
    print("===================================Notes for Patients that got Drug X====================================")
    
    prompt = """You will generate a set of clinical notes for epilepsy patients based on the usage of drug X; each note will 
    include an assessment of whether there was progress. Create {num_cases} sets of notes and use around 10 sentences per note."""

    print("prompt used for drug use : ", f"{prompt}")

    llm(prompt)

    print("==========================================================================================================")
    print("===================================Notes for Patients that did NOT get Drug X=============================")


    prompt = """You will generate a set of clinical notes for epilepsy patients who have not been given drug X and are on other treatments; each note will 
    include an assessment of whether there was progress. Create {num_cases} sets of notes and use around 10 sentences per note."""

    print("prompt used for NO drug use : ", f"{prompt}")

    llm(prompt)

except ValueError:
        print("Invalid input. Please enter a valid integer.")

