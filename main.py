# use ollama and langchain to create clinical notes
# before running this, run 'ollama serve' on local machine


from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import Ollama
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from datetime import datetime
import sys, time

# selected_model="llama2:13b"
selected_model="mistral"
# selected_model="gemma:7b"
clinical_notes = "patient_notes.txt"
trial_notes_1 = "RCT1_raw_arm1.txt"
result = "result.txt"


def main():
    # the number of case notes to generate    
    num_cases = 4
    print(f" Creating Case Notes for {num_cases} patients")
    llm = Ollama(
    model=selected_model, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]), temperature=1
    )
    # tbd: A general instruction
    # llm("""I shall ask you to generate clinnical notes. Make each note about an unique patient with a different name and id.
    # Wait for further instructions before generating responses""")


    prompt = """You are a clinician specializing in epilepsy patients. You will generate a Clinical Note for an epilepsy patient. Each note should include
        the patient's name, id, date of visit, current medications, chief complaint, general observations, assessment and a followup plan
        Use the following format for the note:
        Patient Name:
        Patient Id:
        Date of Visit:
        Current Medications:
        Chief Complaint:
        General Observations:
        Assessment:
        Follow-up Plan:

        Make sure that patient names and ids are unique for each note you generate""" 
    # prompt_without_drug = """You will generate a Clinical Note for an epilepsy patient who does not use drug X. Each note should include
        # the patient's name, id, date of visit, current medications, chief complaint, general observations, assessment and a followup plan"""


    create_notes(llm, num_cases, prompt, clinical_notes)
    # create_notes(llm, num_cases, prompt_without_drug, fl_no_drug)
    print()
    print(f"Created notes at the file {clinical_notes}")

    print(" Lets ask questions about the generated notes based on related information")
    
    process_files(llm, [f"{clinical_notes}", f"{trial_notes_1}"], result)
 


def create_notes(llm, num_cases, prompt, filename):
    # llm = Ollama(
    #     model=selected_model, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
    # )

    with open(filename, 'a') as f:
        sys.stdout = f
        print()
        stamp = datetime.now()
        print("===========================================Clinical Note====================================")
        for i in range(num_cases):
            # prompt = """You will generate a Clinical Note for an epilepsy patient based on the usage of drug X; the note shall
            # include an assessment of whether there was progress. Each note should include
            # the patient's name, id, date of visit, current medications, chief complaint, general observations, assessment and a followup plan"""        
            llm(prompt)
            # print(res)
            print()
            print("========================================================================================")
            time.sleep(5) # it seems like more time ==> better results
        f.flush()
        sys.stdout = sys.__stdout__

def process_files(llm, fl_list, result):
    
    print(" Processing the files: ", fl_list)
    documents = []
    for fl in fl_list:
        loader = TextLoader(f"{fl}")
        documents.extend(loader.load())

    # loader = TextLoader(f"{filename}")
    # data = loader.load()

    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    all_splits = text_splitter.split_documents(documents)
    oembed = OllamaEmbeddings(base_url="http://localhost:11434", model="nomic-embed-text")
    vectorstore = Chroma.from_documents(documents=all_splits, embedding=oembed)

    question="""You are now an expert on evaluating clinical trials for epilipsey; tell me how well did 
    the investigational drug work given the data?"""

    qachain=RetrievalQA.from_chain_type(llm, retriever=vectorstore.as_retriever())
    
    with open(result, 'a') as f:
        sys.stdout = f
        print()
        stamp = datetime.now()
        print("===================================Trial Evaluation====================================")
  
        qachain.invoke({"query": question})
        print("=======================================================================================")

        f.flush()
        sys.stdout = sys.__stdout__




if __name__ == "__main__":
    main()
