#!/usr/bin/env python3
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All, LlamaCpp
import os
import argparse
from timeit import default_timer as timer
load_dotenv()

embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get('PERSIST_DIRECTORY')

model_type = os.environ.get('MODEL_TYPE')
model_path1 = os.environ.get('MODEL_PATH1')
model_path2 = os.environ.get('MODEL_PATH2')
model_n_ctx = os.environ.get('MODEL_N_CTX')
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS',4))

from constants import CHROMA_SETTINGS

def main():
    # Parse the command line arguments
    args = parse_arguments()
    if args.m==1:
        model_type="GPT4All"
    else:
        model_type="LlamaCpp"
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
    # activate/deactivate the streaming StdOut callback for LLMs
    callbacks = [] if args.mute_stream else [StreamingStdOutCallbackHandler()]
    # Prepare the LLM
    match model_type:
        case "GPT4All":
            llm = GPT4All(model=model_path1, n_ctx=model_n_ctx, backend='gptj', callbacks=callbacks, verbose=False)
        case "LlamaCpp":
            llm = LlamaCpp(model_path=model_path2, n_ctx=model_n_ctx, callbacks=callbacks, verbose=False)
        case _default:
            print(f"Model {model_type} not supported!")
            exit()
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents= not args.hide_source)
    # Interactive questions and answers
    while True:
        query = input("\nEnter a query: ")
        if query == "exit":
            break
        
        start = timer()
        # Get the answer from the chain
        res = qa(query)
        answer, docs = res['result'], [] if args.hide_source else res['source_documents']
        end = timer()
        
        # Print the result
        print("\n\n> Question:")
        print(query)
        print("\n> Answer:")
        print(answer)
        print("Process Time:",(end - start)," Secs")   
                 
        # for document in docs:
        #     print("\n> " + document.metadata["source"] + ":")
        #     print(document.page_content)

def parse_arguments():
    parser = argparse.ArgumentParser(description='privateGPT: Ask questions to your documents without an internet connection, '
                                                 'using the power of LLMs.')
    parser.add_argument("--hide-source", "-S", action='store_true',
                        help='Use this flag to disable printing of source documents used for answers.')

    parser.add_argument("--mute-stream", "-M",
                        action='store_true',
                        help='Use this flag to disable the streaming StdOut callback for LLMs.')
    parser.add_argument("--m",type=int, required=False,help='choose model 1 or 2.')


    return parser.parse_args()


if __name__ == "__main__":
    main()
