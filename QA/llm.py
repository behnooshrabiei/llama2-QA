from langchain.llms import CTransformers

# Local CTransformers wrapper for Llama-2-13B-Chat
llm = CTransformers(model='../models/llama-2-13b-chat.ggmlv3.q4_K_M.bin', # Location of downloaded GGML model
                    model_type='llama', # Model type Llama
                    config={'max_new_tokens': 256,
                            'temperature': 0.01})