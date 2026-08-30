import random
import json
import requests
import gradio as gr
from transformers import pipeline
import openai
import os
import time
from langchain.document_loaders import OnlinePDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain


class chatgpt:

      def __init__(self):
        self.conversation = [{"role": "system", "content": "You are a friendly assistant who uses casual language and humor, in your conversations you often use emojis to reflect your mood."},]
        self.apikey="api_key"
        self.chat_model="gpt-3.5-turbo"
        self.modelPromt = "The Series 9 9kg QuickDrive™ Washing Machine WW90T986DSX is a high-performance appliance with a washing capacity of 9.0 kg. It has been rated with an energy efficiency class of A, ensuring energy-saving operations. The machine utilizes Bubble technology, generating bubbles for effective cleaning of clothes. It also features VRT (Vibration Reduction Technology), which minimizes vibration and noise during operation, providing a quieter experience. With dimensions of 600 x 850 x 600 mm, it offers a compact and space-efficient design. The net weight of the washing machine is 79 kg, indicating its sturdy and durable construction."
        self.modelThreshold = 90
        self.chat_temperature = 0.5
        self.chat_top_p = 1
        self.chat_choices_num = 1
        self.chat_stream = False
        self.chat_max_tokens = 1
        self.chat_presence_penalty = 0.1
        self.chatfrequency_penalty = 0.5
        self.system_message = "You are a friendly assistant who uses casual language and humor, in your conversations you often use emojis to reflect your mood."
        self.systemmassistant="null"
        self.langchaincontrol=False

      def systemUpdate(self,api_key,systemm,systemmassistant,chat_model,chat_temperature,chat_top_p,chat_choices_num,chat_stream,chat_max_tokens,chat_presence_penalty,chatfrequency_penalty,modelPromt,modelThreshold):
        self.apikey=api_key
        self.chat_model=chat_model
        self.modelPromt = modelPromt
        self.modelThreshold = modelThreshold
        self.chat_temperature = chat_temperature
        self.chat_top_p = chat_top_p
        self.chat_choices_num = chat_choices_num
        self.chat_stream = chat_stream
        self.chat_max_tokens = chat_max_tokens
        self.chat_presence_penalty = chat_presence_penalty
        self.chatfrequency_penalty = chatfrequency_penalty
        self.system_message = systemm
        self.systemmassistant=systemmassistant  


      def gptRequest(self, messages,apikey,system_message,systemmassistant,chat_model,chat_temperature,chat_max_tokens,chat_top_p,chat_choices_num,chat_stream,chat_presence_penalty,chatfrequency_penalty):
          openai.api_key =  self.apikey
          if self.systemmassistant=="null":
            conversation = [{"role": "system", "content": f"{self.system_message}"},{"role": "user", "content":f"{messages}"}]
          else:
            conversation = [{"role": "system", "content": f"{self.system_message}"},{"role": "assistant", "content": f"{systemmassistant}"},{"role": "user", "content":f"{messages}"}]
          response = openai.ChatCompletion.create(
          model=self.chat_model,  
          messages=conversation,
          temperature=float(self.chat_temperature),
          max_tokens=int(self.chat_max_tokens),
          top_p=float(self.chat_top_p),
          n=int(self.chat_choices_num),
          stream=bool(self.chat_stream),
          presence_penalty=float(self.chat_presence_penalty),
          frequency_penalty=float(self.chatfrequency_penalty),)
          return response.choices[0].message['content']


      def pdf_changes(self,pdf_doc):
          if pdf_doc is not None:
              self.langchaincontrol=True
              self.pdf=pdf_doc
          else:
              self.langchaincontrol=False

      def langchain(self,pdf_doc,question,history):
            os.environ['OPENAI_API_KEY'] = self.apikey
            loader = OnlinePDFLoader(pdf_doc.name)
            documents = loader.load()
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            texts = text_splitter.split_documents(documents)
            embeddings = OpenAIEmbeddings()
            db = Chroma.from_documents(texts, embeddings)
            retriever = db.as_retriever()
            qa = ConversationalRetrievalChain.from_llm(llm=OpenAI(temperature=float(self.chat_temperature)), retriever=retriever, return_source_documents=False)
            langchain_history = [(msg[1], self.history[i+1][1] if i+1 < len(self.history) else "") for i, msg in enumerate(self.history) if i % 2 == 0]
            result = qa({"question": question,"chat_history": langchain_history})
            return result["answer"]


 

      def answer_EcoQA(self,message,history,chatbot):
          self.history = history or []
          chat_history_tuples = []
          model_checkpoint = "TunahanGokcimen/Question-Answering-Bert-base-cased-squad2"
          question_answerer = pipeline("question-answering", model=model_checkpoint)
          context=str(self.modelPromt)
          question = str(message)
          bot_response= question_answerer(question=question, context=context)
          if float(bot_response["score"])> float(self.modelThreshold/100):
                answer = "This answer was given by the Q&A model: "+bot_response["answer"]+"\n\n"+"Model Confidence Score:"+str(bot_response["score"])+"\n\n"+"EcoQA Threshold:"+str(self.modelThreshold/100)
          elif self.langchaincontrol==True:
                langchainanswer=  self.langchain(self.pdf,question,chat_history_tuples)
                answer = "This answer was given by the Langchain "+"\n\n"+"EcoQA Threshold:"+str(self.modelThreshold/100)+"\n\n"+langchainanswer                 
          else:             
                answer= "This answer was given by the "+str(self.chat_model)+"\n\n"+"EcoQA Threshold:"+str(self.modelThreshold/100)+"\n\n"+self.gptRequest(question,self.apikey,self.system_message,systemmassistant,self.chat_model,self.chat_temperature,chat_max_tokens,self.chat_top_p,self.chat_choices_num,self.chat_stream,self.chat_presence_penalty,self.chatfrequency_penalty)          
          self.history.append((question, answer))
          return self.history, self.history 
        

      def system_message(self, systemm):
          self.conversation.clear()
          self.conversation = [{"role": "system", "content": f"{systemm}"},]

      def Clean(self):
          self.history.clear()
          self.conversation.clear()
          self.conversation = [{"role": "system", "content": "You are a friendly assistant who uses casual language and humor, in your conversations you often use emojis to reflect your mood."},]
          return self.history, self.history



theme = gr.themes.Default(primary_hue="blue",secondary_hue="red").set(
    button_primary_background_fill="*primary_200",
    button_primary_background_fill_hover="*primary_300",
    button_secondary_background_fill="*primary_200",
    button_secondary_background_fill_hover="*primary_300",
)
# User input
block = gr.Blocks(theme=gr.themes.Soft())
chatgpt = chatgpt()

with block:
    gr.Markdown("""<h1><center>EcoQA System</center></h1>
    """)
    with gr.Row():
        with gr.Column():
                api_key = gr.Textbox(type="password", label="Enter your OpenAI API key here", placeholder="sk-...0VYO",style={"font-family": "Arial", "font-weight": "bold", "font-size": "60px"},)
                systemm = gr.Textbox(label="ChatGpt User(System) Prompt", placeholder="You are a helpful assistant.")
                systemmassistant= gr.Textbox(label="ChatGpt User(Assistant) Prompt", placeholder="Tell me story")
                modelPromt = gr.Textbox(label="Q&A Model Prompt", placeholder="Hi, how are things ?")
                chat_model = gr.inputs.Dropdown(label="Model", choices=[
                "gpt-3.5-turbo", "gpt-3.5-turbo-0301", "gpt-3.5-turbo-16k",
                "gpt-4", "gpt-4-0314", "gpt-4-32k", "gpt-4-32k-0314"],
                default="gpt-3.5-turbo")
                chat_temperature = gr.Slider(label="temperature", value=0.5, minimum=0, maximum=2, visible=True)
                chat_top_p = gr.Slider(label="top_p", value=1, minimum=0, maximum=1, visible=True)
                chat_choices_num = gr.Slider(label="choices num(n)", value=1, minimum=1, maximum=20, visible=True)
                chat_stream = gr.Checkbox(label="stream", value=False, visible=True)
                chat_max_tokens = gr.Slider(label="max_tokens", value=512, minimum=180, maximum=4096, visible=True)
                chat_presence_penalty = gr.Slider(label="presence_penalty", value=0.1, minimum=-2, maximum=2, visible=True)
                chatfrequency_penalty = gr.Slider(label="frequency_penalty", value=0.5, minimum=-2, maximum=2, visible=True)
                modelThreshold  = gr.Slider(label="System Threshold", value=90, minimum=0, maximum=100, visible=True)
                submit = gr.Button("Submit Parameters",variant="primary")
                submit.click(chatgpt.systemUpdate, inputs=[api_key,systemm,systemmassistant,chat_model,chat_temperature,chat_top_p,chat_choices_num,chat_stream,chat_max_tokens,chat_presence_penalty,chatfrequency_penalty,modelPromt,modelThreshold])
        with gr.Column():
            pdf_doc = gr.File(label="Load a pdf", file_types=['.pdf'], type="file",height=100)
            load_pdf = gr.Button("Load pdf to langchain",variant="primary")
            load_pdf.click(chatgpt.pdf_changes, inputs=[pdf_doc])
            chatbot = gr.Chatbot(autoscale=True,height=650)
            message = gr.Textbox(label="Message", placeholder="Hi, how are things ?")
            state = gr.State()
            submit = gr.Button("Send message",variant="primary")
            submit.click(chatgpt.answer_EcoQA, inputs=[message,chatbot,state], outputs=[chatbot, state])
            clean = gr.Button("Clean",variant="secondary")
            clean.click(chatgpt.Clean, outputs=[chatbot, state])


block.launch(debug=True)