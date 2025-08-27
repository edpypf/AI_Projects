# Example: Using LCEL to reproduce a "Basic Prompting" scenario
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama
import gradio as gr

# 2. Define the prompt
prompt = PromptTemplate.from_template(
    "What is the capital of {topic}?"
)

# 3. Define the model
model = ChatOllama(model="llama3:instruct")  # Using Ollama

# 4. Chain the components together using LCEL
chain = (
    # LCEL syntax: use the pipe operator | to connect each step
    {"topic": RunnablePassthrough()}  # Accept user input
    | prompt                          # Transform it into a prompt message
    | model                           # Call the model
    | StrOutputParser()               # Parse the output as a string
)

# 5. Execute
def ask_ai(quesiton):
    result = chain.invoke({"Question":quesiton})
    return result

# 6. Output to Web Gui
# print("User prompt: 'What is the capital of Germany?'")
# print("Model answer:", result)

# 建立 Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# 🧠 Proxy AI Agent Interface")
    with gr.Row():
        input_box = gr.Textbox(label="please input your question")
        output_box = gr.Textbox(label="Model answer", lines=5)
    submit_btn = gr.Button("send")
    submit_btn.click(fn=ask_ai, inputs=input_box, outputs=output_box)

# 啟動
demo.launch()