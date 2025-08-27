import gradio as gr

def greet(name):
    return f"Hello {name}!"

# Create a simple Gradio interface
demo = gr.Interface(fn=greet, inputs="text", outputs="text")

print(f"Gradio version: {gr.__version__}")
print("Gradio is working correctly!")

