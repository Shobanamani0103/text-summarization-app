from transformers import pipeline

# Load summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Sample text
text = """Your long article or document goes here..."""

# Generate summary
summary = summarizer(text, max_length=130, min_length=30, do_sample=False)
print(summary[0]['summary_text'])

# Optional: Gradio UI
import gradio as gr

def summarize_text(text):
    return summarizer(text, max_length=130, min_length=30, do_sample=False)[0]['summary_text']

gr.Interface(fn=summarize_text, inputs="text", outputs="text").launch()
