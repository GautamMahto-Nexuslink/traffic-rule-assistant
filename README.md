# traffic-rule-assistant

How to Run
Step 1 — Run the full pipeline (extract + chunk + embed):


export GROQ_API_KEY="your_key_here"
python main.py pipeline
Step 2 — Start the chatbot:


python main.py chat
If GROQ_API_KEY is not set, it will prompt you for it interactively.

The install is still running (downloading torch + CUDA packages ~2GB). Once it finishes, you can run python main.py pipeline immediately. Your Groq API key from console.groq.com is needed only for the chat step.