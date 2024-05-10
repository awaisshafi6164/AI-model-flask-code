from flask import Flask, request

app = Flask(__name__)

@app.route("/")
def index():
    input_text = request.args.get("text")
    if input_text:
        return f"You entered: {input_text}"
    else:
        return "No text input provided"

app.run(host="0.0.0.0", port=5001)
