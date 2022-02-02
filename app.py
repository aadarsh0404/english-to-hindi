from flask import Flask, request, render_template
import jsonify
import torch
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

app = Flask(__name__)

model = torch.load('model.h5')
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-one-to-many-mmt", src_lang="en_XX")

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """ -------------- READING DATA -------------- """
    if request.method=='POST':
        data = request.form['English']
        
        model_inputs = tokenizer(data, return_tensors="pt")
        generated_tokens = model.generate(
            **model_inputs,
            forced_bos_token_id=tokenizer.lang_code_to_id["hi_IN"]
            )
        
        """ -------------- PREDICTION -------------- """
        translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        
        return render_template('index.html', prediction_text=translation[0])
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run()
