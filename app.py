from flask import Flask, render_template, request, flash
import Pikel_File_Model as pfm
app = Flask(__name__)

@app.route('/')
def upload_form():
    return render_template('main.html')

@app.route('/predict', methods=['POST'])  
def predict():
   if 'image' in request.files:
      image = request.files['image']
      data = pfm.model_potato(image)
      return f"<h1>{data}</h1>"
   return 'No image found in the request!'
    


if __name__ == '__main__':
    app.run(debug=True)
