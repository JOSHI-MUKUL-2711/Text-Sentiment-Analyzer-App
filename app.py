from flask import Flask, render_template, request
import pickle

app = Flask(__name__)



@app.route("/")
@app.route("/home")
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    
    text = request.form.values()

    vec1 = pickle.load(open('vec.pkl','rb'))
    model = pickle.load(open('model_v1.pkl','rb'))
    prediction = model.predict(vec1.transform(text))

    if prediction == 1:
        return render_template('home.html',text_prediction = 'The text shows a positive sentiment.')
    else:
        return render_template('home.html', text_prediction = 'The text shows a negative sentiment.')

@app.route("/about")
def about():
    return render_template('about.html',title = 'About')

if __name__ == '__main__':
    app.run(debug=True)
