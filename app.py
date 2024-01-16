from flask import Flask, render_template, request
app = Flask(__name__)

@app.route('/')

def home():
    return render_template('index.html')


@app.route("/predictdata", methods=['GET','POST'])

def predict_data():
    if request.method =="GET":
        return render_template('home.html')
    else:
        pass
if __name__=="__main__":
    app.run()