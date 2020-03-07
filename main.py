from flask import Flask, render_template, request
import pickle


app = Flask(__name__)


#def ValuePredictor(img_to_check):

    #path = Path("./model/")
    #learn = load_learner(path)
    #pred_class,pred_idx,outputs = learn.predict(img_to_check)

    #return str(pred_class)

@app.route("/")
def home():
    return render_template("index.html")


@app.route('/results')
def about():
    return render_template("results.html")


@app.route("/test")
def test():
    return render_template("takeTest.html")


@app.route('/results', methods=['POST'])
def handle_data():
    projectpath = request.form
    print(projectpath)
    return render_template("results.html", pred=projectpath)


if __name__ == "__main__":
    app.run(debug=True)
