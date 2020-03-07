from flask import Flask, render_template, request
import pickle
import os
import pickle
from PIL import Image


UPLOAD_FOLDER = 'uploads'


app = Flask(__name__)
app.config['UPLOAD_PATH'] = 'static/upload'


#def ValuePredictor(img_to_check):

    #path = Path("./model/")
    #learn = load_learner(path)
    #pred_class,pred_idx,outputs = learn.predict(img_to_check)

    #return str(pred_class)



def ValuePredictor(folder): 
    #to_predict = np.array(to_predict_list).reshape(1, 12) 
    loaded_model = pickle.load(open("model/resnet.pkl", "rb")) 
    result = loaded_model.predict(folder) 
    #return result[0] 





@app.route("/")
def home():
    return render_template("index.html")


@app.route('/results')
def about():
    return render_template("results.html")


@app.route("/test")
def test():
    return render_template("takeTest.html",result="")



     
@app.route('/test', methods=['POST'])  
def success():
    uploaded_files = request.files.getlist("file[]")
    # x=[1,2,3]
    for f in uploaded_files:
        f.save(os.path.join(app.config['UPLOAD_PATH'], f.filename))

        #print (f)

        img=Image.open(open("static/upload/aisehi.png", 'rb'))    
        result = ValuePredictor(img)
        print(result)

        print(23)

    return render_template("takeTest.html",result=result)



@app.route("/frames")
def frame():
    return render_template("frames.html")


@app.route('/results', methods=['POST'])
def handle_data():
    projectpath = request.form
    print(projectpath)
    return render_template("results.html", pred=projectpath)


if __name__ == "__main__":
    app.run(debug=True)
