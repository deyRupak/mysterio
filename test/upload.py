from flask import Flask,render_template,flash,request,redirect
from flask import request 
import os


UPLOAD_FOLDER = 'uploads'


app = Flask(__name__)  

app.config['UPLOAD_PATH'] = 'static/uploadd'


@app.route('/success', methods = ['GET'])  
def upload():  
    return render_template("file_upload_form.html",name='',c='')  
     
@app.route('/success', methods = ['POST'])  
def success():
    uploaded_files = request.files.getlist("file[]")
    x=[1,2,3]
    for f in uploaded_files:
        f.save(os.path.join(app.config['UPLOAD_PATH'], f.filename))

        print (f)
    return render_template("file_upload_form.html", name=uploaded_files, c=x)

# def success():  
#     if request.method == 'POST':  
#         images = request.files.to_dict() #convert multidict to dict
#         for image in images:     #image will be the key 
#             print(images[image])        #this line will print value for the image key
#             file_name = images[image].filename
#             images[image].save(os.path.join(app.config['UPLOAD_FOLDER'], images[image].filename))
#            return render_template("success.html", name = images[image].filename)  
      
if __name__ == '__main__':  
    app.run(debug = True)  