from flask import Flask, render_template, request
import pickle
import os,shutil
import pickle
from PIL import Image

# import useful libraries
import numpy as np # linear algebra
from matplotlib import pyplot as plt
from PIL import Image
from glob import glob
import os

# import pytorch modules
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import ImageFolder


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# define paths
train_path = './intel-image-classification/seg_train/seg_train'
test_path = './intel-image-classification/seg_test/seg_test'
pred_path = 'static/upload'


transformer = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5],
                         [0.5, 0.5, 0.5])
])

train_loader = DataLoader(
    ImageFolder(train_path, transform=transformer),
    num_workers=8, batch_size=200, shuffle=True
)
model = torch.load("model/resnet.pkl")

UPLOAD_FOLDER = 'uploads'





classes = train_loader.dataset.class_to_idx

def predict(model, path, sample_size):
	i=0
	ds={}
	for file in glob(os.path.join(path, '*.jpg'))[:sample_size]:
		with Image.open(file) as f:
			img = transformer(f).unsqueeze(0)
			with torch.no_grad():
				out = model(img.to(device)).cpu().numpy()
			for key, value in classes.items():
				if value == np.argmax(out):
					ds.update({f.filename :key})
					i+=1
				else :
					print(out)
			plt.imshow(np.array(f))
            
            
	return ds
	 
            #plt.show()
error = nn.CrossEntropyLoss()
opt = optim.Adam(model.parameters())

def train(model, train_loader, n_epochs=100):
    model = model.to(device)
    model.train()
    
    for epoch in range(n_epochs):
        epoch_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = error(out, y)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()        
        if epoch % int(0.1*n_epochs) == 0:
            print(f'epoch: {epoch} \t Train Loss: {epoch_loss:.4g}')
            
app = Flask(__name__)
app.config['UPLOAD_PATH'] = 'static/upload'


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
    return render_template("takeTest.html",name='')



     
@app.route('/test', methods=['POST'])  
def success():
	shutil.rmtree("static/upload")
	os.makedirs("static/upload")
	uploaded_files = request.files.getlist("file[]")
	for f in uploaded_files:
		f.save(os.path.join(app.config['UPLOAD_PATH'], f.filename))
	length=len(uploaded_files)
	x=predict(model,pred_path,length)
	print(x)
	return render_template("results.html",name=x)



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
