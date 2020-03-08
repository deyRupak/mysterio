from flask import Flask, render_template, request
import pickle
import os,shutil
import pickle
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
# import useful libraries
import numpy as np # linear algebra
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
train_path = './intel-image-classification/seg_test/seg_test'
test_path = './intel-image-classification/seg_test/seg_test'
pred_path = 'static/upload'


transformer = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5],
                         [0.5, 0.5, 0.5])
])


model = torch.load("model/resnet.pkl")

UPLOAD_FOLDER = 'uploads'
train_loader = DataLoader(ImageFolder(train_path, transform=transformer),num_workers=8, batch_size=200, shuffle=True)

error = nn.CrossEntropyLoss()
opt = optim.Adam(model.parameters())
a=[]
b=[]
def train(model, train_loader, n_epochs,lr,momentum,batch_size):
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
			a.append(epoch)
			b.append(epoch_loss)
			fig,ax= plt.subplots()
			ax.plot(a,b)
			ax.set(xlabel="Training Loss",ylabel="Epoch")
			fig.savefig("static/loss2.png")

			#print(a,b)
			#print(f'epoch: {epoch} \t Train Loss: {epoch_loss:.4g}')
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
					pass
            
            
	return ds
	 
            #plt.show()
error = nn.CrossEntropyLoss()
opt = optim.Adam(model.parameters())


            
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

# @app.route('/results')
# def about():
#     return render_template("results.html")


@app.route("/test")
def test():

    return render_template("takeTest.html",name='')


     
@app.route('/test', methods=['POST']) 
def dec():
	model = torch.load("model/resnet.pkl")
	if request.form == {}:
		shutil.rmtree("static/upload")
		os.makedirs("static/upload")
		uploaded_files = request.files.getlist("file[]")
		for f in uploaded_files:
			f.save(os.path.join(app.config['UPLOAD_PATH'], f.filename))
		length=len(uploaded_files)
		x=predict(model,pred_path,length)
		print(x)
		return render_template("results.html",name=x)
	else:
		projectpath = request.form
		print(projectpath)
		return render_template("results.html", pred=projectpath)



@app.route("/frames")
def frame():
    return render_template("frames.html")


@app.route('/test2', methods=['POST'])
def edit():
	a=[]
	b=[]
	req = request.form
	dest_path='static/edit'
	x=request.form
	for val in x:	
		try:
			shutil.move('./'+val,dest_path+'/'+x[val])
		except:
			pass
	dest_path='static/edit'
	train_path = dest_path
	train_loader = DataLoader(
    ImageFolder(train_path, transform=transformer),
    num_workers=8, batch_size=200, shuffle=True
	)
	print("edit function")
	print(req)
	train(model,train_loader,int(req['epoch']),float(req['lr']),float(req['moment']),int(req['batch']))
	#fig,ax= plt.subplot()
	#ax.plot(a,b)
	#ax.set(xlabel="Loss",ylabel="Epoch")
	#fig.savefig("testloss.png")
	return render_template("results.html")



if __name__ == "__main__":
    app.run(debug=True)
