import numpy as np
from flask import Flask, render_template
from flask import request
import pp 

app = Flask(__name__)

def sigmoid(z):

    """
    Calcula a função sigmoidal  
    """

    z = 1/(1+np.exp(-z))
    
    return z

def predicao(Theta1, Theta2, Xval):

    m = Xval.shape[0] # número de amostras
    num_labels = Theta2.shape[0]
    
    p = np.zeros(m)

    a1 = np.hstack( [np.ones([m,1]), Xval] )
    h1 = sigmoid( np.dot(a1, Theta1.T) )

    a2 = np.hstack( [np.ones([m,1]), h1] ) 
    h2 = sigmoid( np.dot(a2, Theta2.T) )
    
    Ypred = np.argmax(h2,axis=1)
            
    return Ypred


@app.route('/', methods=['GET', 'POST'])
def review():

	if request.method == 'POST':


		category = request.form.get('category')

		review_text = request.form.get('review')

		# # #try:
			
		# with open(category+'/vocabulary.txt', 'r') as in_file:
		# 	vocabulary = in_file.read().split('\n')

		# vocabulary = vocabulary[:-1] #remove last \n

		# for i in range(len(vocabulary)):
		# 	if vocabulary[i][0] == "(" and vocabulary[i][-1] == ")":
		# 		elements = vocabulary[i][1:-1].split(",")
		# 		vocabulary[i] = tuple(elements)

		# Theta1 = np.load(category+'/Theta1.npy')

		# Theta2 = np.load(category+'/Theta2.npy')

		# sample = pp.bow(vocabulary, review_text)

		# classe = predicao(Theta1, Theta2, np.asmatrix(sample))

		# print("Classe = " + str(classe))

		print("AAAAAAAAAAAAAAAAAAAA")


		return render_template('review.html', answer=True, classe=0)

		# #except:

		# #	pass

		
	return render_template('review.html', answer=False)	

if __name__ == '__main__':
    app.run(debug=True)