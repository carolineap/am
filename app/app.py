import scipy
import numpy as np
from flask import Flask, render_template
from flask import request
import pp 
import svmutil
from svmutil import svm_read_problem
from svmutil import svm_problem
from svmutil import svm_parameter
from svmutil import svm_train
from svmutil import svm_predict
from svmutil import svm_save_model
from svmutil import svm_load_model

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def review():

	if request.method == 'POST':

		review_text = request.form.get('review')

		with open('vocabulary.txt', 'r') as in_file:
			vocabulary = in_file.read().split('\n')

		model = svm_load_model('test.model')

		sample = pp.bow(vocabulary, review_text)

		#print(type(sample))

		#print(sample)

		classe = svm_predict([], sample, model, '-q')

		return render_template('review.html', classe=classe[0][0])

	return render_template('review.html')	

if __name__ == '__main__':
    app.run()