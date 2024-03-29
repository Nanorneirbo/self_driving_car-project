import numpy as np
import matplotlib.pyplot as plt

def draw(x1,x2):
	line = plt.plot(x1,x2,'-')
	plt.pause(0.0001)
	line[0].remove()
	
def sigmoid(score):
	#work out probability of them being in positive region)
	return 1/(1 + np.exp(-score))

	
def calculate_error(line_parameters, points, y):
	#figure out the probability error using cross entropy (ln method)
	m = points.shape[0]
	probabilities = sigmoid(points*line_parameters)
	cross_entropy = -(np.log(probabilities).T * y + np.log(1-probabilities).T*(1-y))/m
	return cross_entropy
	
def gradient_descent(line_parameters, points, y, alpha):
	m = points.shape[0]
	for i in range(500):
		probabilities = sigmoid(points*line_parameters)
		gradient = points.T * (probabilities -y)*(alpha/m)
		# make the changes small so mult by alpha
		line_parameters = line_parameters - gradient
		w1 = line_parameters.item(0)
		w2 = line_parameters.item(1)
		b = line_parameters.item(2)
		x1 = np.array([points[:,0].min(), points[:,0].max()])
		x2 = -b/w2 +x1 * (-w1/w2)
		draw(x1,x2)
		print(gradient)
		
n_pts = 100
np.random.seed(0)
bias = np.ones(n_pts)
# create a distribution with std dev 2 
random_x1_values = np.random.normal(10, 2, n_pts)
random_x2_values = np.random.normal(12, 2, n_pts)

# transpose 
top_region = np.array([random_x1_values,random_x2_values,bias]).T
bottom_region = np.array([np.random.normal(5, 2, n_pts), np.random.normal(6,2,n_pts),bias]).T 

all_points = np.vstack((top_region, bottom_region))

line_parameters = np.matrix([np.zeros(3)]).T



#probabilities = sigmoid(linear_comnbination)
y = np.array([np.zeros(n_pts), np.ones(n_pts)]).reshape(n_pts*2,1)


#print((calculate_error(line_parameters, all_points, y))) 


fig, ax = plt.subplots(figsize=(4,4))
ax.scatter(top_region[:, 0], top_region[:, 1], color = 'r')
ax.scatter(bottom_region[:, 0], bottom_region[:, 1], color = 'b')
print(gradient_descent(line_parameters, all_points, y, 0.06))
plt.show()
