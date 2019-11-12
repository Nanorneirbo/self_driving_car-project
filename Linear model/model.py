import numpy as np
import matplotlib.pyplot as plt

def draw(x1,x2):
	line = plt.plot(x1,x2)
	
def sigmoid(score):
	#work out probability of them being in positive region)
	return 1/(1 + np.exp(-score))

	
def calculate_error(line_parameters, points, y):
	#figure out the probability error using cross entropy (ln method)
	m = points.shape[0]
	probabilities = sigmoid(points*line_parameters)
	cross_entropy = -(np.log(probabilities).T * y + np.log(1-probabilities).T*(1-y))/m
	return cross_entropy
n_pts = 100
np.random.seed(0)
bias = np.ones(n_pts)
# create a distribution with std dev 2 
random_x1_values = np.random.normal(10, 2, n_pts)
random_x2_values = np.random.normal(12, 2, n_pts)

# transpose 
top_region = np.array([random_x1_values,random_x2_values,bias]).T
bottom_region = np.array([np.random.normal(5, 2, n_pts), np.random.normal(6,2,n_pts),bias]).T

# double bracket to make it one argument not two 

all_points = np.vstack((top_region, bottom_region))

# first guess line - weights and bias
# will try to use gradient descent later
#w1 = -0.3
#w2 = -0.4
#bias = 3




# make it a transpose so we can find probability using matrix multiplication
line_parameters = np.matrix([w1,w2,bias]).T
# furthest left point
# w1x1 +w2x2 +b =0 so x2 - b/w2 + x1*(-w1/w2)
# x1 is an array of the two horizontal points 
x1 = np.array([bottom_region[:, 0].min(), top_region[:,0].max()])
# x2 is computing the element values off the other array 
x2 = - bias/w2 + x1 *(-w1/w2)

#linear_comnbination = all_points*line_parameters


#probabilities = sigmoid(linear_comnbination)
y = np.array([np.zeros(n_pts), np.ones(n_pts)]).reshape(n_pts*2,1)
#print(y)

print((calculate_error(line_parameters, all_points, y))) 

#PRINT TESTS
#print(probabilities)
#print(linear_comnbination)
#print(all_points.shape)
#print(line_parameters.shape)
#print (x1,x2)
#print (all_points)
#print(top_region)

fig, ax = plt.subplots(figsize=(4,4))
ax.scatter(top_region[:, 0], top_region[:, 1], color = 'r')
ax.scatter(bottom_region[:, 0], bottom_region[:, 1], color = 'b')
draw(x1,x2)
plt.show()
