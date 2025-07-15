import numpy as np

j = int(input("Enter a Number (0-9): "))
step_function = lambda x: 1 if x >= 0 else 0   #returns 1 if input >=0 , otherwise returns 0 
#input is 6-bit binary number
training_data = [
    {'input': [1, 1, 0, 0, 0, 0], 'label': 1}, #even  48
    {'input': [1, 1, 0, 0, 0, 1], 'label': 0},  #odd  49
    {'input': [1, 1, 0, 0, 1, 0], 'label': 1}, #50
    {'input': [1, 1, 0, 1, 1, 1], 'label': 0},  #51
    {'input': [1, 1, 0, 1, 0, 0], 'label': 1},  #52
    {'input': [1, 1, 0, 1, 0, 1], 'label': 0},  #53
    {'input': [1, 1, 0, 1, 1, 0], 'label': 1},  #54
    {'input': [1, 1, 0, 1, 1, 1], 'label': 0},  #55
    {'input': [1, 1, 1, 0, 0, 0], 'label': 1},  #56
    {'input': [1, 1, 1, 0, 0, 1], 'label': 0},  #57
]
weights = np.array([0, 0, 0, 0, 0, 1])
for data in training_data:
    input = np.array(data['input'])
    label = data['label']
    output = step_function(np.dot(input, weights))
    error = label - output
    weights += input * error

input = np.array([int(x) for x in list('{0:06b}'.format(j)) ])  #convert j into 6-bit binary list
output = "odd" if step_function(np.dot(input, weights)) == 0 else "even"  
print(j, " is ", output)

