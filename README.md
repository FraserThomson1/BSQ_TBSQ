# READ ME

## About
In this project, we consider a common problem in a distributed and federated setting 
where n clients are each transmitting a d dimensional vector with real valued entries 
and each vector is encoded by b bits, where b has a lower bound of d and upper bound 
of 32d. The encoding must be computed in such a way that the recipient is able to 
approximate the encoded mean vector of the received vectors without decoding the 
client vectors. We derive computationally efficient algorithms that achieve similar 
accuracy to existing compression algorithms. We also evaluate the algorithms by 
simulating them in distributed learning environments using several machine learning 
models and datasets.



## Set up
To set up the code environment, install all dependencies in the 
requirements.txt file. The file “stoch_quant.py” contains the code for 
compression, decompression, extreme value separation used by BSQ and 
TBSQ. The file “cnn_distributed.py” contains the code for simulating any of the 
algorithms on a selection of neural networks and datasets and includes code for 
using BSQ and TBSQ in a distributed learning setting. The 
“distribution_data_simulation.py” file contains code used for testing the 
algorithms’ speed and accuracy on data generated from specified distribution.
