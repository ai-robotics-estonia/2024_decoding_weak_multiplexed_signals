# Detecting Fiber Generating Signal Using Machine Learning

## Summary
| Company Name | GSCAN ([https://www.gscan.eu/](https://www.gscan.eu/)) |
| :--- | :--- |
| Development Team Lead Name | [Kallol Roy]([https://profile.link](https://kallolroy.me/)) |
| Development Team Lead E-mail | [kallol.roy@ut.ee](mailto:kallol.roy@ut.ee) |



# Description
## Objectives of the Demonstration Project
The challenge of the project is to improve the resolution of the scanning technology based on Cosmic Muon Rays. We developed a new approach to the detection of muon hits from noisy measurements and the recovery of missing muon hits. This approach aims at fast detection of muon rays and generating predictions for missing data points by utilizing contemporary AI technologies, including advanced pattern recognition capabilities of neural networks. The muon detection algorithm implementation combines MATLAB scripts, which are used as generators of ground truth data for learning a neural network-based muon detection system implemented in Python. 

## Activities and Results of the Demonstration Project
### Challenge
The algorithm presented in the report reliably extracts the expected number of muons passing through the system. The activities which lead to achieving high efficiency of the algorithm are the following:
- Preprocessing. The goal is to extract useful data from the noisy experimental data
- The development of a list-based quantization procedure, which takes into account the statistical model of the sensors used in the system.
- Generation of clouds of ray candidates and using fast approximation techniques for generating ground truth measurements.
- Statistical analysis of the muon ray detection and formalization of the detection algorithm as an optimization problem
- Comparative analysis of AI optimization algorithms, including surrogate optimization, genetic algorithm, and particle swarm algorithm
- Developing and implementing of neural network-based machine learning approximation methodology using the PyTorch framework. 


### Data Sources
The research input data is a large database of measurements produced by the muon tomography sensors. 

### AI Technologies
The demultiplexing task with a linear search algorithm is computationally expensive. The use of artificial intelligence (AI) is to make the search algorithm faster by constraining the line search. Internally our machine learning model solves the problem as a linear programming (LP) method with constraints coming from muon physics. The solving of the linear programming (LP) is done by convolutional filters of our model. The convolutional filters of our AI model are trained to solve the problem in linear programming. The objective function is the detection of the most probable area of the muon hits as a maximization criterion. Our AI model thus gives the prediction of the (bounding box) of the muon trajectory. The muon trajectory hits are at the corners of the bounding boxes (polytopes) if solved by linear programming. As our multiplexing tasks are approximated and solved internally as an LP method, the solutions lie close to the corners of the bounding box. The corners bounding box coordinates are given as a solution to the classical linear search method. The classical linear search is done on the constrained area of the bounding box and not on the whole muon detector plates. This fastens the classical linear search method augmented by artificial intelligence. 


### Technological Results
The algorithm presented in the report reliably extracts the expected number of muons
passing through the system. Its complexity allows its almost real-time implementation in
Matlab. The main results which demonstrate the high efficiency of the suggested
approach:
- The number of detected muon rays is very close to the theoretical limit
determined by the empirically estimated intensity of the cosmic rays passing
through the tomograph.
- The mathematical analysis of the detection model allows for estimating the
probability of generating a false ray and the probability of losing a muon ray.
- Testing of the algorithm shows that the parameters of the system well match the
estimated efficiency.

### Technical Architecture
The main computational load is concentrated in “findline” routines where the extracted measured data are approximated by muon rays. For these routines, AI and ML approaches allow for improving muon detection performances.

![image](https://github.com/user-attachments/assets/ad69d4d2-8df2-4f44-a105-547de0d7612c)


### User Interface 
The software package developed in the project includes: 
- Standalone executable program for producing a file of the muon rays coordinates from the given input database of the tomograph measurements. 
- Matlab script for generating output data for use as a ground truth for AI and ML-based technologies. P
- Python software: The AI algorithm is implemented in Python language and uses Pytorch and Tensorflow environment. Additionally, we have used a Fastai library from Meta. The data preprocessing is done through scikit-learn packages. 

All programs during the data processing output informative statistical data demonstrating the properties of the particular data stream and the efficiency of its analysis. The goals of intermediate data output are: 
- Control and validation. 
- Search for scenarios of potential inefficiency. 
- Finding additional possibilities for optimization. 



### Future Potential of the Technical Solution
The developed technology can be used for scanning the objects in the tomographs which use other physical principles.
New horizons of the developed technology:
- Combining the detection algorithm with the muon tomography image-generating
algorithms.
- The technology can be easily scaled to the cosmic rays tomographs with other
geometric parameters and different sensors.
- This detection algorithm along with image reconstruction can be used for civil
engineering construction, medical imaging and other related technologies.

### Lessons Learned
The initial research problem was extended during the research activity. In addition to the muon detection procedure, a thorough mathematical analysis of detection reliability was conducted, validating the algorithm's efficiency and correctness. 


