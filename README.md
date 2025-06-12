# Detecting Fiber Generating Signal Using Machine Learning

## Summary
| Company Name | GSCAN ([https://www.gscan.eu/](https://www.gscan.eu/)) |
| :--- | :--- |
| Development Team Lead Name | [Kallol Roy]([https://profile.link](https://kallolroy.me/)) |
| Development Team Lead E-mail | [kallol.roy@ut.ee](mailto:kallol.roy@ut.ee) |
| Duration of the Demonstration Project | month/year-month/year |
| Final Report | [Example_report.pdf](https://github.com/ai-robotics-estonia/_project_template_/files/13800685/IC-One-Page-Project-Status-Report-10673_PDF.pdf) |

### Each project has an alternative for documentation
1. Fill in the [description](#description) directly in the README below *OR*;
2. make a [custom agreement with the AIRE team](#custom-agreement-with-the-AIRE-team).

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
*Please describe which data was used for the technological solution.*  
- [Source 1],
- [Source 2],
- etc... .

### AI Technologies
*Please describe and justify the use of selected AI technologies.*
- [AI technology 1],
- [AI technology 2],
- etc... .

### Technological Results
*Please describe the results of testing and validating the technological solution.*

Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.

### Technical Architecture
*Please describe the technical architecture (e.g, presented graphically, where the technical solution integration with the existing system can also be seen).*
- [Component 1],
- [Component 2], 
- etc... .

![backend-architecture](https://github.com/ai-robotics-estonia/_project_template_/assets/15941300/6d405b21-3454-4bd3-9de5-d4daad7ac5b7)


### User Interface 
*Please describe the details about the user interface(i.e, how does the client 'see' the technical result, whether a separate user interface was developed, command line script was developed, was it validated as an experiment, can the results be seen in ERP or are they integrated into work process)*

Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.

### Future Potential of the Technical Solution
*Please describe the potential areas for future use of the technical solution.*
- [Use case 1],
- [Use case 2],
- etc... .

### Lessons Learned
*Please describe the lessons learned (i.e. assessment whether the technological solution actually solved the initial challenge).*

Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.

# Custom agreement with the AIRE team
*If you have a unique project or specific requirements that don't fit neatly into the Docker file or description template options, we welcome custom agreements with our AIRE team. This option allows flexibility in collaborating with us to ensure your project's needs are met effectively.*

*To explore this option, please contact our demonstration projects service manager via katre.eljas@taltech.ee with the subject line "Demonstration Project Custom Agreement Request - [Your Project Name]." In your email, briefly describe your project and your specific documentation or collaboration needs. Our team will promptly respond to initiate a conversation about tailoring a solution that aligns with your project goals.*
