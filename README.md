# Malware Mutation System for Offensive Security

This project presents an innovative approach to designing a mutation algorithm for malware, leveraging advanced machine learning techniques to enhance its adaptability and evasion capabilities against anti-virus software and signature-based detection techniques.

## Objectives

- Develop a robust malware mutation algorithm using machine learning techniques such as Reinforcement Learning and Generative Models.
- Generate mutated variants of existing malware samples capable of bypassing static ML-based and signature-based detection methods employed by anti-virus solutions.
- Achieve superior computational efficiency and faster convergence rates compared to existing approaches.

## Key Innovations

1. **Diffusion-based Generative Adversarial Network (GAN) Model**:
  - Introduces an innovative approach that amplifies the randomization of noise in the malware feature vector.
  - Employs diffusion models in conjunction with the GAN model to effectively mitigate the mode collapse problem.
  - Achieves faster True Positive Rate (TPR) convergence and generates more diverse mutated samples.

![image](https://github.com/AryanShr/offenSec/assets/96382618/126c3743-e5d9-4bc5-9a04-bc7ecf7724b8)

2. **Enhanced Reinforcement Learning (RL) Algorithm**:
  - Implements changes to reduce exploration and increase exploitation using an inverse square function.
  - Incorporates a reward shaping methodology to distribute rewards based on various factors, enabling superior distribution of the action space during mutation.
  - Outperforms the original gym malware algorithm by Hein et al. (2022) in terms of action space distribution.

![image](https://github.com/AryanShr/offenSec/assets/96382618/05620f8a-011e-400d-b7c7-396ecdc5584d)

![image](https://github.com/AryanShr/offenSec/assets/96382618/1eb9007c-54f3-48b9-80b9-4fbf9f53b3d8)

## Architecture Overview

The overall architecture flow follows these steps:

1. Input malware sample to the GAN model.
2. Generate an Adversarial Feature Vector.
3. Mutate the malware using actions suggested by the Reinforcement Learning algorithm.
4. Utilize the Adversarial Feature Vector to select imports and sections to modify in the malware.
5. Rebuild the malware without breaking it, resulting in a mutated malware file.

![Architecture Diagram](https://github.com/AryanShr/offenSec/assets/96382618/73913c80-9939-428d-a371-593f4bf9a809)

## Getting Started

To run the project, follow these steps:

1. Generate the Adversarial Feature Vector:
  - Extract imports and sections from a dataset of PE files using the `Feature extraction` notebook in the `AdversarialFeatureVectorGeneration` folder.
  - Run the `MalGAN.ipynb` notebook to obtain the Adversarial Feature Vector.
  - Execute `advFeatureExtractor.ipynb` to generate a list of imports and sections using the pickle file from the previous step.

2. Mutate the malware using the Deep Reinforcement Learning model:
  - Install the required dependencies by running `pip install -r requirements.txt` in the `malenv` directory.
  - Prepare your malware and benign datasets in the `Data` folder, following the provided folder structure.
  - Execute `python rl_train.py` to train the Reinforcement Learning model.
  - Customize the hyperparameters (number of episodes, mutations per episode, and buffer size) by modifying the respective variables in the `main` function of `rl_train.py`.
  - The trained model will be stored in the `modified/updatedEpsilon` directory at every 10 episodes (configurable).
  - To test the trained Reinforcement Learning model, run `python rl_test.py` and provide the path to the trained model.

## Documentation

For an in-depth understanding of the project, please refer to the report [Major_Project.pdf](https://github.com/AryanShr/offenSec/files/15384725/Major_Project__Version_463_.1.pdf).


## References

CyberForce, “Pesidious: Malware mutation using reinforcement learning and generative adversarial networks,” 2020

Anderson, H., Kharkar, A., Filar, B., Evans, D. and Roth, P. (2018). Learning to Evade Static PE Machine Learning Malware Models via Reinforcement Learning. [online] arXiv.org. Available at: https://arxiv.org/abs/1801.08917.

T. D. L. Phan, T. An, N. H. Nguyen, N. N. Khoa, N. V. Pham, and P. Van-Hau, “Leveraging reinforcement learning and generative adversarial networks to craft mutants of windows malware against black-box malware detectors,” in Proceedings of the 11th International Symposium on Information and Communication Technology, pp. 31–38, 2022.

J. Ho, A. Jain, and P. Abbeel, “Denoising diffusion probabilistic models,” 2020.

Fang, Z., Wang, J., Li, B., Wu, S., Zhou, Y. and Huang, H. (2019). Evading Anti-Malware Engines With Deep Reinforcement Learning. [online] Ieeexplore.ieee.org. Available at: https://ieeexplore.ieee.org/abstract/document/8676031 [Accessed 25 Aug. 2019].
https://resources.infosecinstitute.com. (2019). 

Malware Researcher’s Handbook (Demystifying PE File). [online] Available at: https://resources.infosecinstitute.com/2-malware-researchers-handbook-demystifying-pe-file/#gref.

Hu, W. and Tan, Y. (2018). Generating Adversarial Malware Examples for Black-Box Attacks Based on GAN. [online] arXiv.org. Available at: https://arxiv.org/abs/1702.05983.

I. J. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, and Y. Bengio, “Generative adversarial networks,” 2014.

## Contributors

* **Aryan Shrivastav** - *B.Tech 4th Year, National Institute of Technology Goa*
* **Aditya** - *B.Tech 4th Year, National Institute of Technology Goa*
* **Dr. Modi Chirag Navinchandra** - *Associate Professor, National Institute of Technology Goa* 
