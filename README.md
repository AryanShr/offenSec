# Malware Mutation System for Offensive Security

## Project Objective
This project aims to design a **mutation algorithm for malware** using advanced techniques to enhance its adaptability and evasion capabilities. 

## Approach
The approach involves the use of **Machine Learning techniques** such as **Reinforcement Learning** and **Generative Models**. The ultimate goal is to generate mutations of existing malwares that can bypass anti-viruses which utilize static ML-based techniques and most of the signature-based detection techniques.

## Comparison with Existing Projects
While most projects in this domain utilize **Deep Reinforcement Learning** in conjunction with **Generative Adversarial Networks (GANs)** to overcome each other's limitations, this project also employs similar tech stacks. However, it aims to enhance the results while maintaining the computational efficiency of the algorithms at work.

## Project Background
This project was developed as a **final year BTech project**. For an in-depth understanding of the project, please refer to the report here.[Major_Project.pdf](https://github.com/AryanShr/offenSec/files/15384725/Major_Project__Version_463_.1.pdf)


## Project Innovations
This project diverges from the conventional use of GAN models, like the MalGAN model, which are known to have multiple limitations, including the notorious mode collapse problem that results in the generation of similar outcomes.

To tackle this, our project introduces an innovative approach that amplifies the randomization of noise in the malware feature vector. This is accomplished by employing diffusion models in conjunction with the GAN model, resulting in the ultimate **Diffusion-based GAN model**. This pioneering approach effectively resolves the mode collapse problem and achieves a faster True Positive Rate (TPR) convergence.

![image](https://github.com/AryanShr/offenSec/assets/96382618/126c3743-e5d9-4bc5-9a04-bc7ecf7724b8)


In the Reinforcement Learning (RL) algorithm from gym malware, we have implemented changes to reduce exploration and increase exploitation by using an inverse square function. We have also introduced a reward shaping methodology to distribute rewards based on various factors. This has enabled us to achieve a superior distribution of the action space during mutation compared to the original gym malware, which also follows the algorithm by Hein et al. (2022).

![image](https://github.com/AryanShr/offenSec/assets/96382618/05620f8a-011e-400d-b7c7-396ecdc5584d)

![image](https://github.com/AryanShr/offenSec/assets/96382618/1eb9007c-54f3-48b9-80b9-4fbf9f53b3d8)


#### The overall architecture flow is as follows: Input to GAN -> Generation of Adversarial Feature Vector -> Mutating malware with actions -> Using Adversarial Feature Vector to select imports and sections to modify the malware -> Rebuilding the malware without breaking it -> Resulting in a Mutated Malware file.

![image](https://github.com/AryanShr/offenSec/assets/96382618/73913c80-9939-428d-a371-593f4bf9a809)

