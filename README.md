# MSDS-631
Deep learning is a sub-field of machine learning that focuses on learning complex, hierarchical
feature representations from raw data. Over the past decade, deep learning has been remarkably
successful at solving a massive set of problems on data types including images and sequential data.
This success drove the extension of deep learning to other discrete domains such as sets, point
clouds , graphs, and 3D shapes.


This course introduces the student to a range of topics and concepts in deep learning including
the foundation neural networks, most common neural network architectures such as MLP, convolutional neural networks, and recurrent neural networks to name a few. We will also go over advanced topics such as generative models, geometric deep learning and graph neural networks.
The course covers a practical aspects of deep learning and students get to learn how to use pytroch
for creation/training/ineference of various networks. Intuition, mathematical notions as well as the
practical aspects are all emphasized throughout the course and by the end of the course the student
should have a solid theoerical and practical foundation of deep learning


# Class details
INSTRUCTOR. Mustafa Hajij.
Class dates : May 22, 2023 - Jun 30, 2023	
Lectures are Tuesday and Friday between 10am and 12pm 	
San Francisco-101 Howard 155

INSTRUCTION FORMAT. Class runs for 1:50 hours 2 day/week. Instructor-student interaction during lecture is encouraged and we'll mix in mini-exercises, final project and attendance. All programming will be done in the Python 3 programming language, unless otherwise specified.

TARDINESS. Please be on time for class. It is a big distraction if you come in late.

Grade distribution :
* Final Project 50 % (see below for details)
* Homeworks 40 %
* Attendance and particupation 6 %. Each class attendance contributes 0.5%.
* Quiz 4 %

# Syllabus

Note material Additional does not optional. These are material that will not be covered in class but might probably be needed to finish your homwork.

* Day 1:
  * Review1 [Linear Algebra for DL](https://github.com/USFCA-MSDS/MSDS-631/blob/main/Linear_algebra_for_DL.ipynb) (notebook)
  * Review2 [Probability for DL](https://github.com/USFCA-MSDS/MSDS-631/blob/main/Introduction_to_Probability_in_Python.ipynb) (notebook)
  * Review3 [Some basic math concepts](https://github.com/USFCA-MSDS/MSDS-631/blob/main/PDFs/MathBackground-.pdf)
  * Review4  [MLE](https://github.com/USFCA-MSDS/MSDS-631/blob/main/PDFs/MaxLike%20Estimation_MLE.pdf)
  * Lecture 1 Slide [An introduction to NNs](https://github.com/USFCA-MSDS/MSDS-631/blob/main/PDFs/An%20Introduction%20to%20NNs-.pdf)
  * Fundamental algorithm [The gradient descent](https://github.com/USFCA-MSDS/MSDS-631/blob/main/the_gradient_descent_algorithm.ipynb) (notebook)
  * Pytorch [Introduction to Pytorch](https://github.com/USFCA-MSDS/MSDS-631/blob/main/Introduction_to_pytorch-.ipynb) (notebook)
  * A visual introduction to NN [playground](https://playground.tensorflow.org/#activation=tanh&batchSize=10&dataset=circle&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=4,2&seed=0.93552&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false)
  * How does a DL model transform the input data to the final prediction ? [Youtube video](https://www.youtube.com/watch?v=UOvPeC8WOt8)
  * What is a neural network ? [YouTube](https://www.youtube.com/watch?v=aircAruvnKk)
  * A minimal implementation of a NN [notebook](https://github.com/USFCA-MSDS/MSDS-631/blob/main/Minimal_implementation_of_NN.ipynb)
  * Train your first NN [MLP and dense layers](https://github.com/USFCA-MSDS/MSDS-631/blob/main/MLP_and_Dense_layer.ipynb) (notebook)
  * Additional : The gradient descent algorithm for dense layers [GD](https://github.com/USFCA-MSDS/MSDS-631/blob/main/Introduction_to_gradient_decent_for_dense_layers_in_the_context_of_binary_classification%20(2).ipynb) (notebook)
  * Additional : Stochastic Gradient Decent [Notebook](https://github.com/USFCA-MSDS/MSDS-631/blob/main/Stochastic_Dradient_Descent.ipynb)
  * Interview Questions [notebook](https://github.com/USFCA-MSDS/MSDS-631/blob/main/Some_interview_Questions_related_to_lecture_1.ipynb)
* Day 2:
  * Lecture 2 Slide [Neural Network as a classifier, backprop and universaltity of NN](https://github.com/USFCA-MSDS/MSDS-631/blob/main/Lecture_2.pdf)
  * An illustration of the sigmoid function [notebook](https://github.com/USFCA-MSDS/MSDS-631/blob/main/sigmoid_and_binary_classification.ipynb)
  * Why function composition is useful in DL [notebook](https://github.com/USFCA-MSDS/MSDS-631/blob/main/Why_composition_of_functions_is_useful_in_DL.ipynb)
  * Why neural networks can learn anything? [Youtube](https://www.youtube.com/watch?v=0QczhVg5HaI) 
  * The universal approximation theorem for neural network [visual](https://gaoxiangluo.github.io/2020/09/27/Visual-and-Rigorous-Proof-of-Universal-Approximation-Theorem-UAT/)
  * A video illustration of the universal approximation theorem of NNs [Youtube](https://www.youtube.com/watch?v=FBpPjjhJGhk)
  * A minimal introduction to automatic differentiation [Notebook](https://github.com/USFCA-MSDS/MSDS-631/blob/main/Introduction_to_auto_diff_in_pytorch.ipynb)
* Day 3:
  *  Lecture 3 Slide [Conv networks](https://github.com/USFCA-MSDS/MSDS-631/blob/main/conv_nets.pdf)  
  * Introduction to conv and pooling layers using Numpy [Notebook](https://github.com/USFCA-MSDS/MSDS-631/blob/main/Introduction_to_pooling_and_cov_layers_using_numpy.ipynb)
  * An introduction to CNNs [MNIST example](https://github.com/USFCA-MSDS/MSDS-631/blob/main/AlexNet.ipynb) (notebook)
  * Resnets and inception modules [Notebook](https://github.com/USFCA-MSDS/MSDS-631/blob/main/advanced_layers.ipynb)
* Day 4:
  * HW2 helper [Super res NN](https://github.com/USFCA-MSDS/MSDS-631/blob/main/super_res_nn.pdf)
  * MLE review  [MLE](https://github.com/USFCA-MSDS/MSDS-631/blob/main/PDFs/MaxLike%20Estimation_MLE.pdf)
  * The maximal likelihood estimation [MLE and its relations to other loss functions](https://github.com/USFCA-MSDS/MSDS-631/blob/main/Maximum_likelihood_estimation%20(2).ipynb) (notebook)
  *  Lecture Slide 4 :  [Probabilistic Deep Learning](https://github.com/USFCA-MSDS/MSDS-631/blob/main/P_DL.pdf)
* Day 5:
  *  Lecture 5 Word2Vec [PDF](https://github.com/USFCA-MSDS/MSDS-631/blob/main/word2vec.pdf)
  *  Lecture 5 Introduction to Language Models [PDF](https://github.com/USFCA-MSDS/MSDS-631/blob/main/Introduction_to_language_models.pdf)
  *  Word2Vec [notebook](https://github.com/USFCA-MSDS/MSDS-631/blob/main/word2vec_simplified.ipynb)
  *  Gensim Word2Vec [notebook](https://github.com/USFCA-MSDS/MSDS-631/blob/main/gensim_word2vec.ipynb)
  *  Very Simple Language Model [notebook](https://github.com/USFCA-MSDS/MSDS-631/blob/main/simple_language-model.ipynb)
  *  Minimal RNN [notebook](https://github.com/USFCA-MSDS/MSDS-631/blob/main/minimal_RNN_.ipynb)
  *  Interview Questions [file](https://github.com/USFCA-MSDS/MSDS-631/blob/main/interview_questions_day5)
* Day 6:
  * nn.Embedding [notebook](https://github.com/USFCA-MSDS/MSDS-631/blob/main/nn_Embedding.ipynb)
  * nn.Embedding from scratch [notebook](https://github.com/USFCA-MSDS/MSDS-631/blob/main/nn_Embedding_in_details.ipynb)
  * Cross Entropy in Pytorch [notebook](https://github.com/USFCA-MSDS/MSDS-631/blob/main/CrossEntropyLoss_.ipynb)  
  * Transformers [notebook](https://github.com/USFCA-MSDS/MSDS-631/blob/main/Transformers.ipynb)
  * Bert [notebook](https://github.com/USFCA-MSDS/MSDS-631/blob/main/Bert.ipynb)
  * Lecture 6 Transformers [PDF](https://github.com/USFCA-MSDS/MSDS-631/blob/main/transformers.pdf)
  * A great blog article about transformers [article](http://jalammar.github.io/illustrated-transformer/)
 
 
Additional
  * Common loss functions in Pytorch [notebook](https://github.com/USFCA-MSDS/MSDS-631/blob/main/common_loss_functions_in_pytorch.ipynb)
  
# Quiz (Total 4 %)
In class quiz on June 16.

# Attendance and particupation (Total 6 %)
Each class is 0.5 % Total # of class = 12.

# Homeworks (Total 40 %)

* HW1 [file](https://github.com/USFCA-MSDS/MSDS-631/blob/main/HW1)  Due date Monday May 29th
* HW2 [file](https://github.com/USFCA-MSDS/MSDS-631/blob/main/HW2)  Due date Monday June 05 at 11:59PM
* HW3 [file](https://github.com/USFCA-MSDS/MSDS-631/blob/main/HW3)  Due date Wed June 14 at 11:59PM
* HW4 [file]()  Due date Monday June 26 at 11:59PM


# Final Project (Total 50 %)

## Task1 : The Project Proposal, Data Selection (Total 5 %) (Due 11th of June)

The Project Proposal, Data Selection, and Description component hold a weightage of 5% in determining your final course grade. This stage of the project requires you to submit a well-structured project proposal that encompasses several key elements: ​

### Project Proposal: 

You need to outline the objectives and goals of your project clearly. Explain the problem you intend to address using deep learning techniques and describe the overall approach you plan to take.​

 ### Data Selection: 
 Selecting appropriate data is crucial for the success of your project. You should detail the sources from which you will obtain your data and explain why these sources are suitable for your project. Additionally, discuss any preprocessing steps that may be required.​

 ### Data Description: 
 Provide a comprehensive description of the data you will be working with. This includes information about the data format, size, attributes, and any inherent challenges or limitations associated with the data. Clearly state how the selected data aligns with your project objectives.​


## Task2 : Jupyter notebook   (Total 25 %)   (Due 30th of June)

This project requires you to create a comprehensive and well-structured Jupyter notebook that effectively presents your work. The notebook should include the following components:

### Data Preprocessing (5%)
Describe the methods and steps employed to preprocess and prepare the data for your deep learning model. This may involve tasks such as data cleaning, feature engineering, data augmentation, or any other relevant preprocessing techniques.

### Model Implementation (10%)
Detail the architecture and implementation details of your deep learning model. Include code snippets, well-commented code cells, or references to external code repositories if applicable. Explain the rationale behind your model choices and any modifications or enhancements you made to existing models.

### Methods (5%)
Provide a clear description of the methodologies used in your project. Explain the algorithms, techniques, or frameworks employed, ensuring that your approach is well-documented and reproducible.

### Experiments and Results (10%)
Present the experiments conducted during your project and report the obtained results. Include relevant performance metrics, accuracy scores, loss curves, or any other measurements used to evaluate your model's performance. Use tables, graphs, or visualizations to effectively communicate your experimental findings.

### Conclusion and Future work (5%)
Summarize the key findings and insights derived from your experiments and results. Discuss the implications of your results in the context of your project goals and objectives. Reflect on the strengths and limitations of your approach and provide suggestions for further improvement.

Please ensure that your Jupyter notebook is well-documented and organized, making it easy for others to understand and reproduce your work. The total grade for the project is out of 30% of your final course grade.


## Task3 : Recorded presentation (10%)  (Due 30th of June)

The recorded presentation contributes 10% to your overall course grade. You will be required to deliver a presentation, recorded in video format, where you showcase and explain your project, including its objectives, methodology, results, and conclusions. This should not exceed 10 minutes and should be presented  by all team members.​


# Deep Learning Project Ideas

Here are 20 project ideas for deep learning:

1. Image Classification: Build a deep learning model to classify images into various categories.
2. Object Detection: Create a model to detect objects in images or videos and draw bounding boxes around them.
3. Sentiment Analysis: Develop a model to analyze the sentiment (positive, negative, neutral) of text data such as reviews or social media posts.
4. Language Translation: Build a deep learning model to translate text from one language to another.
5. Facial Recognition: Create a model that can recognize and identify faces in images or videos.
6. Recommendation System: Develop a recommendation system using deep learning to suggest personalized recommendations for users.
7. Text Generation: Train a deep learning model to generate text, such as song lyrics, poems, or story paragraphs.
8. Anomaly Detection: Build a model that can detect anomalies or outliers in datasets.
9. Speech Recognition: Create a model that can transcribe spoken words into written text.
10. Fraud Detection: Develop a deep learning model to detect fraudulent transactions or activities.
11. Image Segmentation: Build a model to segment images into different regions or objects.
12. Emotion Recognition: Create a model that can recognize and classify emotions from facial expressions.
13. Time Series Prediction: Develop a model that can predict future values in time series data, such as stock prices or weather patterns.
14. Medical Diagnosis: Build a model for diagnosing medical conditions or diseases based on medical images or patient data.
15. Style Transfer: Create a model that can transfer the style of one image to another while preserving content.
16. Speech Synthesis: Train a deep learning model to generate human-like speech.
17. Video Action Recognition: Develop a model to recognize and classify actions or activities in videos.
18. Music Generation: Train a deep learning model to compose original music pieces.
19. Image Super-Resolution: Build a model to enhance the resolution and quality of low-resolution images.
20. Self-Driving Cars: Develop a deep learning model to control a simulated or real autonomous vehicle.

These project ideas cover a wide range of modern and interesting applications that can be implemented using deep learning techniques. Choose the one that intrigues you the most and embark on your deep learning project!



