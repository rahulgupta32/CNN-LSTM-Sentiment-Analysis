# CNN-LSTM-Sentiment-Analysis
A Hybrid CNN-LSTM Approach for Sentiment Analysis on IMDB Movie Reviews


CNN-LSTM Sentiment Analysis
This project implements a hybrid CNN-LSTM deep learning model for sentiment analysis. It uses a dataset of movie reviews to classify each review as positive or negative. The notebook combines the feature extraction strength of CNNs with the temporal understanding capability of LSTMs for improved accuracy.

 ### Overview
The primary goal of this project is to:
•	Clean and preprocess text data.
•	Use word embeddings (via Keras Tokenizer and padded sequences).
•	Build and train a hybrid CNN-LSTM model.
•	Evaluate the model performance on test data.

 ### Technologies & Libraries Used
•	Python
•	Keras / TensorFlow
•	NumPy, Pandas
•	Matplotlib & Seaborn (for visualization)
•	Scikit-learn (for metrics like accuracy, confusion matrix)
•	NLTK (for stopword removal and tokenization)

### Dataset
The notebook uses the IMDb movie reviews dataset from Keras. You can download the dataset from here: https://drive.google.com/file/d/1udefbVj0MAP7Vuae7D7tLtcGr7YNrQH6/view?usp=sharing



 ### Key Components
•	Data Loading & Preprocessing
I.	Load the IMDb dataset.
II.	Convert sequences to padded inputs.




•	Model Architecture
I.	Embedding Layer
II.	1D Convolutional Layer (CNN)
III.	Max Pooling
IV.	LSTM Layer
V.	Dense Output Layer with Sigmoid activation


•	Training
Model is compiled with binary_crossentropy loss and adam optimizer.


•	Evaluation
Accuracy_loss, ROC Curve, and confusion matrix visualization.



Flowchart of the Proposed Model: ![light_workflow_diagram](https://github.com/user-attachments/assets/405bed26-b742-44fd-a8d1-858a23d31887)




ROC_Curve: ![roc_curve](https://github.com/user-attachments/assets/80f59e22-81af-4eb8-bd0b-c94ffcfcd5a3)




Precision_Recall_Curve: ![precision_recall_curve](https://github.com/user-attachments/assets/710eb40e-2d7b-42c2-993c-23a81dfa598e)





Accuracy_Loss_Curve: ![train_test_accuracy_loss](https://github.com/user-attachments/assets/5fca9d55-0625-4070-ade6-9268ceaba133)


