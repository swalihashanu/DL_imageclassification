# DL_imageclassification
The employed classification model is a Convolutional Neural Network (CNN) designed with three convolutional layers featuring max-pooling, a dropout layer, and a flattening layer. The dropout layer, set at a 0.2 dropout rate, removes 20% of input units during training. Subsequently, dense layers, including two with ReLU activation and one with softmax activation for multiclass classification, are incorporated. The model is assessed using sparse categorical crossentropy loss, accuracy metrics, and the Adam optimizer.

## Training Process:

**Data Loading and Preprocessing:**
- Images of celebrities like Lionel Messi, Maria Sharapova, Roger Federer, Serena Williams, and Virat Kohli are loaded and resized to (128, 128) pixels.
- The dataset is divided into training and testing sets.

**Model Architecture:**
- The CNN model is constructed with convolutional layers, max pooling layers, dropout layer, and dense layers.
- The model is compiled using the Adam optimizer and sparse categorical crossentropy loss.

**Training:**
- The model is trained for 20 epochs on the training set with a batch size of 32.
- Training progress, accuracy, and loss are monitored.

**Evaluation:**
- Accuracy and loss are assessed using graphical representations.

**Model Prediction:**
- The trained model is utilized to predict labels on the test set.
- Predictions and actual labels are saved in a CSV file.

**Critical Findings:**
- Model accuracy, indicating its generalization ability to new data, is evaluated.
- Modifying learning rates, dropout rates, or the model architecture somehow enhances accuracy.
