<h1>Food Classification with Transfer Learning using EfficientNetV2B0</h1>

<p>This repository contains a Convolutional Neural Network (CNN) model for food classification using the Food101 dataset. The model leverages transfer learning with EfficientNetV2B0 as the base model. The goal of this project is to surpass the accuracy reported in the reference experiment paper.</p>

<h2>Table of Contents</h2>
<ul>
  <li><a href="#introduction">Introduction</a></li>
  <li><a href="#dataset">Dataset</a></li>
  <li><a href="#model-architecture">Model Architecture</a></li>
  <li><a href="#training-and-fine-tuning">Training and Fine-Tuning</a></li>
  <li><a href="#results">Results</a></li>
  <li><a href="#tools-used">Tools Used</a></li>
  <li><a href="#references">References</a></li>
  <li><a href="#contributing">Contributing</a></li>
  <li><a href="#license">License</a></li>
  <li><a href="#fine-tuned-model">Fine-Tuned Model Access</a></li>
  <li><a href="#loading-fine-tuned-model">Loading Fine-Tuned Model</a></li> <!-- New section -->
</ul>

<h2 id="introduction">Introduction</h2>
<p>Food classification is a challenging task due to the large variety of food types and the visual similarity between certain classes. This project uses transfer learning with EfficientNetV2B0 to build a high-accuracy food classification model. The goal is to achieve and surpass the accuracy reported in the experiment paper <a href="https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/static/bossard_eccv14_food-101.pdf">link to paper</a>.</p>

<h2 id="dataset">Dataset</h2>
<p>The Food101 dataset consists of 101,000 images of food divided into 101 categories, with 750 training images and 250 test images per class. The dataset can be directly loaded from TensorFlow Datasets.</p>

<h2 id="model-architecture">Model Architecture</h2>
<p>The base model used is EfficientNetV2B0, a state-of-the-art CNN architecture known for its efficiency and performance. The pre-trained weights on ImageNet are used for transfer learning. The model architecture is modified by adding a few fully connected layers on top of the base model for the food classification task.</p>

<h2 id="training-and-fine-tuning">Training and Fine-Tuning</h2>
<p>1. <strong>Transfer Learning</strong>: The EfficientNetV2B0 base model is used with pre-trained weights, and only the top layers are trained initially.<br>
2. <strong>Fine-Tuning</strong>: After achieving a satisfactory baseline accuracy, the entire model is fine-tuned by unfreezing the base model and training with a lower learning rate.</p>

<h3>Steps:</h3>
<ol>
  <li>Load the Food101 dataset.</li>
  <li>Preprocess the images and labels.</li>
  <li>Build the model using EfficientNetV2B0 as the base.</li>
  <li>Compile the model with appropriate loss function and optimizer.</li>
  <li>Train the top layers.</li>
  <li>Fine-tune the entire model.</li>
  <li>Evaluate the model on the test set.</li>
</ol>

<h2 id="results">Results</h2>
<p>The model achieved an accuracy of 81% on the Food101 dataset, surpassing the accuracy reported in the reference experiment paper.</p>

<h2 id="tools-used">Tools Used</h2>
<ul>
  <li>Google Colab</li>
  <li>TensorFlow</li>
  <li>NumPy</li>
  <li>Pandas</li>
  <li>Matplotlib</li>
</ul>

<h2 id="references">References</h2>
<ul>
  <li><a href="https://arxiv.org/abs/1905.11946">EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks</a></li>
  <li><a href="https://www.tensorflow.org/datasets/catalog/food101">Food101 Dataset</a></li>
  <li><a href="https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/static/bossard_eccv14_food-101.pdf">Experiment Paper</a></li>
</ul>

<h2 id="contributing">Contributing</h2>
<p>Contributions are welcome! Please submit a pull request or open an issue to discuss your ideas.</p>

<h2 id="license">License</h2>
<p>This project is licensed under the MIT License. See the <a href="LICENSE">LICENSE</a> file for details.</p>

<h2 id="fine-tuned-model">Fine-Tuned Model Access</h2>
<p>The fine-tuned model weights can be accessed and downloaded from the following link:</p>
<p><a href="https://drive.google.com/drive/folders/1vQviiq34mWrn_Kop94N8t2izCGGJXq86?usp=drive_link">Download Fine-Tuned Model</a></p>

<h2 id="loading-fine-tuned-model">Loading Fine-Tuned Model</h2>
<p>To load the fine-tuned model in your Python code, you can use the <code>tf.keras.models.load_model</code> function. First, make sure TensorFlow is installed:</p>
<pre><code>pip install tensorflow</code></pre>
<p>Then, use the following code snippet to load the model:</p>
<pre><code>import tensorflow as tf

Replace 'path_to_your_model_weights' with the actual path where you saved the model weights
model = tf.keras.models.load_model('path_to_your_model_weights')</code></pre>
<p>Now you can use the <code>model</code> object for inference on new images.</p>

</body>
</html>
