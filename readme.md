# Deep Learning Using Python

This intensive two-day workshop provides a comprehensive introduction to deep learning and its implementation in Python using PyTorch.
The course is designed for participants who are new to deep learning, covering both the theoretical foundations and practical implementation of modern neural network architectures.
Starting with the biological and mathematical basis of artificial neural networks, the course progresses through multilayer perceptrons and convolutional neural networks, and culminates in transformer architectures and large language models.
Through hands-on coding with PyTorch and real-world datasets, participants gain both conceptual understanding and practical skills in building, training, and applying deep learning models to research problems.

#### Who Should Attend

This course is designed for researchers, data scientists, and professionals who:

- Want to learn deep learning from the ground up
- Have experience with Python programming and basic data science
- Need to apply neural networks to research or applied problems
- Are interested in modern architectures including CNNs and transformers
- Want to understand both the theory and practice of deep learning
- Plan to work with image data, text data, or other complex data types

#### Prerequisites

- Familiarity with Python programming (writing functions, loops, working with libraries)
- Some exposure to NumPy arrays and basic array operations
- Awareness of basic machine learning concepts (training/test splits, overfitting) is helpful
- Some exposure to linear algebra concepts (vectors, matrices, matrix multiplication) is helpful
- Familiarity with basic calculus concepts (derivatives, gradients) is helpful but not required
- No prior experience with neural networks or deep learning required
- No prior PyTorch experience required

#### Day 1: Foundations and Multilayer Perceptrons

**Session 1 (2 hours) — Introduction to Artificial Neural Networks**

- Biological neurons vs artificial neurons: history and motivation
- The perceptron: weighted sums, bias, activation functions
- From perceptrons to multilayer networks: the need for non-linearity
- Activation functions: sigmoid, tanh, ReLU, GELU
- Network architecture: input layers, hidden layers, output layers
- Forward propagation: computing predictions
- The universal approximation theorem
- History of neural networks: perceptrons, AI winters, the deep learning revolution

**Session 2 (2 hours) — Training Neural Networks**

- Loss functions: mean squared error, cross-entropy
- Backpropagation: computing gradients through the network
- Gradient descent and optimisation: SGD, momentum, Adam
- Batch training: mini-batches, epochs, iterations
- Learning rates and convergence
- Overfitting and regularisation: dropout, weight decay, early stopping
- Train/validation/test splits for deep learning
- Introduction to PyTorch: tensors, autograd, computational graphs

**Session 3 (2 hours) — Multilayer Perceptrons with PyTorch**

- PyTorch fundamentals: tensors, devices (CPU/GPU), data types
- Building networks with nn.Module
- Defining layers, forward passes, and network architecture
- Training loops: forward pass, loss computation, backward pass, parameter updates
- MNIST digit classification: a complete MLP example
- Monitoring training: loss curves, validation accuracy
- Using skorch: scikit-learn wrapper for PyTorch
- Practical tips: debugging, common errors, GPU acceleration

#### Day 2: Convolutional Networks and Transformers

**Session 1 (2 hours) — Convolutional Neural Networks**

- Limitations of MLPs for images: parameter explosion, lack of spatial structure
- Convolutional layers: filters, kernels, feature maps
- Convolution operation: local connectivity, parameter sharing, translation invariance
- Pooling layers: max pooling, average pooling, dimensionality reduction
- CNN architectures: stacking conv layers, increasing depth
- Building CNNs in PyTorch: Conv2d, MaxPool2d, BatchNorm2d
- Image classification with CNNs using real-world image datasets
- Visualising learned features and filters

**Session 2 (2 hours) — Introduction to Language Models and Transformers**

- The language modelling task: predicting the next word/token
- From RNNs to transformers: the attention revolution
- Tokenisation: character-level, byte-pair encoding (BPE), subword tokens
- Embeddings: representing words as vectors
- Self-attention mechanism: queries, keys, values
- Multi-head attention: learning multiple attention patterns
- Positional encodings: incorporating sequence order
- Causal masking: autoregressive generation
- Transformer blocks: attention + feedforward networks
- The GPT architecture: decoder-only transformers

**Session 3 (2 hours) — Implementing and Using GPT Models**

- Building a minimal GPT from scratch in PyTorch
- Training a character-level language model
- Text generation: sampling, temperature, top-k, top-p
- Using the Hugging Face Transformers library
- Working with pre-trained models: AutoModel, AutoTokenizer
- Text classification with transformers
- Fine-tuning pre-trained models on custom datasets
- The Trainer API: high-level training interface
- Understanding LLMs: scale, emergence, capabilities and limitations
- Current landscape: GPT-4, Claude, LLaMA, and the open-source ecosystem

### Format

- 2 days, 6 sessions (2 hours each)
- Intensive hands-on workshop with extensive live coding
- Python scripts or Jupyter notebooks for all examples and exercises
- Participants write and execute code throughout the course
- Real datasets: MNIST digits, text corpora
- Mix of conceptual explanations and practical implementation
- From-scratch implementations to understand fundamentals, production libraries for practical applications
- All code and materials provided for independent learning after the course

### Software

Software requirements and installation instructions are in [software.md](software.md).

### Learning Outcomes

By the end of this course, participants will be able to:

- Understand the mathematical and conceptual foundations of neural networks
- Explain how backpropagation and gradient descent train neural networks
- Build and train multilayer perceptrons using PyTorch
- Implement convolutional neural networks for image classification
- Understand the transformer architecture and self-attention mechanism
- Build simple language models and GPT-style transformers from scratch
- Use pre-trained models from Hugging Face for text classification and generation
- Choose appropriate architectures for different types of data (tabular, images, text)
- Apply regularisation techniques to prevent overfitting
- Monitor training progress and diagnose common training problems
- Use GPUs to accelerate training when available
- Read and understand deep learning research papers and code

### What This Course Does Not Cover

- Classical machine learning algorithms (decision trees, random forests, SVMs)
- Advanced CNN architectures (ResNets, vision transformers)
- Object detection, segmentation, or advanced computer vision tasks
- Recurrent neural networks in depth (covered minimally in historical context)
- Generative adversarial networks (GANs)
- Reinforcement learning
- Deployment and production systems
- Distributed training or very large-scale models
