![alt text](https://github.com/Ebimsv/Torch-Linguist/blob/main/pics/Language-model.jpg)

# About This Project
This project is a step-by-step guide on building a language model using PyTorch. It aims to provide a comprehensive understanding of the process involved in developing a language model and its applications.

# Step 1: Accurate and concise definition of the problem
Language modeling, or LM, is the use of various statistical and probabilistic techniques to determine the probability of a given sequence of words occurring in a sentence. 
Language models analyze bodies of text data to provide a basis for their word predictions.

Language modeling is used in artificial intelligence (AI), natural language processing (NLP), natural language understanding (NLU), and natural language generation(NLG) systems, particularly ones that perform text generation, machine translation and question answering.

![alt text](https://github.com/Ebimsv/Torch-Linguist/blob/main/pics/nlp_nlu_nlg.png)

Large language models (LLMs) also use language modeling. These are advanced language models, such as OpenAI's GPT-3 and Google's Palm 2, that handle billions of training data parameters and generate text output.

The effectiveness of a language model is typically evaluated using metrics like **cross-entropy** and **perplexity**, which measure the model's ability to predict the next word accurately (I will cover them in **Step 2**). Several datasets, such as WikiText-2, WikiText-103, One Billion Word, Text8, and C4, among others, are commonly used for evaluating language models.
**Note**: In this project, I use WikiText-2.

# Step 2: Advancements and types of Language Models:  

## Different types of language models:
The research of LM has received extensive attention in the literature, which can be divided into four major development stages:

<details>
  <summary><b>1. Statistical language models (SLM)</b></summary><br/>
   
SLMs are developed based on statistical learning methods that rose in the 1990s. The basic idea is to build the word prediction model based on the **Markov assumption**, e.g., predicting the next word based on the most recent context. The SLMs with a fixed context length **n** are also called **n-gram language models**, e.g., bigram and trigram language models. SLMs have been widely applied to enhance task performance in information retrieval (IR) and natural language processing (NLP). However, they often suffer from the curse of dimensionality:  
it is difficult to accurately estimate high-order language models since an exponential number of transition probabilities need to be estimated.
Thus, specially designed smoothing strategies such as back-off estimation and Good–Turing estimation have been introduced to alleviate the data sparsity problem.
</details>

<details>
  <summary><b>2. Neural language models (NLM)</b></summary><br/>
   
Neural language models (NLM). NLMs characterize the probability of word sequences by neural networks, e.g., multi-layer perceptron (MLP) and recurrent neural networks (RNNs).
As a remarkable contribution, is the concept of **distributed representation**. Distributed representations, also known as **embeddings**, the idea is that the "meaning" or "semantic content" of a data point is distributed across multiple dimensions. For example, in NLP, words with similar meanings are mapped to points in the vector space that are close to each other. This closeness is not arbitrary but is learned from the context in which words appear. This context-dependent learning is often achieved through neural network models, such as **Word2Vec** or **GloVe**, which process large corpora of text to learn these representations.

One of the key advantages of distributed representations is their ability to capture fine-grained semantic relationships. For instance, in a well-trained word embedding space, synonyms would be represented by vectors that are close together, and it's even possible to perform arithmetic operations with these vectors that correspond to meaningful semantic operations (e.g., "king" - "man" + "woman" might result in a vector close to "queen").

**Applications of Distributed Representations:**  
Distributed representations have a wide range of applications, particularly in tasks that involve natural language understanding. They are used for:

**Word Similarity**: Measuring the semantic similarity between words.  
**Text Classification**: Categorizing documents into predefined classes.  
**Machine Translation**: Translating text from one language to another.  
**Information Retrieval**: Finding relevant documents in response to a query.  
**Sentiment Analysis**: Determining the sentiment expressed in a piece of text.  

Moreover, distributed representations are not limited to text data. They can also be applied to other types of data, such as images, where deep learning models learn to represent images as high-dimensional vectors that capture visual features and semantics. 
</details> 

## Different training approaches of Language model:

<details>
  <summary><b>1. Causal Language Models (e.g., GPT-3)</b></summary><br/>
   
Causal language models, also known as **autoregressive models**, generate text by predicting the next word in a sequence given the previous words. These models are trained to maximize the likelihood of the next word using techniques like the transformer architecture. During training, the input to the model is the entire sequence up to a given token, and the model's goal is to predict the next token. This type of model is useful for tasks such as **text generation**, **completion**, and **summarization**.
</details>

<details>
  <summary><b>2. Masked Language Models (e.g., BERT)</b></summary><br/>
   
   Masked language models (MLMs) are designed to learn contextual representations of words by predicting **masked or missing words** in a sentence. During training, a portion of the input sequence is randomly masked, and the model is trained to predict the original words given the context. MLMs use bidirectional architectures like transformers to capture the dependencies between the masked words and the rest of the sentence. These models excel in tasks such as **text classification**, **named entity recognition**, and **question answering**.
</details>

<details>
  <summary><b>3. Sequence-to-Sequence Models (e.g., T5)</b></summary><br/>
   
Sequence-to-sequence (Seq2Seq) models are trained to map an input sequence to an output sequence. They consist of **an encoder** that processes the input sequence and **a decoder** that generates the output sequence. Seq2Seq models are widely used in tasks such as **machine translation**, **text summarization**, and **dialogue systems**. They can be trained using techniques like recurrent neural networks (RNNs) or transformers. The training objective is to maximize the likelihood of generating the correct output sequence given the input.
</details>

It's important to note that these training approaches are **not mutually exclusive**, and researchers often combine them or employ variations to achieve specific goals. For example, models like T5 combine the autoregressive and masked language model training objectives to learn a diverse range of tasks.

Each training approach has its own strengths and weaknesses, and the choice of the model depends on the specific task requirements and available training data. Researchers and practitioners often experiment with different architectures and training methodologies to improve the performance of language models and adapt them to various natural language processing tasks.  

For more information, please refer to the [A Guide to Language Model Training Approaches](https://medium.com/@tom_21755/understanding-causal-llms-masked-llm-s-and-seq2seq-a-guide-to-language-model-training-d4457bbd07fa#:~:text=CLM%20models%20focus%20on%20predicting,good%20for%20tasks%20requiring%20the) chapter in the "medium.com" website.

## Different Types of Models for Language Modeling  

Language modeling involves building models that can generate or predict sequences of words or characters. Here are some different types of models commonly used for language modeling:

<details>
  <summary><b>1. N-gram Language Models</b></summary><br/>
N-gram language models are a traditional approach to language modeling that rely on statistical probabilities to predict the next word in a sequence of words. The "N" in N-gram refers to the number of previous words considered as context for prediction. 

In an N-gram model, the probability of a word is estimated based on its occurrence in the training data relative to its preceding N-1 words. For example, in a trigram model (N=3), the probability of a word is determined by the two words that immediately precede it. This approach assumes that the probability of a word depends only on a fixed number of preceding words and does not consider long-range dependencies.

Here are some examples of n-grams: 
- **Unigram**: "This", "article", "is", "on", "NLP"  
- **Bigram**: "This article", "article is", "is on", "on NLP"  
- **Trigram**: "Please turn your", "turn your homework"  
- **4-gram**: "What is N-gram method"    

![alt text](https://github.com/Ebimsv/Torch-Linguist/blob/main/pics/N-gram.png)

Here are the advantages and disadvantages of N-gram language models:

**Advantages**:

  1. **Simplicity**: They have a straightforward probabilistic framework that can be easily computed and interpreted.
  2. **Efficiency**: N-gram models are computationally efficient compared to more complex models. They require minimal memory and processing power, making them suitable for resource-constrained environments.
  3. **Robustness**: N-gram models can handle out-of-vocabulary words and noisy data reasonably well. They can still provide reasonable predictions based on the available n-gram statistics, even if they encounter unseen words.

**Disadvantages**:

  1. **Lack of Contextual Understanding**: N-gram models have limited contextual understanding since they consider only a fixed number of preceding words. They cannot capture long-range dependencies or understand the broader context of a sentence.
  2. **Data Sparsity**: N-gram models suffer from data sparsity issues, especially when the vocabulary size is large. As the n-gram order increases, the number of unique n-grams decreases exponentially, leading to sparse data and difficulties in accurately estimating probabilities.
  3. **Limited Generalization**: N-gram models often struggle with generalization to unseen or rare word combinations. They may assign low probabilities to valid but infrequent word sequences, leading to suboptimal predictions in such cases.
  4. **Lack of Linguistic Understanding**: N-gram models do not incorporate linguistic knowledge explicitly. They cannot capture syntactic or semantic relationships between words, limiting their ability to generate coherent and contextually appropriate language.  
  
Here's an example of using n-grams in Torchtext:
  
```
import torchtext
from torchtext.data import get_tokenizer
from torchtext.data.utils import ngrams_iterator

tokenizer = get_tokenizer("basic_english")
# Create a tokenizer object using the "basic_english" tokenizer provided by torchtext
# This tokenizer splits the input text into a list of tokens

tokens = tokenizer("I love to code in Python")
# The result is a list of tokens, where each token represents a word or a punctuation mark

print(list(ngrams_iterator(tokens, 3)))

['i', 'love', 'to', 'code', 'in', 'python', 'i love', 'love to', 'to code', 'code in', 'in python', 'i love to', 'love to code', 'to code in', 'code in python']
```

**Note**:
- The n-gram model, typically using trigram, 4-gram, or 5-gram 
- N-gram model inadequate for language modeling due to the presence of long-range dependencies in language.
</details>

<details>
  <summary><b>2. Recurrent Neural Network (RNN)</b></summary><br/>
Recurrent Neural Network (RNN) models, including LSTM (Long Short-Term Memory) and GRU (Gated Recurrent Unit), are all variations of neural network architectures designed to handle sequential data. Here's an overview of each model along with their advantages and disadvantages:

RNNs are the fundamental type of neural network for sequential data processing. They have recurrent connections that allow information to be passed from one step to the next, enabling them to capture dependencies across time. However, traditional RNNs suffer from the vanishing/exploding gradient problem and struggle with long-term dependencies.

**Advantages of RNNs**:

  1. Ability to capture sequential dependencies and context.
  2. Flexibility in handling variable-length input and output sequences.
  3. Suitable for tasks such as text generation, speech recognition, and language translation.

**Disadvantages of RNNs**:

  1. Difficulty in learning long-term dependencies due to the vanishing/exploding gradient problem.
  2. Limited contextual understanding of complex linguistic structures.
  3. Sequential nature limits parallelization, leading to slower processing times.   

![alt text](https://github.com/Ebimsv/Torch-Linguist/blob/main/pics/RNN.png)

PyTorch code snippet for defining a basic RNN in PyTorch:
```
import torch
import torch.nn as nn

rnn = nn.RNN(input_size=10, hidden_size=20, num_layers=2)
# input_size  – The number of expected features in the input x
# hidden_size – The number of features in the hidden state h
# num_layers  – Number of recurrent layers. E.g., setting num_layers=2 would mean stacking two RNNs together

# Create a randomly initialized input tensor
input = torch.randn(5, 3, 10)  # (sequence length=5, batch size=3, input size=10)

# Create a randomly initialized hidden state tensor
h0 = torch.randn(2, 3, 20)  # (num_layers=2, batch size=3, hidden size=20)

# Apply the RNN module to the input tensor and initial hidden state tensor
output, hn = rnn(input, h0)

print(output.shape)  # torch.Size([5, 3, 20])
# (sequence length=5, batch size=3, hidden size=20)   


print(hn.shape)  # torch.Size([2, 3, 20]) 
# (num_layers=2, batch size=3, hidden size=20)
```
</details>

<details>
  <summary><b>3. Long Short-Term Memory (LSTM):</b></summary><br/>
LSTM is an extension of the RNN architecture that addresses the vanishing gradient problem. It introduces memory cells and gating mechanisms to selectively retain or forget information over time. LSTMs have proven effective in capturing long-term dependencies and maintaining contextual information.  

**Advantages of LSTMs**:

  1. Efficiently capture and propagate information over long sequences.
  2. Mitigate the vanishing/exploding gradient problem.
  3. Improved ability to handle long-term dependencies.
  4. Better representation of sequential data and context.

**Disadvantages of LSTMs**:

  1. Increased computational complexity compared to traditional RNNs.
  2. Possibility of overfitting on small datasets.
  3. Still face challenges in understanding complex linguistic structures.

![alt text](https://github.com/Ebimsv/Torch-Linguist/blob/main/pics/LSTM.png)

- **Input Gate**: Controls information flow, determines updates.
- **Forget Gate**: Discards irrelevant information from the past.
- **Update Gate**: Calculates new candidate values for cell state.
- **Output Gate**: Controls output flow, determines output selection.

PyTorch code snippet for defining a basic LSTM in PyTorch:
```
import torch
import torch.nn as nn

input_size = 100
hidden_size = 64
num_layers = 2
batch_size = 1
seq_length = 10

lstm = nn.LSTM(input_size, hidden_size, num_layers)
input_data = torch.randn(seq_length, batch_size, input_size)
h0 = torch.zeros(num_layers, batch_size, hidden_size)
c0 = torch.zeros(num_layers, batch_size, hidden_size)

output, (hn, cn) = lstm(input_data, (h0, c0))
```
The output shape of the LSTM layer will also be `[seq_length, batch_size, hidden_size]`. This means that for each input in the sequence, there will be a corresponding output hidden state. In the provided example, the output shape is `torch.Size([10, 1, 64])`, indicating that the LSTM was applied to a sequence of length 10, with a batch size of 1, and a hidden state size of 64.

Now, let's discuss the `hn` (hidden state) tensor. Its shape is `torch.Size([2, 1, 64])`. The first dimension, 2, represents the number of layers in the LSTM. In this case, the `num_layers` argument was set to 2, so there are 2 layers in the LSTM model. The second dimension, 1, corresponds to the batch size, which is 1 in the given example. Finally, the last dimension, 64, represents the size of the hidden state.

Therefore, the `hn` tensor contains the final hidden state for each layer of the LSTM after processing the entire input sequence, following the LSTM's ability to retain long-term dependencies and mitigate the vanishing gradient problem.

For more information, please refer to the [Long Short-Term Memory (LSTM)](https://d2l.ai/chapter_recurrent-modern/lstm.html) chapter in the "Dive into Deep Learning" documentation.
</details>

<details>
  <summary><b>4. Gated Recurrent Unit (GRU):</b></summary><br/>
GRU is another variation of the RNN architecture that aims to simplify the LSTM model. It combines the forget and input gates of the LSTM into a single update gate and merges the cell state and hidden state. GRUs have similar capabilities to LSTMs but with fewer parameters, making them computationally more efficient.  

**Advantages of GRUs**:

  1. Simpler architecture compared to LSTMs, leading to reduced computational complexity.
  2. Effective in capturing long-term dependencies.
  3. Improved training speed and efficiency.
  4. Suitable for tasks with limited computational resources.

**Disadvantages of GRUs**:

  1. May have slightly reduced modeling capacity compared to LSTMs.
  2. Still face challenges in understanding complex linguistic structures.

Overall, LSTM and GRU models overcome some of the limitations of traditional RNNs, particularly in capturing long-term dependencies. LSTMs excel in preserving contextual information, while GRUs offer a more computationally efficient alternative. The choice between LSTM and GRU depends on the specific requirements of the task and the available computational resources.

```
import torch
import torch.nn as nn

input_size = 100
hidden_size = 64
num_layers = 2
batch_size = 1
seq_length = 10

gru = nn.GRU(input_size, hidden_size, num_layers)
input_data = torch.randn(seq_length, batch_size, input_size)
h0 = torch.zeros(num_layers, batch_size, hidden_size)

output, hn = gru(input_data, h0)
```
The output shape of the GRU layer will also be `[seq_length, batch_size, hidden_size]`. This means that for each input in the sequence, there will be a corresponding output hidden state. In the provided example, the output shape is `torch.Size([10, 1, 64])`, indicating that the GRU was applied to a sequence of length 10, with a batch size of 1, and a hidden state size of 64.

Now, let's discuss the `hn` (hidden state) tensor. Its shape is `torch.Size([2, 1, 64])`. The first dimension, 2, represents the number of layers in the GRU. In this case, the `num_layers` argument was set to 2, so there are 2 layers in the GRU model. The second dimension, 1, corresponds to the batch size, which is 1 in the given example. Finally, the last dimension, 64, represents the size of the hidden state.

Therefore, the `hn` tensor contains the final hidden state for each layer of the GRU after processing the entire input sequence, following the GRU's ability to capture and retain information over long sequences while mitigating the vanishing gradient problem.

For more information, please refer to the [Gated Recurrent Units (GRU)](https://d2l.ai/chapter_recurrent-modern/gru.html) chapter in the "Dive into Deep Learning" documentation.
</details>

<details>
  <summary><b>5. comparing RNN, LSTM, and GRU</b></summary><br/>
RNNs are designed to capture dependencies between previous and current inputs, making them suitable for tasks such as language modeling and speech recognition. However, they suffer from the vanishing gradient problem, limiting their ability to capture long-term dependencies. To address this issue, LSTM networks were introduced. LSTM networks use memory cells and gates to selectively retain or discard information, allowing them to remember important information over longer sequences. GRU networks are a simplified version of LSTMs that use fewer gates, resulting in a more streamlined architecture. While LSTMs and GRUs both alleviate the vanishing gradient problem, GRUs are computationally more efficient due to their simpler structure. The choice between LSTM and GRU depends on the specific task and data characteristics.  

![alt text](https://github.com/Ebimsv/Torch-Linguist/blob/main/pics/RNN_LSTM_GRU.png)
</details>

<details>
  <summary><b>6. Transformer models</b></summary><br/>
Transformer models are a type of neural network architecture that has gained significant attention in the field of language modeling. Introduced by Vaswani et al. in 2017 [Google] (https://arxiv.org/pdf/1706.03762.pdf), transformers rely on self-attention mechanisms to capture global dependencies efficiently. They have achieved remarkable success in various natural language processing tasks, including language modeling, machine translation, and text generation.  

![alt text](https://github.com/Ebimsv/Torch-Linguist/blob/main/pics/transformer.png)

**Advantages**:  

  1. Capturing Long-Range Dependencies: Transformers excel at capturing long-range dependencies in sequences by using self-attention mechanisms. This allows them to consider all positions in the input sequence when making predictions, enabling better understanding of context and improving the quality of generated text.

  2. Parallel Processing: Unlike recurrent models, transformers can process the input sequence in parallel, making them highly efficient and reducing training and inference times. This parallelization is possible due to the absence of sequential dependencies in the architecture.

  3. Scalability: Transformers are highly scalable and can handle large input sequences effectively. They can process sequences of arbitrary lengths without the need for truncation or padding, which is particularly advantageous for tasks involving long documents or sentences.

  4. Contextual Understanding: Transformers can capture rich contextual information by attending to relevant parts of the input sequence. This allows them to understand complex linguistic structures, semantic relationships, and dependencies between words, resulting in more coherent and contextually appropriate language generation.

**Disadvantages of Transformer Models**:  

  1. High Computational Requirements: Transformers typically require significant computational resources compared to simpler models like n-grams or traditional RNNs. Training large transformer models with extensive datasets can be computationally expensive and time-consuming.

  2. Lack of Sequential Modeling: While transformers excel at capturing global dependencies, they may not be as effective at modeling strictly sequential data. In cases where the order of the input sequence is crucial, such as in tasks involving time-series data, traditional RNNs or convolutional neural networks (CNNs) may be more suitable.

  3. Attention Mechanism Complexity: The self-attention mechanism in transformers introduces additional complexity to the model architecture. Understanding and implementing attention mechanisms correctly can be challenging, and tuning hyperparameters related to attention can be non-trivial.

  4. Data Requirements: Transformers often require large amounts of training data to achieve optimal performance. Pretraining on large-scale corpora, such as in the case of pretrained transformer models like GPT and BERT, is common to leverage the power of transformers effectively.

For more information, please refer to the [The Transformer Architecture](https://d2l.ai/chapter_attention-mechanisms-and-transformers/transformer.html) chapter in the "Dive into Deep Learning" documentation.

Despite these limitations, transformer models have revolutionized the field of natural language processing and language modeling. Their ability to capture long-range dependencies and contextual understanding has significantly advanced the state of the art in various language-related tasks, making them a prominent choice for many applications.

</details>

## Evaluating language model
Perplexity, in the context of language modeling, is a measure that quantifies how well a language model predicts a given test set, with lower perplexity indicating better predictive performance. In simpler terms, perplexity is calculated by taking the inverse probability of the test set and then normalizing it by the number of words. 

The lower the perplexity value, the better the language model is at predicting the test set.
**Minimizing perplexity is the same as maximizing probability**

The formula for perplexity as the inverse probability of the test set, normalized by the number of words, is as follows:

![alt text](https://github.com/Ebimsv/Torch-Linguist/blob/main/pics/Perplexity-1.png)
![alt text](https://github.com/Ebimsv/Torch-Linguist/blob/main/pics/Perplexity-2.png)

### Interpreting perplexity as a branching factor
Perplexity can be interpreted as a measure of the branching factor in a language model. 
The branching factor represents the average number of possible next words or tokens given a particular context or sequence of words.

The Branching factor of a language is the number of possible next words that can follow any word. We can think of perplexity as the weighted average branching factor of a language.

# Step 3: Choose the appropriate method: Language Modeling with Embedding Layer and LSTM

The Language Modeling with Embedding Layer and LSTM code is a powerful tool for building and training language models. This code implementation combines two fundamental components in natural language processing: an **embedding layer** and a **long short-term memory (LSTM)** network.

The embedding layer is responsible for converting text data into distributed representations, also known as **word embeddings**. These embeddings capture semantic and syntactic properties of words, allowing the model to understand the meaning and context of the input text. The embedding layer maps each word in the input sequence to a high-dimensional vector, which serves as the input for subsequent layers in the model.

The LSTM layer in the code implementation processes the word embeddings generated by the embedding layer, capturing the sequence information and learning the underlying patterns and structures in the text.

By combining the embedding layer and LSTM network, the code enables the construction of a language model that can generate coherent and contextually appropriate text. Language models built using this approach can be trained on large textual datasets and are capable of generating realistic and meaningful sentences, making them valuable tools for various natural language processing tasks such as text generation, machine translation, and sentiment analysis.

This code implementation provides a simple, clear, and concise foundation for building language models based on the embedding layer and LSTM architecture. It serves as a starting point for researchers, developers, and enthusiasts who are interested in exploring and experimenting with state-of-the-art language modeling techniques.

Through this code, you can gain a deeper understanding of how embedding layers and LSTMs work together to capture the complex patterns and dependencies within text data. With this knowledge, you can further extend the code and explore advanced techniques, such as incorporating attention mechanisms or transformer architectures, to enhance the performance and capabilities of your language models.

## This is the diagram of proposed model  

![alt text](https://github.com/Ebimsv/Torch-Linguist/blob/main/pics/LM.png)

The model we will construct corresponds to the diagram provided above, illustrating the three key components: 
an embedding layer, LSTM layers, and a classification layer. While the objectives of the LSTM and classification layers are already familiar to us, let's delve into the significance of the embedding layer.

The embedding layer plays a crucial role in the model by transforming each word, represented as an index, into a vector of **E dimensions**. This vector representation allows subsequent layers to learn and extract meaningful information from the input. It is worth noting that using indices or one-hot vectors to represent words can be inadequate as they assume no relationships between different words. 

The mapping process carried out by the embedding layer is a learned procedure that takes place during training. Through this training phase, the model gains the ability to associate words with specific vectors in a way that captures semantic and syntactic relationships, thereby enhancing the model's understanding of the underlying language structure.

# Step 4: Implementation of the selected method
## Dataset
The WkiText-103 dataset, developed by Salesforce, contains over 100 million tokens extracted from the set of verified Good and Featured articles on Wikipedia. It has 267,340 unique tokens that appear at least 3 times in the dataset. Since it has full-length Wikipedia articles, the dataset is well-suited for tasks that can benefit of long term dependencies, such as language modeling.

The **WikiText-2** dataset is a small version of the **WikiText-103** dataset as it contains only 2 million tokens. This small dataset is suitable for testing your language model.

![alt text](https://github.com/Ebimsv/Torch-Linguist/blob/main/pics/wikitext-2.png)

### 1. Prepare and preprocess data 

This repository contains code for performing exploratory data analysis on the UTK dataset, which consists of images categorized by age, gender, and ethnicity.

#### Contents

1. [Download WikiText-2 dataset](#Download-WikiText-2-dataset)
2. [Tokenize data and build a vocabulary](#Tokenize-data-and-build-a-vocabulary)
3. [Plot Histograms for Age, Gender, and Ethnicity](#plot-histograms-for-age-gender-and-ethnicity)
4. [Calculate Cross-Tabulation of Gender and Ethnicity](#calculate-cross-tabulation-of-gender-and-ethnicity)
5. [Create Violin Plots and Box Plots for Age (Separated by Gender)](#create-violin-plots-and-box-plots-for-age-separated-by-gender)
6. [Create Violin Plots and Box Plots for Age (Separated by Ethnicity)](#create-violin-plots-and-box-plots-for-age-separated-by-ethnicity)

<details>
  <summary><b>1. Download WikiText-2 dataset</b></summary><br

To download a dataset using Torchtext, you can use the `torchtext.datasets` module in Python. 
Here's an example of how to download the Wikitext-2 dataset using Torchtext:  

```
import torchtext
from torchtext.datasets import WikiText2  
data_path = "path/to/save/dataset"
train_dataset, valid_dataset, test_dataset = WikiText2(root=data_path) 
```
</details>

<details>
  <summary><b>2. Tokenize data and build a vocabulary</b></summary><br/>

To build a vocabulary and save it in PyTorch using `build_vocab_from_iterator` from `torchtext.vocab` for the Wikitext-2 dataset while using a tokenizer from `torchtext.data.utils.get_tokenizer`, you can follow these steps:

Import the necessary modules:

```
import torch
import torchtext
from torchtext.datasets import Wikitext2
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer
```

Load the Wikitext-2 dataset:

```
train_dataset, valid_dataset, test_dataset = Wikitext2()
```

Define a tokenizer using get_tokenizer:

```
tokenizer = get_tokenizer('basic_english')
```

Define a function to yield tokenized sentences from the dataset:

```
def yield_tokens(dataset):
    for example in dataset:
        yield tokenizer(example)
```

Build the vocabulary using build_vocab_from_iterator:

```
vocab = build_vocab_from_iterator(yield_tokens(train_dataset))
```

Save the vocabulary to a file:

```
torch.save(vocab, 'wikitext2_vocab.pt')
```
</details>
