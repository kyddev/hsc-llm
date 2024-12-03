
<h1 align="center">
  HSC LLM
  <br>
</h1>

<h4 align="center">A zero-cost and accessible way to study for the HSC</h4>

<p align="center">
  <a href="#key-features">Key Features</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#how-to-upload-an-entire-directory-of-documents">How to Upload</a> •
  <a href="#how-to-evaluate-the-model">How to Evaluate</a> •
  <a href="#license">License</a>
</p>

<!-- ![screenshot](https://raw.githubusercontent.com/amitmerchant1990/electron-markdownify/master/app/img/markdownify.gif) -->

> **Note**
> This is a project submitted as a Bachelor of Computing Science (Honours) thesis. 

## Key Features

* Uses Meta's Llama3.1-8b model as the backbone of the application
  - High quality answers all locally. 
* Implements Retrieval Augmented Generation (RAG) for document upload and search
  - Further improved with a reranking model for more relevant search results. 
* Friendly User Interface using Streamlit
* Cross platform
  - Windows and macOS ready.
* Generates metric scores for answers 
  - Provides a simple ground truth dataset to measure LLM performance. 

## How To Use

To clone and run this application, you'll need [Git](https://git-scm.com), [Ollama](https://ollama.com/) and [Anaconda](https://www.anaconda.com/download/) installed on your computer. 

Make sure to download the relevant models using Ollama and the command line.

```bash
# Download Meta Llama3.1:8b, Llama3.2-1b and Google Gemma2:9b
$ ollama pull llama3.1:8b

$ ollama pull llama3.2:1b

$ ollama pull gemma2:9b

```

Next, from your command line:

```bash

# Download Ollama from the official Ollama
# Clone this repository
$ git clone https://github.com/kyddev/hsc-llm

# Go into the repository
$ cd hsc-llm

# Install dependencies by using anaconda 
$ conda env create -f environment.yml

# Run the app using the Miniconda terminal
$ streamlit run app.py
```

## How To Upload An Entire Directory Of Documents

Please use the `directory_embedder.ipynb` to embed an entire directory of documents into the application.

## How to Evaluate the Model

To evaluate the models, please use the notebooks provided within this following order.


* Assuming that there are files currently embedded within the application.
    - Use `generation_testbench.ipynb` to generate the responses using the LLM model.
    - Afterwards, use `metrics.calculation.ipynb` to calculate the BERTScore and METEOR score of the model.
    - Lastly, use `llmjudge_evaluation.ipynb` to generate the evaluation of the Gemma2 model using the golden dataset `ground_truth.csv`. 


## Credits

This software uses the following open source packages:

- [Python](https://www.python.org/)
- [Langchain](https://www.langchain.com/)
- [Ollama](https://ollama.com/)
- [Chroma](https://www.trychroma.com/)

## License

MIT



