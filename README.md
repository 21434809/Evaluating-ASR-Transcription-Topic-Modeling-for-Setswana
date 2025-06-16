# Evaluating ASR Transcription and Topic Modeling for Setswana

## Project Overview
This project evaluates the performance of Automatic Speech Recognition (ASR) systems and applies topic modeling techniques to Setswana transcriptions. The goal is to analyze the quality of ASR transcriptions and explore how well topic modeling can extract meaningful topics from Setswana text data.

## Features
- **ASR Transcription**: Scripts for processing audio files and generating transcriptions using models like Wav2Vec2.
- **Topic Modeling**: Implementation of Latent Dirichlet Allocation (LDA) and BERTopic for extracting topics from Setswana text.
- **Data Preprocessing**: Includes lemmatization, stopword removal, and sentence splitting.
- **Visualization**: Generate visualizations such as word clouds and statistical charts.

## Repository Structure
- **data/**: Contains raw and processed datasets, including Setswana stopwords and transcriptions.
- **docs/**: Includes results and visualizations generated during the analysis.
- **models/**: Directory for storing trained models.
- **notebooks/**: Jupyter notebooks for interactive exploration and experimentation.
- **src/**: Python scripts for preprocessing, transcription, and topic modeling.
- **references/**: Additional reference materials.

## Requirements
### Python Packages
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `gensim`
- `nltk`
- `torch`
- `transformers`

### Additional Tools
- Jupyter Notebook (optional, for interactive exploration)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/Evaluating-ASR-Transcription-Topic-Modeling-for-Setswana.git
   cd Evaluating-ASR-Transcription-Topic-Modeling-for-Setswana
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## How to Run the Project
### Data Preparation
1. Place your Setswana audio files in the appropriate directory.
2. Ensure the stopword list and other preprocessing files are in the `data/` folder.

### Transcription
Run the transcription scripts in `src/`:
```bash
python src/transcribe_setswana.py
```

### Topic Modeling
Use the notebooks in `notebooks/` for topic modeling:
1. Open `BertTopic.ipynb` or `LDA.ipynb`.
2. Follow the instructions to preprocess data and generate topics.

### Visualization
Generate visualizations using the scripts or notebooks provided in the repository.

## Results
Results and visualizations are stored in the `docs/` folder, including:
- Word clouds
- Statistical charts
- Topic modeling outputs

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request.

## License
This project is licensed under the MIT License.





