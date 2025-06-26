# Intent Classification for Punjabi Sentences

This repository focuses on intent classification for Punjabi sentences using multiple AI models and evaluation techniques. It provides a complete workflow for extracting, processing, and evaluating intent predictions from various models, with a special emphasis on both general and idiomatic sentences.

## Repository Structure

```
intent_to_be_pushed/
│
├── bert_score.ipynb
├── embeddings.ipynb
├── fix.ipynb
├── gemini.ipynb
├── gpt.ipynb
├── idioms.ipynb
├── sarvam.ipynb
├── translating.ipynb
│
├── csv/
│   ├── Gemini_Unseen.xlsx
│   ├── Gpt_Unseen.xlsx
│   ├── Sarvam_Unseen.xlsx
│   └── unseen_data1xlsx.xlsx
│
├── important_csv/
│   ├── filtered_punjabi.xlsx
│   ├── Gemini_Intent_Fixed.xlsx
│   ├── Gpt_Intent_Fixed.xlsx
│   ├── punjabi_and_intent.xlsx
│   └── Sarvam_Final_Intent.xlsx
│
└── results/
    ├── BERT.png
    ├── DotProduct_ParaphraseModel.csv
    ├── Intent_Similarity_Comparison_LaBSE.csv
    └── tpaired_test.png
```

## Folder Descriptions

- **results/**: Contains output files such as images and CSVs summarizing model performance, comparisons, and evaluation metrics.
- **csv/**: Contains unseen/test datasets (e.g., `Gemini_Unseen.xlsx`, `Gpt_Unseen.xlsx`, `Sarvam_Unseen.xlsx`, `unseen_data1xlsx.xlsx`) used for evaluating model generalization.
- **important_csv/**: Contains training datasets (e.g., `Gpt_Intent_Fixed.xlsx`, `Gemini_Intent_Fixed.xlsx`, `punjabi_and_intent.xlsx`, `Sarvam_Final_Intent.xlsx`, `filtered_punjabi.xlsx`) used for model development and evaluation.

## Notebook Descriptions

- **bert_score.ipynb**: Calculates BERTScore metrics (Precision, Recall, F1) for intent predictions from different models (Gemini, GPT, Sarvam) against ground truth.
- **embeddings.ipynb**: Compares model predictions using sentence embeddings (SBERT, LaBSE, Paraphrase models), computes distances/similarities, and ranks models.
- **fix.ipynb**: Pre-processes and fixes intent prediction files, ensuring no corruption or duplicates in the intent columns for Gemini and GPT outputs.
- **gemini.ipynb**: Runs intent extraction using the Gemini API for both training and unseen/test datasets, saving results in batches.
- **gpt.ipynb**: Runs intent extraction using the GPT (OpenRouter) API for both training and unseen/test datasets, saving results in batches.
- **idioms.ipynb**: Focuses on intent extraction and analysis for idiomatic Punjabi sentences, including step-by-step reasoning, embedding-based evaluation, and statistical significance testing.
- **sarvam.ipynb**: Runs intent extraction using the Sarvam.ai API for both training and unseen/test datasets, saving results in batches.
- **translating.ipynb**: Translates English sentences to Punjabi and back, computes cosine similarity between original and back-translated sentences, and filters based on a similarity threshold. Also includes code for fine-tuning SBERT on parallel data.

## Workflow Overview

1. **Data Preparation**: Training datasets are stored in `important_csv/`, and unseen/test datasets are in `csv/`.
2. **Intent Extraction**: Use `gemini.ipynb`, `gpt.ipynb`, and `sarvam.ipynb` to generate intent predictions from different models for both training and test data.
3. **Data Cleaning**: Use `fix.ipynb` to clean and preprocess the intent prediction files, ensuring data quality.
4. **Evaluation**:
    - Use `bert_score.ipynb` to compute BERTScore metrics for model predictions.
    - Use `embeddings.ipynb` to compare model outputs using various sentence embedding models and similarity metrics.
    - Use `idioms.ipynb` for specialized analysis on idiomatic sentences, including statistical significance testing.
5. **Translation & Augmentation**: Use `translating.ipynb` for data augmentation via translation and for fine-tuning embedding models.
6. **Results**: All evaluation outputs, comparison plots, and summary CSVs are saved in the `results/` folder.

## Requirements
- Python 3.x
- Jupyter Notebook
- pandas, numpy, scikit-learn, sentence-transformers, bert-score, openpyxl, matplotlib, seaborn, and other relevant libraries (see individual notebooks for details)

## Usage
1. Clone the repository and install the required dependencies.
2. Open the notebooks in Jupyter and follow the workflow as described above.
3. Review the results in the `results/` folder.

## Citation
If you use this repository or its datasets in your research, please cite appropriately.

---

For questions or contributions, please open an issue or submit a pull request. 