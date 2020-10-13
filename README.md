# CODAH Dataset
The **CO**mmonsense **D**ataset **A**dversarially-authored by **H**umans (CODAH) is an evaluation set for commonsense question-answering in the sentence completion style of SWAG. 
As opposed to other automatically generated NLI datasets, CODAH is adversarially constructed by humans who can view feedback from a pre-trained model and use this information to design challenging commonsense questions.
Our experimental results show that CODAH questions present a complementary extension to the SWAG dataset, testing additional modes of common sense

We are releasing the original dataset used in the CODAH paper experiments, a total of 2776 entires in the final CODAH dataset (with 25 duplicates removed). 
Questions are tagged with categories indicating types of common sense tested by the question.

## Data Format
The current CODAH dataset is available in `data/full_data.tsv`.
* Column 1: Concatenation of single letter question categorizations based on the following coding system (details of the categories described in the paper):
	* Idioms (i)
	* Reference (r)
	* Polysemy (p)
	* Negation (n)
	* Quantitative (q)
	* Others (o)
* Column 2: Question prompt
* Column 3-6: Candidate commonsense answers
* Column 7: Correct answer label

## Code
The experiments can be run with `run_cv.py` (note: this script expects the Python 3 executable to be invokable as `python3`). However, a few steps need to be taken first:

1. The data is expected to be formatted into CV folds in the directories `./gitignore/data/altsizes/codah_XX`, for each `XX` in {20, 40, 60, 80} (representing 20%, 40%, 60%, 80% of the data being used for training). Each `codah_XX` directory should contain five folders named `fold0`, `fold1`, `fold2`, `fold3`, and `fold4`. Each of those `foldX` directories should contain `train.tsv` and `test.tsv`, as well as five more sub-directories `fold0`, ..., `fold4` for the sub-folds (CV over the train set), and each sub-fold should have `train.tsv` and `test.tsv` files as well.
2. The BERT and GPT models need to be fine-tuned on the SWAG training set and saved in `./gitignore/saved_models/swag_bert_for_cv/` (or `swag_gpt1_for_cv` as appropriate).

After completing these steps, `python3 run_cv.py` will run the experiments and print a summary of the results (this may take several weeks depending on available hardware).


## Paper

Details of the dataset can be found in this paper: https://www.aclweb.org/anthology/W19-2008.pdf

Bibtex:
```
@inproceedings{chen2019codah,
  title={CODAH: An Adversarially-Authored Question Answering Dataset for Common Sense},
  author={Chen, Michael and D'Arcy, Mike and Liu, Alisa and Fernandez, Jared and Downey, Doug},
  booktitle={Proceedings of the 3rd Workshop on Evaluating Vector Space Representations for NLP},
  pages={63--69},
  year={2019}
}
```

