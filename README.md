# CODAH Dataset
The **CO**mmonsense **D**ataset **A**dversarially-authored by **H**umans (CODAH) is an evaluation set for commonsense question-answering in the sentence completion style of SWAG. 
As opposed to other automatically generated NLI datasets, CODAH is adversarially constructed by humans who can view feedback from a pre-trained model and use this information to design challenging commonsense questions.
Our experimental results show that CODAH questions present a complementary extension to the SWAG dataset, testing additional modes of common sense

We are releasing the original dataset used in the CODAH paper experiments, a total of 2776 entires in the final CODAH dataset (with 25 duplicates removed). 
Questions are tagged with categories indicating types of common sense tested by the question.

## Data Format
The current CODAH dataset is available in `dataset.tsv`.
* Column 1: Concatenation of single letter question categorizations based on the following coding system (details of the categories described in the paper):
	* Idioms(i)
	* Negation(n)
	* Polysemy(p)
	* Quantitative Reasoning(q)
	* Reference(r)
	* Others(o)
* Column 2: Question prompt
* Column 3-6: Candidate commonsense answers
* Column 7: Correct answer label

