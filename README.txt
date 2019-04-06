dataset.tsv:
Column 1: Concatenation of single letter question categorizations based on the following coding system (details of the categories described in the paper):
	Idioms(i)
	Negation(n)
	Polysemy(p)
	Quantitative Reasoning(q)
	Reference(r)
	Others(o)
Column 2: Question prompt
Column 3-6: Candidate commonsense answers
Column 7: Correct answer label

We removed 25 duplicate questions from the original dataset used in the AQuA paper experiments, resulting in a total of 2776 entires in the final AQuA dataset.