# CRF-BLSTMParser

This parser is a graph-based parser with first order factorization and built on the [C++ neural network library](https://github.com/clab/lstm-parser) made by Dyer et al[1]. It has following features:

1) We utilize the neural network model proposed by Wang and Chang[2] to score the dependency tree because of bidirectional LSTM (BLSTM) efficiently capturing richer contextual information. 

2) The parser is first order factorization and decodes with the [Eisner algorithm](https://github.com/zzsfornlp/nngdparser/tree/master/src/algorithms "implementation") so it run fast.

3) We use the conditional random field model to train the parser because it can alleviate the label bias problem [3] and implement it as follow (adding a CRF layer),

	a) Calculating the scores of the possible arcs in a sentence.
	b) Running the inside-outside algorithm to calculate the marginal probability 'p(w1-->w2)' of each dependency arc.
	c) summing all 'Iscorrect(w1-->w2) - p(w1-->w2)' * 'Embedding node(w1-->w2)' to get the final neural node 'Nf'.
	d) Running back propagation from 'Nf' to get the gradients of the parameters.
	e) Updating the parameters.

```c
	evaluate(hg, sen);
	double* scores_label = temp_scores;
	decodeProjectiveL(sen.size(), scores_label, parser_config::LABEL_SIZE, outsen);

	double z = LencodeMarginals(length, scores_label, parser_config::LABEL_SIZE, marginals, marginals_pure);
	double log_p = -z;
	vector<Expression> args;
	for (int h = 0;h < length;h++) {
		for (int m = 1;m < length;m++) {
			if (m == h)
				continue;
			int idx = get_index2(length, h, m);
			if ( exp_score_labels[idx] == 0) {
				printf("ERROR!");
				continue;
			}
			for (int l = 0;l < parser_config::LABEL_SIZE;l++) {
				int idx_lbl = get_index2(length, h, m, l, parser_config::LABEL_SIZE);
				double gs = 0;
				if (sen[m].parent == h && sen[m].prel_id == l) {
					gs = -1;
					log_p += scores_label[idx_lbl];
				}
				args.push_back((gs + marginals[idx_lbl])*pick(*exp_score_labels[idx], l));
			}

		}
	}
```

# Result


# References
[1] Dyer C, Ballesteros M, Ling W, et al. Transition-based dependency parsing with stack long short-term memory[C]. 53rd Annual Meeting of the Association for Computational Linguistics and the 7th International Joint Conference on Natural Language Processing of the Asian Federation of Natural Language Processing, ACL-IJCNLP 2015, July 26, 2015 - July 31, 2015, 2015: 334-343.

[2] Wang, W., Chang, B.: Graph-based dependency parsing with bidirectional lstm. In: ACL 2016, Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics, August 7-12, 2016, Berlin, Germany, Volume 1: Long Papers (2016)

[3] Andor, D., Alberti, C., Weiss, D., Severyn, A., Presta, A., Ganchev, K., Petrov, S., Collins, M.: Globally normalized transition-based neural networks. In: ACL 2016, Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics, August 7-12, 2016, Berlin, Germany, Volume 1: Long Papers (2016)
