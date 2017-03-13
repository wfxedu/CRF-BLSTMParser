# CRF-BLSTMParser 

This `simple parser` is a graph-based parser with first order factorization and built on the [C++ neural network library](https://github.com/clab/lstm-parser) made by Dyer et al. It has following features: 

1) We utilize the neural network model proposed by Wang and Chang[1] to score the dependency tree because of bidirectional LSTM (BLSTM) efficiently capturing richer contextual information. Based on their model, we exploit probabilistic model, conditional random field, to alleviate the label bias problem [2].  

2) The parser is first order factorization and decodes with the [Eisner algorithm](https://github.com/zzsfornlp/nngdparser/tree/master/src/algorithms "implementation") so it runs fast. It is similar to the parser built by Zhang et al.[3] except that this parser instead employs BLSTM recurrent neural network and use dropout. Please note that Dropout node is changed to get stable results as follows,

```c
void Dropout::forward(const vector<const Tensor*>& xs, Tensor& fx) const {
	if (cnn_train_mode) {
		Tensor m(dim, (float*)aux_mem);
		TensorTools::RandomBernoulli(m, (1.f - p), 1.0f);//1.f / (1.f - p)
		(*fx) = (**xs[0]).cwiseProduct(*m);
	}
	else {
		float*v = (float*)aux_mem;
		v[0] = (1.f - p);
		(*fx) = (**xs[0])*v[0];
	}
}

void Dropout::backward(const vector<const Tensor*>& xs, const Tensor& fx, const Tensor& dEdf, unsigned i, Tensor& dEdxi) const {
	if (cnn_train_mode) {
		Tensor m(dim, (float*)aux_mem);
		(*dEdxi) += (*dEdf).cwiseProduct(*m);
	}
	else {
		float*v = (float*)aux_mem;
		(*dEdxi) += (*dEdf)*v[0];
	}
}
```
3) The gradients calculating is implemented as follow (adding a _CRF layer_),

	a) Calculating the scores of the possible arcs in a sentence.
	b) Running the inside-outside algorithm to calculate the marginal probability `p(w1-->w2)` of each dependency arc.
	c) summing all `Iscorrect(w1-->w2) - p(w1-->w2)` * `Embedding node(w1-->w2)` to get the final neural node `Nf`.
	d) Running back propagation from `Nf` to get the gradients of the parameters.
	e) Updating the parameters.
# The codes for calculating gradients
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
Evaluating with English Penn Treebank (PTB) and Chinese Treebank (CTB) version 5 with the standard splits.<br>
Development Results: <br>
	CTB5: UAS 88.3%, LAS 86.9% <br>
	PTB-YM:UAS 93.2%, LAS 92.3%<br>
Test Results:<br>
	CTB5: UAS 88.16%, LAS 86.85%<br>
	PTB-YM: UAS 93.74%, LAS 92.83%<br>
# References
[1] Wang, W., Chang, B. Graph-based dependency parsing with bidirectional lstm. In: ACL 2016, Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics, August 7-12, 2016, Berlin, Germany, Volume 1: Long Papers (2016)

[2] Andor, D., Alberti, C., Weiss, D., Severyn, A., Presta, A., Ganchev, K., Petrov, S., Collins, M. Globally normalized transition-based neural networks. In: ACL 2016, Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics, August 7-12, 2016, Berlin, Germany, Volume 1: Long Papers (2016)

[3] Zhang Z, Zhao H, Qin L. Probabilistic graph-based dependency parsing with convolutional neural network[C]. 54th Annual Meeting of the Association for Computational Linguistics, ACL 2016, August 7, 2016 - August 12, 2016, 2016: 1382-1392.
