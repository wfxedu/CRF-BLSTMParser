# CRF-BLSTMParser

This parser is a graph-based parser with first order factorization and built on the C++ neural network library made by Dyer et al[1]. It has following features:

1) We utilize the neural network model proposed by Wang and Chang[2] to score the dependency tree because of bidirectional LSTM (BLSTM) efficiently capturing richer contextual information. 

2) The parser is first order factorization and decodes with the Eisner algorithm so it run fast.

3) We use the conditional random field model to train the parser because it can alleviate the label bias problem [3] and implement it as follow (adding a CRF layer),

	a) Calculating the scores of the possible arcs in a sentence.
	b) U_i

%
\begin{eqnarray}
\begin{array}{lll}
\begin{cases}
i_c = tanh(b_i + M_{xi} \cdot dropout(x) + M_{ci} \cdot c_{t?1} + M_{hi} \cdot h_{t?1} \\
c_t = i_c \cdot tanh(b_c + M_{xc} \cdot dropout(x) + M_{hi} \cdot h_{t?1}) + (1 ? i_c ) \cdot c_{t?1} \\
i_o = logistic(b_o + M_{xo} \cdot dropout(x) + M_{co} \cdot c_t + M_{ho} \cdot h_{t?1} ) \\
h_t = dropout(tanh(c_t) \cdot i_o ) 
\end{cases}
\end{array}
\end{eqnarray}
%
%

# Result


# References
[1] Dyer C, Ballesteros M, Ling W, et al. Transition-based dependency parsing with stack long short-term memory[C]. 53rd Annual Meeting of the Association for Computational Linguistics and the 7th International Joint Conference on Natural Language Processing of the Asian Federation of Natural Language Processing, ACL-IJCNLP 2015, July 26, 2015 - July 31, 2015, 2015: 334-343.

[2] Wang, W., Chang, B.: Graph-based dependency parsing with bidirectional lstm. In: ACL 2016, Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics, August 7-12, 2016, Berlin, Germany, Volume 1: Long Papers (2016)

[3] Andor, D., Alberti, C., Weiss, D., Severyn, A., Presta, A., Ganchev, K., Petrov, S., Collins, M.: Globally normalized transition-based neural networks. In: ACL 2016, Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics, August 7-12, 2016, Berlin, Germany, Volume 1: Long Papers (2016)