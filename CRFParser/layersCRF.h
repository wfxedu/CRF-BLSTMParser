#pragma once
#include "commonCRF.h"
#include "TreeReaderCRF.h"
#include "LocalLSTM2.h"
namespace CRFParser {

class layer {
public:
	ComputationGraph* m_hg;
	parser_config* m_cfg;
public:
	void Init_layer(parser_config& cfg) { m_cfg = &cfg; }
	void Init_Graph(ComputationGraph* hg) { m_hg = hg; }
public:
	layer() {};
};

class word_layer : public layer {
public:
	LookupParameters* p_w; // word embeddings
	LookupParameters* p_t; // pretrained word embeddings (not updated)
	LookupParameters* p_p; // pos tag embeddings

	Parameters* p_w2l; // word to LSTM input
	Parameters* p_pre2l; // word to LSTM input
	Parameters* p_p2l; // POS to LSTM input
	Parameters* p_t2l; // pretrained word embeddings to LSTM input
	Parameters* p_ib; // LSTM input bias

					  // l =ib + w2l*w + p2l*p + t2l*t
	Expression ib;
	Expression w2l;
	Expression pre2l;
	Expression p2l;
	Expression t2l;

	void Init_layer(parser_config& cfg);
	void Init_Graph(ComputationGraph* hg);
	Expression build(unsigned word_id, unsigned pos_id, unsigned orgword_id, Expression pre);
};


struct elements_layer {
	LookupParameters* p_w;
	LookupParameters* p_p;
	LookupParameters* p_t;
	Parameters* p_Wwp;
	Parameters* p_Bwp;
	LSTMBuilder* elements_fwrnn;
	LSTMBuilder* elements_bwrnn;

	Expression Wwp;
	Expression Bwp;

	vector<Expression> embeddings_sentence;

	void init(Model* model, word_layer* wl, int word_dim, int pos_dim);
	void init_expression(ComputationGraph* hg);
	vector<Expression>& build(sentence_tr& sen, ComputationGraph* hg);
};

enum {
	RE_LEFT = 0,
	REL_RIGHT = 1
};

struct feature {
	Parameters* p_W1x[2];
	Parameters* p_B1[2];

	Expression W1x[2];
	Expression B1[2];

	void init(Model* model, unsigned int fea_dim, unsigned int out_dim) {
		for (int i = 0;i < 2;i++) {
			p_W1x[i] = model->add_parameters(Dim(out_dim, fea_dim));
			p_B1[i] = model->add_parameters(Dim(out_dim, 1));
			//p_B1[i]->need_regression = false;
		}
	}

	void init_expression(ComputationGraph* hg) {
		for (int i = 0;i < 2;i++) {
			W1x[i] = parameter(*hg, p_W1x[i]);
			B1[i] = parameter(*hg, p_B1[i]);
		}
	}

	Expression build(Expression x, int d) {
		return affine_transform({ B1[d], W1x[d],  x });
	}
};

}//namespace CRFParser {