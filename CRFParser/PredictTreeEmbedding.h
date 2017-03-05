#pragma once
#include "common.h"
#include "Oracle.h"
#include "TreeReader.h"
#include "LocalLSTM2.h"

class word_layer;

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

	Parameters* p_start_sen[2];
	Parameters* p_end_sen[2];
	Expression start_sen[2];
	Expression end_sen[2];

	Expression start_sen_out[2];
	Expression end_sen_out[2];

	vector<Expression> embeddings_sentence;

	void init(Model* model, word_layer* wl, int word_dim, int pos_dim);
	void init_expression(ComputationGraph* hg);

	vector<Expression>& build(sentence_tr& sen, ComputationGraph* hg);
};

struct tree_item {
	vector<int> left_children;
	vector<int> right_children;
	int par_idx;

	LSTMState pend_exp_left0;
	LSTMState pend_exp_left;
	LSTMState pend_exp_right0;
	LSTMState pend_exp_right;
	Expression pend_exp_enc_base;
};
typedef vector<tree_item> sentree_tr;

void build_sentree(sentence_tr& sen, sentree_tr& sentree);

class PredictTreeEmbedding
{
public:
	elements_layer base_input;
	LSTM_Element tree_rnnleft1;
	LSTM_Element tree_rnnleft2;
	LSTM_Element tree_rnnright1;
	LSTM_Element tree_rnnright2;
	Parameters* p_Btree;
	Parameters* p_Wtree;
	Parameters* p_Wtreel;

	LookupParameters* p_r; // relation embeddings

public:
	PredictTreeEmbedding(Model* model);
	void Init_layer(parser_config& cfg, word_layer* wl, LookupParameters* p);
	vector<Expression> build(sentence_tr& sen, ComputationGraph* hg);
};

