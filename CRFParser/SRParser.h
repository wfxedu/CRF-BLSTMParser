#pragma once
#include "common.h"
#include "Oracle.h"
#include "PredictTreeEmbedding.h"

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


class parser_state : public layer 
{
public:
	LSTMBuilder stack_lstm; // (layers, input, hidden, trainer)
	LSTMBuilder buffer_lstm;
	LSTMBuilder action_lstm;

	Parameters* p_action_start;  // action bias
	Parameters* p_buffer_guard;  // end of buffer
	Parameters* p_stack_guard;  // end of stack

	Expression action_start;
public:
	vector<Expression> word2inner_mean;
	vector<Expression> buffer;
	vector<int> bufferi;

	vector<Expression> stack;  // variables representing subtree embeddings
	vector<int> stacki; // position of words in the sentence of head of subtree
public:
	void Init_layer(parser_config& cfg);
	void Init_Graph(ComputationGraph* hg);

	void Init_state(const t_unvec& raw_sent, const t_unvec& sent, 
		const t_unvec& sentPos, word_layer& layer, vector<Expression>& pre_embedding);
public:
	vector<Expression> log_probs;
	vector<unsigned> results;
	void accum(Expression action_scores, int idx) { log_probs.push_back(pick(action_scores, idx)); results.push_back(idx); }
	Expression log_p_total() { return -sum(log_probs); }

public:
	void clear() { buffer.clear(); word2inner_mean.clear(); bufferi.clear(); stack.clear(); stacki.clear(); log_probs.clear(); results.clear(); }

};

class pstate_layer : public layer
{
public:
	Parameters* p_pbias; // parser state bias
	Parameters* p_A; // action lstm to parser state
	Parameters* p_B; // buffer lstm to parser state
	Parameters* p_S; // stack lstm to parser state

	Expression pbias;
	Expression S;
	Expression B;
	Expression A;

	void Init_layer(parser_config& cfg);
	void Init_Graph(ComputationGraph* hg);
	Expression build(parser_state& state);
};


class action_layer : public layer
{
public:
	LookupParameters* p_a; // input action embeddings
	Parameters* p_abias;  // action bias
	Parameters* p_p2a;   // parser state to action

	Expression p2a;
	Expression abias;

	void Init_layer(parser_config& cfg);
	void Init_Graph(ComputationGraph* hg);

	Expression build(Expression pt, vector<unsigned>& current_valid_actions);
	Expression lookup_act(unsigned action) { return  lookup(*m_hg, p_a, action); }
};

class composition_layer : public layer{
public:
	LookupParameters* p_r; // relation embeddings
	Parameters* p_H; // head matrix for composition function
	Parameters* p_D; // dependency matrix for composition function
	Parameters* p_R; // relation matrix for composition function
	Parameters* p_cbias; // composition function bias
	Expression H;
	Expression D;
	Expression R;
	Expression cbias;


	void Init_layer(parser_config& cfg);
	void Init_Graph(ComputationGraph* hg);
	// composed = cbias + H * head + D * dep + R * relation
	Expression build(Expression head, Expression dep, Expression relation);

	Expression lookup_rel(unsigned rel) { return lookup(*m_hg, p_r, rel); }
};


////////////////////////////////////////////////
class dictionary;
class ShiftReduceParser
{
public:
	word_layer			nn_words;
	parser_state		nn_parser;
	pstate_layer		nn_pstate;
	action_layer		nn_actions;
	composition_layer	nn_composition;
	parser_config&		nn_cfg;

	PredictTreeEmbedding nn_pre;

	map<int, Expression> nn_tree2embedding;
public:
	vector<unsigned>	possible_actions;
	vector<string>		setOfActions;
	void build_setOfActions(dictionary* prel2Int);
public:
	ShiftReduceParser(parser_config& cfg);

	vector<pair<int, int>> ShiftReduceParser::log_prob_parser(
		ComputationGraph* hg,
		const vector<int>& raw_sent,  // raw sentence
		const vector<int>& sent,  // sent with oovs replaced
		const vector<int>& sentPos,
		const vector<int>& goldhead,
		const vector<int>& goldrel,
		map<int, vector<int>>&  goldhead2deps,
		sentence_tr& sentinfo,
		bool btrain, double *right, double iter_num);

};

