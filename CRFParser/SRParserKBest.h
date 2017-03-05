#pragma once
#include "common.h"
#include "SRParser.h"
#include "KBestRLSTM.h"

struct parser_store {
	vector<Expression> buffer;
	vector<int> bufferi;

	vector<Expression> stack; 
	vector<int> stacki; 

	vector<Expression> log_probs;
	vector<unsigned> results;

	void clear() { buffer.clear(); bufferi.clear(); stack.clear(); stacki.clear(); log_probs.clear(); results.clear(); }
	void accum(Expression action_scores, int idx) { log_probs.push_back(pick(action_scores, idx)); results.push_back(idx); }
	Expression log_p_total() { return -sum(log_probs); }

	void copy(parser_store& ps) { 
		buffer = ps.buffer; bufferi = ps.bufferi; stack = ps.stack; 
		stacki = ps.stacki; log_probs = ps.log_probs; results = ps.results; 
	}
};

class parser_stateKBest : public layer 
{
public:
	KBestRLSTM stack_lstm; // (layers, input, hidden, trainer)
	KBestRLSTM buffer_lstm;
	KBestRLSTM action_lstm;

	Parameters* p_action_start;  // action bias
	Parameters* p_buffer_guard;  // end of buffer
	Parameters* p_stack_guard;  // end of stack

	Expression action_start;
public:
	std::vector<parser_store> kbest_stores;
public:
	void Init_layer(parser_config& cfg);
	void Init_Graph(ComputationGraph* hg);
	void Init_state(const t_unvec& raw_sent, const t_unvec& sent, const t_unvec& sentPos, word_layer& layer);
public:
	void accum(Expression action_scores, int idx, int kbi) { kbest_stores[kbi].accum(action_scores, idx); }
	Expression log_p_total(int kbi) { return kbest_stores[kbi].log_p_total(); }
public:
	void clear() { for (int i = 0; i < kbest_stores.size(); i++) kbest_stores[i].clear(); }
};
////////////////////////////////////////////////
class pstate_layerKBest : public layer
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
	Expression build(parser_stateKBest& state, int kbi);
};
////////////////////////////////////////////////
class ShiftReduceParserKBest
{
public:
	word_layer			nn_words;
	parser_stateKBest	nn_parser;
	pstate_layerKBest	nn_pstate;
	action_layer		nn_actions;
	composition_layer	nn_composition;
	parser_config&		nn_cfg;

	map<int, Expression> nn_tree2embedding;
public:
	vector<unsigned>	possible_actions;
public:
	ShiftReduceParserKBest(parser_config& cfg);

	void take_action(score_pair& act, const vector<string>& setOfActions, unsigned kbi);

	vector<unsigned> log_prob_parser(ComputationGraph* hg,
		const vector<unsigned>& raw_sent,  // raw sentence
		const vector<unsigned>& sent,  // sent with oovs replaced
		const vector<unsigned>& sentPos,
		const vector<unsigned>& correct_actions,
		const vector<string>& setOfActions,
		const map<unsigned, std::string>& intToWords,
		double *right,int KBEST=parser_config::KBEST);
};

///////////////////////////////////////////////

