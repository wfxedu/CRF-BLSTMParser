#pragma once
#include "common.h"

class KBestRLSTM;

struct LSTMNode : public RNNBuilder {
	LSTMNode() = default;

	Expression back() const override { return (cur == -1 ? h0_org->back() : h[cur].back()); }

	std::vector<Expression> final_h() const override { return (h.size() == 0 ? *h0_org : h.back()); }
	std::vector<Expression> final_s() const override {
		std::vector<Expression> ret = (c.size() == 0 ? *c0_org : c.back());
		for (auto my_h : final_h()) ret.push_back(my_h);
		return ret;
	}
	unsigned num_h0_components() const override { return 2 * layers; }

	std::vector<Expression> get_h(RNNPointer i) const { return (i == -1 ? *h0_org : h[i]); }
	std::vector<Expression> get_s(RNNPointer i) const {
		std::vector<Expression> ret = (i == -1 ? *c0_org : c[i]);
		for (auto my_h : get_h(i)) ret.push_back(my_h);
		return ret;
	}
public:
	void copy(const RNNBuilder & params) override {};
	void copy(LSTMNode & v) ;
protected:
	void new_graph_impl(ComputationGraph& cg) override;
	void start_new_sequence_impl(const std::vector<Expression>& h0) override;
	Expression add_input_impl(int prev, const Expression& x) override;
public:
	KBestRLSTM* kbest_store;
	// first index is layer, then ...
	std::vector<std::vector<Expression>>* param_vars_org;
	// first index is time, second is layer
	std::vector<std::vector<Expression>> h, c;
	// initial values of h and c at each layer
	// - both default to zero matrix input
	bool has_initial_state; // if this is false, treat h0 and c0 as 0
	std::vector<Expression>* h0_org;
	std::vector<Expression>* c0_org;
	unsigned layers;
};

struct KBestRLSTM {
	KBestRLSTM() = default;
	explicit KBestRLSTM(unsigned layers,
		unsigned input_dim,
		unsigned hidden_dim,
		Model* model, unsigned kbest);
public:
	Expression back(int ibs) const { return lstm_nodes[ibs]->back(); }
	RNNPointer state(int ibs) const{ return lstm_nodes[ibs]->state();}
	//RNNPointer pre_state(RNNPointer lcur, int ibs) const { return lstm_nodes[ibs]->pre_state(lcur); 	}
	
	void new_graph(ComputationGraph& cg);
	void start_new_sequence(const std::vector<Expression>& hinit = {});

	Expression add_input(const Expression& x, int ibs);
	Expression add_input(const RNNPointer& prev, const Expression& x, int ibs);
	void add_input_all(const Expression& x);
	void add_input_all(const RNNPointer& prev, const Expression& x);
	void copy(unsigned from, unsigned to);

	void rewind_one_step(int ibs) {	lstm_nodes[ibs]->rewind_one_step();	}
public:
	std::vector<LSTMNode*>				lstm_nodes;
	// first index is layer, then ...
	std::vector<std::vector<Parameters*>> params;
	// first index is layer, then ...
	std::vector<std::vector<Expression>> param_vars;
	// - both default to zero matrix input
	bool has_initial_state; // if this is false, treat h0 and c0 as 0
	std::vector<Expression> h0;
	std::vector<Expression> c0;
	unsigned layers;
};
