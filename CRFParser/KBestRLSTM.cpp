#include "KBestRLSTM.h"

enum { X2I, H2I, C2I, BI, X2O, H2O, C2O, BO, X2C, H2C, BC };

void LSTMNode::copy(LSTMNode & v)
{
	h.clear(); c.clear(); head.clear();
	h = v.h;
	c = v.c;

	sm = v.sm;
	cur = v.cur;
	head = v.head;
}


void LSTMNode::new_graph_impl(ComputationGraph& cg) 
{

}

// layout: 0..layers = c
//         layers+1..2*layers = h
void LSTMNode::start_new_sequence_impl(const vector<Expression>& hinit) {
	h.clear();
	c.clear();

	h0_org = &kbest_store->h0;
	c0_org = &kbest_store->c0;
	has_initial_state = kbest_store->has_initial_state;
	layers = kbest_store->layers;
	param_vars_org = &kbest_store->param_vars;
}

Expression LSTMNode::add_input_impl(int prev, const Expression& x) {
	std::vector<std::vector<Expression>>& param_vars = *param_vars_org;
	std::vector<Expression>& h0 = *h0_org;
	std::vector<Expression>& c0 = *c0_org;
	//--------------------------------------------------
	h.push_back(vector<Expression>(layers));
	c.push_back(vector<Expression>(layers));
	vector<Expression>& ht = h.back();
	vector<Expression>& ct = c.back();
	Expression in = x;
	for (unsigned i = 0; i < layers; ++i) {
		const vector<Expression>& vars = param_vars[i];
		Expression i_h_tm1, i_c_tm1;
		bool has_prev_state = (prev >= 0 || has_initial_state);
		if (prev < 0) {
			if (has_initial_state) {
				// intial value for h and c at timestep 0 in layer i
				// defaults to zero matrix input if not set in add_parameter_edges
				i_h_tm1 = h0[i];
				i_c_tm1 = c0[i];
			}
		}
		else {  // t > 0
			i_h_tm1 = h[prev][i];
			i_c_tm1 = c[prev][i];
		}
		// input
		Expression i_ait;
		if (has_prev_state)
			//      i_ait = vars[BI] + vars[X2I] * in + vars[H2I]*i_h_tm1 + vars[C2I] * i_c_tm1;
			i_ait = affine_transform({ vars[BI], vars[X2I], in, vars[H2I], i_h_tm1, vars[C2I], i_c_tm1 });
		else
			//      i_ait = vars[BI] + vars[X2I] * in;
			i_ait = affine_transform({ vars[BI], vars[X2I], in });
		Expression i_it = logistic(i_ait);
		// forget
		Expression i_ft = 1.f - i_it;
		// write memory cell
		Expression i_awt;
		if (has_prev_state)
			//      i_awt = vars[BC] + vars[X2C] * in + vars[H2C]*i_h_tm1;
			i_awt = affine_transform({ vars[BC], vars[X2C], in, vars[H2C], i_h_tm1 });
		else
			//      i_awt = vars[BC] + vars[X2C] * in;
			i_awt = affine_transform({ vars[BC], vars[X2C], in });
		Expression i_wt = tanh(i_awt);
		// output
		if (has_prev_state) {
			Expression i_nwt = cwise_multiply(i_it, i_wt);
			Expression i_crt = cwise_multiply(i_ft, i_c_tm1);
			ct[i] = i_crt + i_nwt;
		}
		else {
			ct[i] = cwise_multiply(i_it, i_wt);
		}

		Expression i_aot;
		if (has_prev_state)
			//      i_aot = vars[BO] + vars[X2O] * in + vars[H2O] * i_h_tm1 + vars[C2O] * ct[i];
			i_aot = affine_transform({ vars[BO], vars[X2O], in, vars[H2O], i_h_tm1, vars[C2O], ct[i] });
		else
			//      i_aot = vars[BO] + vars[X2O] * in;
			i_aot = affine_transform({ vars[BO], vars[X2O], in });
		Expression i_ot = logistic(i_aot);
		Expression ph_t = tanh(ct[i]);
		in = ht[i] = cwise_multiply(i_ot, ph_t);
	}
	return ht.back();
}

/////////////////////////////////////////////////////////////
KBestRLSTM::KBestRLSTM(unsigned layers,
	unsigned input_dim,
	unsigned hidden_dim,
	Model* model, unsigned kbest) : layers(layers)
{
	unsigned layer_input_dim = input_dim;
	for (unsigned i = 0; i < layers; ++i) {
		// i
		Parameters* p_x2i = model->add_parameters({ (long)hidden_dim, (long)layer_input_dim });
		Parameters* p_h2i = model->add_parameters({ (long)hidden_dim, (long)hidden_dim });
		Parameters* p_c2i = model->add_parameters({ (long)hidden_dim, (long)hidden_dim });
		Parameters* p_bi = model->add_parameters({ (long)hidden_dim });

		// o
		Parameters* p_x2o = model->add_parameters({ (long)hidden_dim, (long)layer_input_dim });
		Parameters* p_h2o = model->add_parameters({ (long)hidden_dim, (long)hidden_dim });
		Parameters* p_c2o = model->add_parameters({ (long)hidden_dim, (long)hidden_dim });
		Parameters* p_bo = model->add_parameters({ (long)hidden_dim });

		// c
		Parameters* p_x2c = model->add_parameters({ (long)hidden_dim, (long)layer_input_dim });
		Parameters* p_h2c = model->add_parameters({ (long)hidden_dim, (long)hidden_dim });
		Parameters* p_bc = model->add_parameters({ (long)hidden_dim });
		layer_input_dim = hidden_dim;  // output (hidden) from 1st layer is input to next

		vector<Parameters*> ps = { p_x2i, p_h2i, p_c2i, p_bi, p_x2o, p_h2o, p_c2o, p_bo, p_x2c, p_h2c, p_bc };
		params.push_back(ps);
	}  // layers

	for (int i = 0; i < kbest; i++) {
		LSTMNode* nd = new LSTMNode();
		nd->kbest_store = this;
		nd->layers = layers;
		lstm_nodes.push_back(nd);
	}
}

void KBestRLSTM::new_graph(ComputationGraph& cg){
	param_vars.clear();

	for (unsigned i = 0; i < layers; ++i){
		auto& p = params[i];

		//i
		Expression i_x2i = parameter(cg, p[X2I]);
		Expression i_h2i = parameter(cg, p[H2I]);
		Expression i_c2i = parameter(cg, p[C2I]);
		Expression i_bi = parameter(cg, p[BI]);
		//o
		Expression i_x2o = parameter(cg, p[X2O]);
		Expression i_h2o = parameter(cg, p[H2O]);
		Expression i_c2o = parameter(cg, p[C2O]);
		Expression i_bo = parameter(cg, p[BO]);
		//c
		Expression i_x2c = parameter(cg, p[X2C]);
		Expression i_h2c = parameter(cg, p[H2C]);
		Expression i_bc = parameter(cg, p[BC]);

		vector<Expression> vars = { i_x2i, i_h2i, i_c2i, i_bi, i_x2o, i_h2o, i_c2o, i_bo, i_x2c, i_h2c, i_bc };
		param_vars.push_back(vars);
	}

	for (int i = 0; i < lstm_nodes.size(); i++) {
		lstm_nodes[i]->kbest_store = this;
		lstm_nodes[i]->new_graph(cg);
	}
}

void KBestRLSTM::start_new_sequence(const std::vector<Expression>& hinit) {
	if (hinit.size() > 0) {
		assert(layers * 2 == hinit.size());
		h0.resize(layers);
		c0.resize(layers);
		for (unsigned i = 0; i < layers; ++i) {
			c0[i] = hinit[i];
			h0[i] = hinit[i + layers];
		}
		has_initial_state = true;
	}
	else {
		has_initial_state = false;
	}

	for (int i = 0; i < lstm_nodes.size(); i++) {
		lstm_nodes[i]->start_new_sequence(hinit);
	}
}

Expression KBestRLSTM::add_input(const Expression& x, int ibs) {
	return lstm_nodes[ibs]->add_input(x);
}

Expression KBestRLSTM::add_input(const RNNPointer& prev, const Expression& x, int ibs)
{
	return lstm_nodes[ibs]->add_input(prev,x);
}

void KBestRLSTM::add_input_all(const Expression& x)
{
	for (int i = 0; i < lstm_nodes.size(); i++) {
		lstm_nodes[i]->add_input(x);
	}
}

void KBestRLSTM::add_input_all(const RNNPointer& prev, const Expression& x)
{
	for (int i = 0; i < lstm_nodes.size(); i++) {
		lstm_nodes[i]->add_input(prev,x);
	}
}

void KBestRLSTM::copy(unsigned from, unsigned to)
{
	lstm_nodes[to]->copy(*lstm_nodes[from]);
}