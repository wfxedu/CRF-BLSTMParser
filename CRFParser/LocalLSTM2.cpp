#include "LocalLSTM2.h"
using namespace cnn;

LSTM_Element::LSTM_Element(unsigned input_dim, unsigned hidden_dim, Model* model) {
	// i
	Parameters* p_x2i = model->add_parameters({ (long)hidden_dim, (long)input_dim });
	Parameters* p_h2i = model->add_parameters({ (long)hidden_dim, (long)hidden_dim });
	Parameters* p_c2i = model->add_parameters({ (long)hidden_dim, (long)hidden_dim });
	Parameters* p_bi = model->add_parameters({ (long)hidden_dim });
	//p_bi->need_regression = false;
	// o
	Parameters* p_x2o = model->add_parameters({ (long)hidden_dim, (long)input_dim });
	Parameters* p_h2o = model->add_parameters({ (long)hidden_dim, (long)hidden_dim });
	Parameters* p_c2o = model->add_parameters({ (long)hidden_dim, (long)hidden_dim });
	Parameters* p_bo = model->add_parameters({ (long)hidden_dim });
	//p_bo->need_regression = false;
	// c
	Parameters* p_x2c = model->add_parameters({ (long)hidden_dim, (long)input_dim });
	Parameters* p_h2c = model->add_parameters({ (long)hidden_dim, (long)hidden_dim });
	Parameters* p_bc = model->add_parameters({ (long)hidden_dim });
	//p_bc->need_regression = false;

	params = { p_x2i, p_h2i, p_c2i, p_bi, p_x2o, p_h2o, p_c2o, p_bo, p_x2c, p_h2c, p_bc };
	dropout_rate = 0.0f;
}

void LSTM_Element::new_graph(ComputationGraph& cg) {
	param_vars.clear();

	auto& p = params;

	Expression i_x2i = parameter(cg, p[IN_X2I]);
	Expression i_h2i = parameter(cg, p[IN_H2I]);
	Expression i_c2i = parameter(cg, p[IN_C2I]);
	Expression i_bi = parameter(cg, p[IN_BI]);
	//o
	Expression i_x2o = parameter(cg, p[IN_X2O]);
	Expression i_h2o = parameter(cg, p[IN_H2O]);
	Expression i_c2o = parameter(cg, p[IN_C2O]);
	Expression i_bo = parameter(cg, p[IN_BO]);
	//c
	Expression i_x2c = parameter(cg, p[IN_X2C]);
	Expression i_h2c = parameter(cg, p[IN_H2C]);
	Expression i_bc = parameter(cg, p[IN_BC]);

	param_vars = { i_x2i, i_h2i, i_c2i, i_bi, i_x2o, i_h2o, i_c2o, i_bo, i_x2c, i_h2c, i_bc };
}

Expression LSTM_Element::add_input(LSTMState& prehc, const Expression& x) {
	const vector<Expression>& vars = param_vars;
	Expression i_h_tm1 = prehc.H, i_c_tm1 = prehc.C;

	Expression in = x;
	// apply dropout according to http://arxiv.org/pdf/1409.2329v5.pdf
	if (dropout_rate) in = dropout(in, dropout_rate);
	// input
	Expression i_ait = affine_transform(
	{ vars[IN_BI], vars[IN_X2I], in, vars[IN_H2I], i_h_tm1, vars[IN_C2I], i_c_tm1 }
	);
	Expression i_it = logistic(i_ait);
	// forget
	Expression i_ft = 1.f - i_it;
	// write memory cell
	Expression i_awt = affine_transform(
	{ vars[IN_BC], vars[IN_X2C], in, vars[IN_H2C], i_h_tm1 }
	);
	Expression i_wt = tanh(i_awt);
	Expression i_nwt = cwise_multiply(i_it, i_wt);
	Expression i_crt = cwise_multiply(i_ft, i_c_tm1);
	prehc.C = i_crt + i_nwt; //C

	Expression i_aot = affine_transform(
	{ vars[IN_BO], vars[IN_X2O], in, vars[IN_H2O], i_h_tm1, vars[IN_C2O], prehc.C }
	);
	Expression i_ot = logistic(i_aot);
	Expression ph_t = tanh(prehc.C);
	prehc.H = cwise_multiply(i_ot, ph_t); //H

	if (dropout_rate) return dropout(prehc.H, dropout_rate);
	else return prehc.H;
	//return prehc.H; //H
}

Expression LSTM_Element::add_input_init(LSTMState& prehc, const Expression& x) {
	const vector<Expression>& vars = param_vars;
	Expression in = x;
	// apply dropout according to http://arxiv.org/pdf/1409.2329v5.pdf
	if (dropout_rate) in = dropout(in, dropout_rate);
	// input
	Expression i_ait = affine_transform({ vars[IN_BI], vars[IN_X2I], in });
	Expression i_it = logistic(i_ait);
	// forget
	//Expression i_ft = 1.f - i_it;
	// write memory cell
	Expression i_awt = affine_transform({ vars[IN_BC], vars[IN_X2C], in });
	Expression i_wt = tanh(i_awt);
	// output
	prehc.C = cwise_multiply(i_it, i_wt); //C

	Expression i_aot = affine_transform({ vars[IN_BO], vars[IN_X2O], in });
	Expression i_ot = logistic(i_aot);
	Expression ph_t = tanh(prehc.C);
	prehc.H = cwise_multiply(i_ot, ph_t); //H

	if (dropout_rate) return dropout(prehc.H, dropout_rate);
	else return prehc.H;
	//return prehc.H;
}