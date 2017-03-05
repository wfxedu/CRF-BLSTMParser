#pragma once
#include "../cnn/cnn.h"
#include "../cnn/expr.h"

using namespace cnn::expr;
using namespace std;

namespace cnn {
	//enum { AE_X2I, AE_H2I, AE_X2F, AE_H2F, AE_X2O, AE_H2O, AE_X2L, AE_H2L, AE_X2C, AE_H2C, AE_PREC };
	enum { IN_X2I, IN_H2I, IN_C2I, IN_BI, IN_X2O, IN_H2O, IN_C2O, IN_BO, IN_X2C, IN_H2C, IN_BC};

	struct LSTMState {
		Expression H;
		Expression C;
	};

	class LSTM_Element {
	public:
		LSTM_Element(unsigned input_dim, unsigned hidden_dim, Model* model);
		void new_graph(ComputationGraph& cg);
		Expression add_input(LSTMState& preh, const Expression& x);
		Expression add_input_init(LSTMState& prehc, const Expression& x);
	
		void set_dropout(float d) { dropout_rate = d; }
		// in general, you should disable dropout at test time
		void disable_dropout() { dropout_rate = 0; }
	public:
		std::vector<Parameters*> params;
		std::vector<Expression> param_vars;
		float dropout_rate;
	};



}

