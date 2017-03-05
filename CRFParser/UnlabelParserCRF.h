#pragma once

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/program_options.hpp>

#include "cnn/training.h"
#include "cnn/cnn.h"
#include "cnn/expr.h"
#include "cnn/nodes.h"
#include "cnn/lstm.h"
#include "cnn/rnn.h"
#include "TreeReaderCRF.h"
#include "layersCRF.h"
#include "Eisner.h"
#include "HelperCRF.h"

namespace CRFParser {
using namespace cnn::expr;
using namespace cnn;
using namespace std;

struct ParserBuilderUnlabel {
	AlignedMemoryPool<8>* base_pool;

	word_layer      wordlayer;
	elements_layer	inputlayer;
	LSTMBuilder* segment_fwrnn;

	Parameters* p_Wo[2];
	Parameters* p_Bo[2];
	Parameters* p_segment_gard[2];
	feature feaarray[6];

	double* temp_scores;
	vector<Expression*> exp_scores;

	LookupParameters* p_dist; // relation embeddings
	LookupParameters* p_r; // relation embeddings
	enum { FEA_MIN, FEA_MAX, FEA_DIS, FEA_S0, FEA_S1, FEA_S2 };

	explicit ParserBuilderUnlabel(parser_config& cfg);

	void evaluate(ComputationGraph* hg, sentence_tr& sen);

	void decode(ComputationGraph* hg, sentence_tr& sen, vector<pair<int, int>>& outsen);

	double construct_grad(ComputationGraph* hg, sentence_tr& sen, vector<pair<int, int>>& outsen);

	//double margin_loss(ComputationGraph* hg, sentence_tr& sen, vector<pair<int, int>>& outsen);
};

}//namespace CRFParser {