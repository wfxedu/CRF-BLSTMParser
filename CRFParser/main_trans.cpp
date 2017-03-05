#include <cstdlib>
#include <algorithm>
#include <sstream>
#include <iostream>
#include <vector>
#include <limits>
#include <cmath>
#include <chrono>
#include <ctime>
#include <unordered_map>
#include <unordered_set>
#ifdef WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif
#include "execinfo.h"

#include <signal.h>

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

#include "cnn/training.h"
#include "cnn/cnn.h"
#include "cnn/expr.h"
#include "cnn/nodes.h"
#include "cnn/lstm.h"
#include "cnn/rnn.h"
#include "TreeReader.h"
#include "SRParser.h"
#include "util.h"
#include "ParserTrainer.h"
#include "ParserTester.h"

#ifdef WIN32
#include <process.h>
#define getpid _getpid
#endif


using namespace cnn::expr;
using namespace cnn;
using namespace std;

EmbedDict								pretrained;
DatasetTB									sr_corpus;
DatasetTB									pred_corpus;

void fill_cfg() {
	parser_config::LAYERS = 2;
	parser_config::INPUT_DIM = 100;
	parser_config::HIDDEN_DIM = 100;
	parser_config::ACTION_DIM = 20;
	parser_config::PRETRAINED_DIM = 50;
	parser_config::LSTM_INPUT_DIM = 100;
	parser_config::POS_DIM = 12;
	parser_config::REL_DIM = 20;
	parser_config::USE_POS = true;

	parser_config::ROOT_SYMBOL = "ROOT";
	parser_config::kROOT_SYMBOL = 0;
	parser_config::ACTION_SIZE = 0;
	parser_config::VOCAB_SIZE = 0;
	parser_config::POS_SIZE = 0;
}

int main(int argc, char** argv) {
	cnn::Initialize(argc, argv);
	float unk_prob = 0.2f;
	bool train_mode = false;
	char* training_data = 0;
	char* dev_data = 0;
	char* test_data = 0;
	char* training_preddata = 0;
	char* dev_preddata = 0;
	char* test_preddata = 0;

	bool embedding_type = false;
	char* words_embedding = 0;
	int embedding_size = 0;
	char* model_path = 0;
	char* out_model_path = 0;
	char* test_out = 0;
	for (int i = 0; i < argc; i++) {
		if (!strcmp(argv[i], "-unk_p")) {
			unk_prob = atof(argv[i + 1]); i++;
		}
		if (!strcmp(argv[i], "-train")) {
			training_data = argv[i + 1]; i++;
		}
		if (!strcmp(argv[i], "-train_pred")) {
			training_preddata = argv[i + 1]; i++;
		}
		if (!strcmp(argv[i], "-tm")) {
			train_mode = true; 
		}
		if (!strcmp(argv[i], "-em_org")) {
			embedding_type = true;
		}
		if (!strcmp(argv[i], "-em_path")) {
			words_embedding = argv[i + 1]; i++;
		}
		if (!strcmp(argv[i], "-em_sz")) {
			embedding_size = atoi(argv[i + 1]); i++;
		}
		if (!strcmp(argv[i], "-kbest")) {
			parser_config::KBEST = atoi(argv[i + 1]); i++;
			printf("setting kbest=%d\n", parser_config::KBEST);
		}
		if (!strcmp(argv[i], "-model")) {
			model_path = argv[i + 1]; i++;
		}
		if (!strcmp(argv[i], "-out_model_path")) {
			out_model_path = argv[i + 1]; i++;
		}
		if (!strcmp(argv[i], "-dev")) {
			dev_data = argv[i + 1]; i++;
		}
		if (!strcmp(argv[i], "-dev_pred")) {
			dev_preddata = argv[i + 1]; i++;
		}
		if (!strcmp(argv[i], "-test")) {
			test_data = argv[i + 1]; i++;
		}
		if (!strcmp(argv[i], "-test_pred")) {
			test_preddata = argv[i + 1]; i++;
		}
		if (!strcmp(argv[i], "-test_out")) {
			test_out = argv[i + 1]; i++;
		}
	}
	/////////////////////////////////////////////////////////////////////////////////////////////
	fill_cfg();
	unsigned unk_strategy = 1; //only one way "STOCHASTIC REPLACEMENT"
	assert(unk_prob >= 0.); assert(unk_prob <= 1.);
	//------------------------------------------------------
	if (train_mode) {
		cerr << "loading training data..." << endl;
		sr_corpus.load_train (training_data);
		pred_corpus.load_train(training_preddata);
		//sr_corpus.save_config("corpus_cfg.ini");
		cerr << "finished!" << endl;
	}
	else {
		//sr_corpus.load_config("corpus_cfg.ini");
		cerr << "loading training data..." << endl;
		sr_corpus.load_train(training_data);
		pred_corpus.load_train(training_preddata);
	}
	//------------------------------------------------------
	EmbedDict pretrained;
	if (words_embedding) {
		parser_config::PRETRAINED_DIM = embedding_size;
		cerr << "loading EmbedDict..." << endl;
		util::load_EmbedDict(sr_corpus, pretrained, words_embedding, embedding_type);
		cerr << "finished!" << endl;
	}
	parser_config nn_cfg;
	//------------------------------------------------------
	cerr << "Words: " << sr_corpus.words2Int.nTokens << endl;
	parser_config::VOCAB_SIZE = sr_corpus.words2Int.nTokens + 1;
	parser_config::ACTION_SIZE = sr_corpus.prel2Int.nTokens*2 + 6;
	parser_config::POS_SIZE = sr_corpus.pos2Int.nTokens + 10; // bad way of dealing with the fact that we may see new POS tags in the test set
	//------------------------------------------------------
	if (train_mode) {
		const string oname = out_model_path;
		cerr << "out model name: " << oname << endl;

		ParserTrainer trainer(nn_cfg, sr_corpus,pred_corpus, pretrained);
		if (model_path) {
			cerr << "load model: " << model_path << endl;
			trainer.LoadModel(model_path);
			cerr << "finished loading model" << endl;
		}
		trainer.InitParser();
		sr_corpus.load_dev(dev_data);
		pred_corpus.load_dev(dev_preddata);
		trainer.train(unk_prob, unk_strategy, oname);
	}
	else 
	{
		if (test_data) {
			sr_corpus.load_test(test_data);
			pred_corpus.load_test(test_preddata);
		}
		else {
			sr_corpus.load_dev(dev_data);
			pred_corpus.load_dev(dev_preddata);
		}
		ParserTester tester(nn_cfg, sr_corpus, pred_corpus, pretrained);
		if (model_path) {
			cerr << "load model: " << model_path << endl;
			tester.LoadModel(model_path);
			cerr << "finished loading model" << endl;
		}
		tester.InitParser();

		string tout = test_out;
		tester.test(unk_prob, unk_strategy, tout);
	}
}
