#pragma once
#include "cnn/training.h"
#include "cnn/cnn.h"
#include "cnn/expr.h"
#include "cnn/nodes.h"
#include "cnn/lstm.h"
#include "cnn/rnn.h"
//#include "Corpus.h"
#include "TreeReader.h"
#include "SRParser.h"
#include "util.h"

class ParserTrainer
{
public:
	EmbedDict&								pretrained;
	DatasetTB&								sr_corpus;
	DatasetTB&								pred_corpus;

	set<unsigned>							training_vocab;
	set<unsigned>							singletons;
	Model									model;
	parser_config&							nn_config;
	ShiftReduceParser*						sr_parser;

	enum UNK_STRATEGY{
		ENUM_NONE = 0,
		ENUM_STOCHASTIC_REPLACEMENT = 1
	};
public:
	ParserTrainer(parser_config& cfg, DatasetTB& cpr, DatasetTB& pred_cpr, EmbedDict& edict);
	~ParserTrainer();

	void InitParser();

	void LoadModel(char* path);

	void SaveModel(const char* path);

	void train(double unk_prob, unsigned unk_strategy, string oname);


};

