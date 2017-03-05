#pragma once
#include "cnn/training.h"
#include "cnn/cnn.h"
#include "cnn/expr.h"
#include "cnn/nodes.h"
#include "cnn/lstm.h"
#include "cnn/rnn.h"
#include "TreeReader.h"
#include "SRParser.h"
#include "util.h"

class ParserTester
{
public:
	EmbedDict&								pretrained;
	DatasetTB&									sr_corpus;
	DatasetTB&								pred_corpus;

	set<unsigned>							training_vocab;
	Model									model;
	parser_config&							nn_config;
	ShiftReduceParser*						sr_parser;
public:
	ParserTester(parser_config& cfg, DatasetTB& cpr, DatasetTB& pred_cpr, EmbedDict& edict);
	~ParserTester();

	void InitParser();

	void LoadModel(char* path);

	void test(double unk_prob, unsigned unk_strategy, string outf);
};

