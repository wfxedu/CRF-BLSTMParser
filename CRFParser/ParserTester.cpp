#include "ParserTester.h"
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

ParserTester::ParserTester(parser_config& cfg, DatasetTB& cpr, DatasetTB& pred_cpr, EmbedDict& edict) :
nn_config(cfg), sr_corpus(cpr), pred_corpus(pred_cpr), pretrained(edict)
{
	nn_config.model = &model;
	nn_config.pretrained = &pretrained;
	sr_parser = new ShiftReduceParser(nn_config);
}


ParserTester::~ParserTester()
{
	delete sr_parser;
}

void ParserTester::InitParser()
{
	{  // compute the singletons in the parser's training data
		map<unsigned, unsigned> counts;
		for (auto sent : sr_corpus.training.sentences1)
			for (auto word : sent.raw_sent) { training_vocab.insert(word); counts[word]++; }
	}
	sr_parser->build_setOfActions(&sr_corpus.prel2Int);
}


void ParserTester::LoadModel(char* path)
{
	ifstream in(path);
	boost::archive::text_iarchive ia(in);
	ia >> model;
}



void output_conll_loc(
	const vector<int>& sentence, 
	const vector<int>& pos,
	const map<unsigned, string>& intToWords,
	const map<unsigned, string>& intToPos,
	const map<unsigned, string>& intToRel,
	vector<pair<int, int>>& pred, 
	ofstream& outTxt)
{
	for (unsigned i = 0; i < sentence.size()-1 ; ++i) {
		auto index = i;
		
		string wit = intToWords.find(sentence[i])->second;
		auto pit = intToPos.find(pos[i]);
		auto hyp_rel = intToRel.find(pred[i].second);
		int hyp_head = pred[i].first;
		if (hyp_head >= sentence.size()-1)
			hyp_head = 0;
		else
			hyp_head++;
		outTxt << index << '\t'       // 1. ID 
			<< wit << '\t'         // 2. FORM
			<< "_" << '\t'         // 3. LEMMA 
			<< "_" << '\t'         // 4. CPOSTAG 
			<< pit->second << '\t' // 5. POSTAG
			<< "_" << '\t'         // 6. FEATS
			<< hyp_head << '\t'    // 7. HEAD
			<< hyp_rel->second << '\t'     // 8. DEPREL
			<< "_" << '\t'         // 9. PHEAD
			<< "_" << endl;        // 10. PDEPREL
	}
	outTxt << endl;
}

void output_embed_loc(const vector<int>& sentence, map<int, Expression>& tree2embedding, 
	ofstream& outEm)
{
	for (unsigned i = 0; i < (sentence.size() - 1); ++i) {
		map<int, Expression>::iterator iter = tree2embedding.find(i);
		if (iter == tree2embedding.end())
			continue;
		vector<float> embed = as_vector(iter->second.value());
		outEm << i+1 << "\t";
		for (int j = 0; j < embed.size(); j++)
			outEm << embed[j] << "\t";
		outEm << endl;
	}
	outEm << endl;
}

void output_inner_embedding(const vector<int>& sentence,
	vector<Expression>& word2inner_mean, unsigned unk_tok, ofstream& out_inner)
{
	for (unsigned i = 0; i < (sentence.size() - 1); ++i) {
		vector<float> embed = as_vector(word2inner_mean[i].value());

		if (sentence[i] != unk_tok)
			out_inner << i + 1 << "\tN\t";
		else
			out_inner << i + 1 << "\tE\t";

		for (int j = 0; j < embed.size(); j++)
			out_inner << embed[j] << "\t";
		out_inner << endl;
	}
	out_inner << endl;
}


void ParserTester::test(double unk_prob, unsigned unk_strategy, string outf)
{
	ofstream outTxt(outf);
	//ofstream outEm(outf+".em");
	//ofstream out_inner(outf + ".inner");
	bool eval_sc = true;

	const unsigned kUNK = sr_corpus.words2Int.get(treebank::UNK, 0);
	double right = 0;
	double correct_heads = 0;
	double total_heads = 0;
	auto t_start = std::chrono::high_resolution_clock::now();
	unsigned corpus_size = sr_corpus.testing.sentences.size();
	for (unsigned sii = 0; sii < corpus_size; ++sii) {
		cerr << sii << ", ";
		sentence_item& sen_item = sr_corpus.testing.sentences1[sii];
		sentence_tr& sen_tr = pred_corpus.testing.sentences[sii];
		const vector<int>& sentence = sen_item.raw_sent;
		const vector<int>& sentencePos = sen_item.sentPos;

		vector<int> tsentence = sentence;
		for (auto& w : tsentence)
			if (training_vocab.count(w) == 0) w = kUNK;
		for (int wx = 0;wx < tsentence.size();wx++)
			sen_tr[wx].form_useid = tsentence[wx];
		ComputationGraph cg;
		double lp = 0;
		vector<pair<int, int>> pred = sr_parser->log_prob_parser(
			&cg,
			sentence,
			tsentence,
			sen_item.sentPos,
			sen_item.goldhead,
			sen_item.goldrel,
			sen_item.goldhead2deps,
			sen_tr,
			false,
			&right,0);
		
		for (int iw = 1;iw < pred.size();iw++) {
			if (pred[iw].first == sen_item.goldhead[iw])
				correct_heads++;
			total_heads++;

		}
		
		output_conll_loc(sentence, sentencePos,
			sr_corpus.words2Int.int2Tokens,
			sr_corpus.pos2Int.int2Tokens, 
			sr_corpus.prel2Int.int2Tokens,
			pred,
			outTxt);
		//output_embed_loc(sentence, sr_parser->nn_tree2embedding, outEm);
		//output_inner_embedding(sentence, sr_parser->nn_parser.word2inner_mean, kUNK, out_inner);

	}
	cerr << endl;
	if (eval_sc) {
		auto t_end = std::chrono::high_resolution_clock::now();
		cerr << "TEST llh=" << " uas: " << (correct_heads / total_heads) 
			<< "\t[" << corpus_size << " sents in " 
			<< std::chrono::duration<double, std::milli>(t_end - t_start).count() 
			<< " ms]" << endl;
	}
	else {
		auto t_end = std::chrono::high_resolution_clock::now();
		cerr << "\t[" << corpus_size << " sents in " 
			<< std::chrono::duration<double, std::milli>(t_end - t_start).count() 
			<< " ms]" << endl;
	}
}