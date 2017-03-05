#include "ParserTrainer.h"
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

char* loc_asctime(const struct tm *timeptr)
{
	static char result[26];
	sprintf(result, " %.2d:%.2d:%.2d ",
		timeptr->tm_hour, timeptr->tm_min, timeptr->tm_sec);
	return result;
}

ParserTrainer::ParserTrainer(parser_config& cfg, DatasetTB& cpr, DatasetTB& pred_cpr, EmbedDict& edict) :
nn_config(cfg), sr_corpus(cpr), pred_corpus(pred_cpr), pretrained(edict)
{
	nn_config.model = &model;
	nn_config.pretrained = &pretrained;
	sr_parser = new ShiftReduceParser(nn_config);
}

ParserTrainer::~ParserTrainer()
{
	delete sr_parser;
}

void ParserTrainer::InitParser()
{
	{  // compute the singletons in the parser's training data
		map<unsigned, unsigned> counts;
		for (auto sent : sr_corpus.training.sentences1)
			for (auto word : sent.raw_sent) { training_vocab.insert(word); counts[word]++; }
		for (auto wc : counts)
			if (wc.second == 1) singletons.insert(wc.first);
	}
	sr_parser->build_setOfActions(&sr_corpus.prel2Int);
}

void ParserTrainer::LoadModel(char* path)
{
	ifstream in(path);
	boost::archive::text_iarchive ia(in);
	ia >> model;
}

void ParserTrainer::SaveModel(const char* path)
{
	ofstream out(path);
	boost::archive::text_oarchive oa(out);
	oa << model;
}

void ParserTrainer::train(double unk_prob, unsigned unk_strategy, string oname)
{
	unsigned status_every_i_iterations = 100;
	const unsigned kUNK = sr_corpus.words2Int.get(treebank::UNK,0);
	int best_correct_heads = 0, best_correct_heads1 = 0;

	SimpleSGDTrainer sgd(&model);
	//MomentumSGDTrainer sgd(&model);
	sgd.eta_decay = 0.08;
	//sgd.eta_decay = 0.05;
	vector<unsigned> order(sr_corpus.training.sentences.size());
	for (unsigned i = 0; i < sr_corpus.training.sentences.size(); ++i)
		order[i] = i;
	double tot_seen = 0;
	status_every_i_iterations = 
		(status_every_i_iterations> sr_corpus.training.sentences.size())?
		sr_corpus.training.sentences.size(): status_every_i_iterations;
	unsigned si = sr_corpus.training.sentences.size();
	cerr << "Training Sentence: " << sr_corpus.training.sentences.size() << endl;
	unsigned trs = 0;
	double right = 0;
	double llh = 0;
	bool first = true;
	int iter = -1;
	time_t time_start = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
	cerr << "Time: " << loc_asctime(localtime(&time_start)) << endl;
	FILE* logO = fopen(((string)oname + ".log").c_str(), "w");
	while (true) {
		++iter;
		for (unsigned sii = 0; sii < status_every_i_iterations; ++sii) {
			if (si == sr_corpus.training.sentences.size()) {
				si = 0;
				if (first)
					first = false; 
				else 
					sgd.update_epoch(); 
				cerr << "**SHUFFLE\n";
				random_shuffle(order.begin(), order.end());
			}

			tot_seen += 1;
			sentence_item& sen_item = sr_corpus.training.sentences1[order[si]];
			sentence_tr& sentr = pred_corpus.training.sentences[order[si]];
			const vector<int>& sentence = sen_item.raw_sent;;
			vector<int> tsentence = sentence;
			if (unk_strategy == ENUM_STOCHASTIC_REPLACEMENT) {
				for (auto& w : tsentence)
					if (singletons.count(w) && cnn::rand01() < unk_prob) w = kUNK;
			}
			for (int wx = 0;wx < tsentence.size();wx++)
				sentr[wx].form_useid = tsentence[wx];
			int iter_num = (tot_seen / sr_corpus.training.sentences.size());
			ComputationGraph hg;
			sr_parser->log_prob_parser(
				&hg,
				sentence,
				tsentence, 
				sen_item.sentPos,
				sen_item.goldhead,
				sen_item.goldrel,
				sen_item.goldhead2deps,
				sentr,
				true, &right, iter_num);
			double lp = as_scalar(hg.incremental_forward());
			if (lp > 1e8) {
				++si;
				continue;
			}
			if (lp < 0) {
				cerr << "Log prob < 0 on sentence " << order[si] << ": lp=" << lp << endl;
				assert(lp >= 0.0);
			}
			hg.backward();
			sgd.update(1.0);
			llh += lp;
			++si;
			trs += sr_parser->nn_parser.results.size();;
		}
		//sgd.status();
		time_t time_now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
		cerr << "update #" << iter << " (epoch " << (tot_seen / sr_corpus.training.sentences.size())
			<< " |time=" << loc_asctime(localtime(&time_now)) << ")\tllh: " << llh
			<< " ppl: " << exp(llh / trs) << " err: " << (trs - right) / trs << endl;
		llh = trs = right = 0;
		//////////////////////////////////////
		static int logc = 0;
		++logc;
		if (logc % 25 == 1) { // report on dev set
			unsigned dev_size = sr_corpus.deving.sentences1.size();
			int PU_ID = sr_corpus.pos2Int.get("PU", -1);
			// dev_size = 100;
			double right = 0, label_right = 0;;
			double correct_heads = 0;
			double total_heads = 0;
			auto t_start = std::chrono::high_resolution_clock::now();
			for (unsigned sii = 0; sii < dev_size; ++sii) {
				sentence_item& sen_item = sr_corpus.deving.sentences1[sii];
				sentence_tr& sentr = pred_corpus.deving.sentences[sii];

				const vector<int>& sentence = sen_item.raw_sent;
				const vector<int>& sentencePos = sen_item.sentPos;
				vector<int> tsentence = sentence;
				for (auto& w : tsentence)
					if (training_vocab.count(w) == 0) w = kUNK;
				ComputationGraph hg;
				for (int wx = 0;wx < tsentence.size();wx++)
					sentr[wx].form_useid = tsentence[wx];
				vector<pair<int, int>> pred = sr_parser->log_prob_parser(
					&hg,
					sentence,
					tsentence,
					sen_item.sentPos,
					sen_item.goldhead,
					sen_item.goldrel,
					sen_item.goldhead2deps,
					sentr,
					false,
					&right,0);
				for (int iw = 0;iw < pred.size()-1;iw++) {
					if (sen_item.sentPos[iw] == PU_ID)
						continue;
					if (pred[iw].first == sen_item.goldhead[iw]) {
						correct_heads++;
						if (pred[iw].second == sen_item.goldrel[iw])
							label_right++;
					}
					total_heads++;

				}
			}
			auto t_end = std::chrono::high_resolution_clock::now();
			cerr << "  **dev (iter=" << iter << " epoch=" 
				<< (tot_seen / sr_corpus.training.sentences.size()) 
				<< ")\tllh=" << llh << " uas: " << (correct_heads / total_heads) 
				<< " las: " << (label_right / total_heads)
				<< "\t[" << dev_size 
				<< " sents in " << std::chrono::duration<double, std::milli>(t_end - t_start).count()
				<< " ms]" << endl;

			fprintf(logO, "total=%.2f,\tepch=%.3f,\tdev_uas=%.5f,\tdev_las=%.5f\n", 
				tot_seen, (float)(tot_seen / sr_corpus.training.sentences.size()),
				(float)(correct_heads / total_heads), (float)(label_right / total_heads));
			fflush(logO);

			if (correct_heads > best_correct_heads) {
				best_correct_heads = correct_heads;
				char buff[512];
				fprintf(logO, "total=%.2f,\tepch=%.3f,\tdev_uas=%.5f,\tdev_uas1=%.5f -- save point 1-best\n", 
					tot_seen, (float)(tot_seen / sr_corpus.training.sentences.size()),
					(float)(correct_heads / total_heads),0);
				fflush(logO);

				SaveModel((oname+"_kbest").c_str());
			}

		}
	}

}

