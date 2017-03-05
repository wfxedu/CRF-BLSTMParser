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
#include "util.h"

#include <signal.h>
#include "UnlabelParserCRF.h"
#include "labelParserCRF.h"
#ifdef WIN32
#include <process.h>
#define getpid _getpid
#endif
using namespace CRFParser;

static DatasetTB dataset;
namespace po = boost::program_options;
static unordered_map<unsigned, vector<float>> pretrained;
static AlignedMemoryPool<8>* base_pool = nullptr;
static int max_epch = 30;


void InitCommandLine(int argc, char** argv, po::variables_map* conf) {
	po::options_description opts("Configuration options");
	opts.add_options()
		("training_data,T", po::value<string>(), "List of Transitions - Training corpus")
		("dev_data,d", po::value<string>(), "Development corpus")
		("test_data,p", po::value<string>(), "Test corpus")
		("unk_strategy,o", po::value<unsigned>()->default_value(1), "Unknown word strategy: 1 = singletons become UNK with probability unk_prob")
		("unk_prob,u", po::value<double>()->default_value(0.25), "Probably with which to replace singletons with UNK in training data")
		("model,m", po::value<string>(), "Load saved model from this file")
		("pretrained", po::value<string>(), "Load saved model from this file")
		("layers", po::value<unsigned>()->default_value(2), "number of LSTM layers")

		("embeding_dim", po::value<unsigned>()->default_value(100), "input embedding size")
		("input_dim", po::value<unsigned>()->default_value(100), "hidden dimension")
		("hidden_dim", po::value<unsigned>()->default_value(200), "hidden dimension")
		("distance_dim", po::value<unsigned>()->default_value(100), "hidden dimension")
		("segment_dim", po::value<unsigned>()->default_value(100), "hidden dimension")
		("pretrained_dim", po::value<unsigned>()->default_value(50), "pretrained input dimension")
		("pos_dim", po::value<unsigned>()->default_value(25), "POS dimension")
		("rel_dim", po::value<unsigned>()->default_value(25), "relation dimension")
		("max_epch", po::value<unsigned>()->default_value(30), "max_epch")
		("dropout", po::value<double>()->default_value(0.1), "relation dimension")
		("out_model", po::value<string>(), "saving model from this file")
		("out_pred", po::value<string>(), "saving predict from this file")

		("save_cfg", "Save config!")
		("em_org", "Using org embeddings?")
		("train,t", "Should training be run?")
		("words,w", po::value<string>(), "Pretrained word embeddings")
		("help,h", "Help");
	po::options_description dcmdline_options;
	dcmdline_options.add(opts);
	po::store(parse_command_line(argc, argv, dcmdline_options), *conf);
	if (conf->count("help")) {
		cerr << dcmdline_options << endl;
		exit(1);
	}
	if (conf->count("training_data") == 0) {
		cerr << "Please specify --traing_data (-T): this is required to determine the vocabulary mapping, even if the parser is used in prediction mode.\n";
		exit(1);
	}
}
////////////////////////////////////////////////////////////////////////////////////



using namespace::std;

void fill_cfg() {
	parser_config::LAYERS = 2;
	parser_config::INPUT_DIM = 100;
	parser_config::HIDDEN_DIM = 100;
	parser_config::ACTION_DIM = 20;
	parser_config::PRETRAINED_DIM = 100;
	parser_config::LSTM_INPUT_DIM = 100;
	parser_config::POS_DIM = 12;
	parser_config::REL_DIM = 20;
	parser_config::USE_POS = true;

	parser_config::ROOT_SYMBOL = "ROOT";
	parser_config::kROOT_SYMBOL = 0;
	parser_config::ACTION_SIZE = 0;
	parser_config::VOCAB_SIZE = 0;
	parser_config::POS_SIZE = 0;

	parser_config::LABEL_SIZE = -1;
	parser_config::INPUT_DIM_DL = 100;
	parser_config::HIDDEN_DIM_DH = 200;
	parser_config::DISTANCE_DIM = 100;
	parser_config::SEGMENT_DIM = 100;
	parser_config::MAX_DISTANCE = 500;
}

//./model/model_orgfull ./ctb_test.conll ./res_test/ctb_test.pred
//test llh=0 uas: 0.865212 las: 0 sents in 68520.5 ms]
int main(int argc, char** argv) {
	cnn::Initialize(argc, argv);
	base_pool = new AlignedMemoryPool<8>(1024UL * (1UL << 20));
	fill_cfg();
	cerr << "COMMAND:";
	for (unsigned i = 0; i < static_cast<unsigned>(argc); ++i) cerr << ' ' << argv[i];
	cerr << endl;
	string fname = argv[1];
	string testfile = argv[2];
	string outfile = argv[3];

	parser_config nn_cfg;
	nn_cfg.load(fname + ".cfg", &dataset);
	map<unsigned, unsigned> counts;
	load_countsmap(fname + ".counts", counts);
	if(0){
		cerr << "loading EmbedDict..." << endl;
		util::load_EmbedDict(dataset, *nn_cfg.pretrained,
			"sskip.100.vectors", false);
		cerr << "finished!" << endl;
	}
	set<unsigned> training_vocab; 
	set<unsigned> singletons;
	for (auto wc : counts)
		training_vocab.insert(wc.first);
	for (auto wc : counts)
		if (wc.second == 1) singletons.insert(wc.first);
	//-----------------------------------
	const int kUNK = dataset.words2Int.reg(treebank::UNK);
	int PU_POS = dataset.pos2Int.tokens2Int["PU"];
	std::cerr << "Number of words: " << counts.size() << endl;
	Model model;
	nn_cfg.base_pool = base_pool;
	nn_cfg.model = &model;
	ParserBuilderLabel parser(nn_cfg);
	ifstream in(fname.c_str());
	boost::archive::text_iarchive ia(in);
	ia >> *nn_cfg.model;
	//----------------------------
	dataset.load_test(testfile);

	if (dataset.testing.sentences.size() > 0) {
		unsigned test_size = dataset.testing.sentences.size();

		ofstream test_out(outfile);
		// dev_size = 100;
		double llh = 0;
		double trs = 0;
		double right = 0;
		double correct_heads = 0;
		double correct_headlabel = 0;
		double total_heads = 0;
		auto t_start = std::chrono::high_resolution_clock::now();
		for (unsigned sii = 0; sii < test_size; ++sii) {
			sentence_tr& sent = dataset.testing.sentences[sii];
			//sentree_tr&  sentree = dataset.testing.sentrees[sii];
			for (auto& w : sent) {
				if (training_vocab.count(w.form_id) == 0)
					w.form_useid = kUNK;
				else
					w.form_useid = w.form_id;
				w.pos_useid = w.pos_id;
			}

			base_pool->free();
			ComputationGraph hg;
			vector<pair<int, int>> outsen;
			cnn_train_mode = false;
			parser.decode(&hg, sent, outsen);

			for (int k = 1;k < outsen.size();k++) {
				//1	´÷ÏàÁú	_	NR	NR	_	2	VMOD	_	_
				string word = dataset.words2Int.int2Tokens[sent[k].form_id];
				string pos = dataset.pos2Int.int2Tokens[sent[k].pos_id];
				string prels = dataset.prel2Int.int2Tokens[outsen[k].second];
				test_out << (k) << "\t" << word << "\t_\t" << pos << "\t" << pos << "\t_\t" <<
					outsen[k].first << "\t" << prels << "\t_\t" << "\t_\t" << endl;

				if (sent[k].pos_id == PU_POS)
					continue;
				total_heads++;
				if (outsen[k].first == sent[k].parent) {
					correct_heads++;
					if (outsen[k].second == sent[k].prel_id)
						correct_headlabel++;
				}
			}
			if (sii<test_size - 1)
				test_out << endl;
		}
		auto t_end = std::chrono::high_resolution_clock::now();
		cerr << "  **test llh=" << llh << " uas: " << (correct_heads / total_heads) <<
			" las: " << (correct_headlabel / total_heads) << " sents in "
			<< std::chrono::duration<double, std::milli>(t_end - t_start).count() << " ms]"
			<< endl;
	}
	printf("finished!\n");
	return 0;
}

//-T ctb_train.conll -d ctb_dev.conll -p ctb_test.conll --hidden_dim 100  --pretrained_dim 200 --pretrained sskip.100.vectors  --rel_dim 25 --pos_dim 25 --input_dim 100 -m ./model/model_orgfull --dropout 0.1 --out_pred ./res_test/ctb_test_org.pred
int main_Train(int argc, char** argv) {
	cnn::Initialize(argc, argv);
	base_pool = new AlignedMemoryPool<8>(1024UL * (1UL << 20));
	fill_cfg();
	cerr << "COMMAND:";
	for (unsigned i = 0; i < static_cast<unsigned>(argc); ++i) cerr << ' ' << argv[i];
	cerr << endl;
	unsigned status_every_i_iterations = 100;

	po::variables_map conf;
	InitCommandLine(argc, argv, &conf);

	parser_config::LAYERS = conf["layers"].as<unsigned>();
	//parser_config::EMBEDING_DIM = conf["embeding_dim"].as<unsigned>();
	parser_config::INPUT_DIM_DL = conf["input_dim"].as<unsigned>();
	parser_config::HIDDEN_DIM_DH = conf["hidden_dim"].as<unsigned>();
	parser_config::DISTANCE_DIM = conf["distance_dim"].as<unsigned>();
	parser_config::SEGMENT_DIM = conf["segment_dim"].as<unsigned>();
	parser_config::PRETRAINED_DIM = conf["pretrained_dim"].as<unsigned>();
	parser_config::POS_DIM = conf["pos_dim"].as<unsigned>();
	max_epch = conf["max_epch"].as<unsigned>();
	parser_config::REL_DIM = conf["rel_dim"].as<unsigned>();
	const unsigned unk_strategy = conf["unk_strategy"].as<unsigned>();
	parser_config::DROP_OUT = conf["dropout"].as<double>();
	string out_model;
	if(conf.count("out_model"))
		out_model = conf["out_model"].as<string>();
	cerr << "dropout:" << parser_config::DROP_OUT << "; Unknown word strategy: ";
	if (unk_strategy == 1) {
		cerr << "STOCHASTIC REPLACEMENT\n";
	}
	else {
		abort();
	}
	const double unk_prob = conf["unk_prob"].as<double>();
	assert(unk_prob >= 0.); assert(unk_prob <= 1.);
	ostringstream os;
	os << "./model/parser_" << '_' << parser_config::LAYERS
		<< '_' << parser_config::INPUT_DIM_DL
		<< '_' << parser_config::HIDDEN_DIM_DH
		<< '_' << parser_config::POS_DIM
		<< '_' << parser_config::REL_DIM
		 << "_params";
	int best_correct_heads = 0;
	const string fname = out_model;
	string parameter_info = os.str();
	cerr << "Writing parameters to file: " << fname << endl;

	dataset.load_train(conf["training_data"].as<string>());
	dataset.load_dev(conf["dev_data"].as<string>());
	if(conf.count("test_data"))
		dataset.load_test(conf["test_data"].as<string>());
	const int kUNK = dataset.words2Int.reg(treebank::UNK);
	const int kUNK_POS = dataset.pos2Int.reg(treebank::NONE_POS);
	//------------------------------------------------------
	EmbedDict pretrained;
	if (conf.count("pretrained")) {
		cerr << "loading EmbedDict..." << endl;
		util::load_EmbedDict(dataset, pretrained,
			conf["pretrained"].as<string>().c_str(), conf.count("em_org"));
		cerr << "finished load pretrained! \t" << pretrained.size() << endl;
	}
	//------------------------------------------------------
	set<unsigned> training_vocab; // words available in the training corpus
	set<unsigned> singletons;
	map<unsigned, unsigned> counts;
	{  // compute the singletons in the parser's training data
		for (auto sent : dataset.training.sentences)
			for (auto word : sent) { training_vocab.insert(word.form_id); counts[word.form_id]++; }
		for (auto wc : counts)
			if (wc.second == 1) singletons.insert(wc.first);
	}
	//------------------------------------------------------
	std::cerr << "Number of words: " << counts.size() << endl;
	parser_config::VOCAB_SIZE = dataset.words2Int.int2Tokens.size()+1;
	parser_config::POS_SIZE = dataset.pos2Int.int2Tokens.size() + 10;
	parser_config::LABEL_SIZE = dataset.prel2Int.tokens2Int.size() + 1;
	int PU_POS = dataset.pos2Int.tokens2Int["PU"];
	Model model;
	parser_config nn_cfg;
	nn_cfg.base_pool = base_pool;
	nn_cfg.model = &model;
	nn_cfg.pretrained = &pretrained;
	ParserBuilderLabel parser(nn_cfg);
	if (conf.count("model")) {
		ifstream in(conf["model"].as<string>().c_str());
		boost::archive::text_iarchive ia(in);
		ia >> *nn_cfg.model;
	}
	//----------------------------
	if (conf.count("save_cfg")) {
		nn_cfg.save(fname + ".cfg",&dataset);
		save_countsmap(fname + ".counts", counts);
		nn_cfg.load(fname + ".cfg", &dataset);
		load_countsmap(fname + ".counts", counts);
		printf("exit first!\n");
		exit(-1);
	}
	//----------------------------
	const std::vector<LookupParameters*> &lookup_params = model.lookup_parameters_list();
	const std::vector<Parameters*> &params = model.parameters_list();
	long total_number = 0;
	for (auto p : params) 
		total_number+=p->size();
	for (auto p : lookup_params)
		total_number += p->size();
	cerr << "#Parameters Number: " << total_number << endl;
	//----------------------------
	int nsentences = dataset.training.sentences.size();
	if (conf.count("train")) {
		ofstream log_out(fname + ".log");
		log_out << parameter_info << endl;
		AdamTrainer sgd11(0, /* lambda =*/ 1e-6, /* alpha =*/ 0.001,
			/* beta_1 =*/ 0.9, /* beta_2 =*/ 0.9, /* eps =*/ 1e-8);


		SimpleSGDTrainer sgd(&model);
		sgd.eta_decay = 0.05;
		vector<unsigned> order(nsentences);
		for (unsigned i = 0; i < nsentences; ++i)
			order[i] = i;
		double tot_seen = 0;
		status_every_i_iterations = status_every_i_iterations>nsentences? nsentences: status_every_i_iterations;
		unsigned si = nsentences;
		std::cerr << "NUMBER OF TRAINING SENTENCES: " << nsentences << endl;
		log_out << "NUMBER OF TRAINING SENTENCES: " << nsentences << endl;
		double right = 0;
		double ttw = 0;
		double llh = 0;
		bool first = true;
		int iter = -1;
		time_t time_start = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
		cerr << "TRAINING STARTED AT: " << put_time(localtime(&time_start), "%c") << endl;
		log_out << "TRAINING STARTED AT: " << put_time(localtime(&time_start), "%c") << endl;
		
		int batch = 0;
		while (1) {
			++iter;
			for (unsigned sii = 0; sii < status_every_i_iterations; ++sii) {
				if (si == nsentences) {
					si = 0;
					if (first) { first = false; }
					else { sgd.update_epoch(); }
					cerr << "**SHUFFLE\n";
					random_shuffle(order.begin(), order.end());
				}
				tot_seen += 1;

				base_pool->free();
				sentence_tr& sent = dataset.training.sentences[order[si]];
				if (unk_strategy == 1) {
					for (auto& w : sent) {
						if (cnn::rand01() < unk_prob / (unk_prob + counts[w.form_id]))
							w.form_useid = kUNK;
						else
							w.form_useid = w.form_id;

						if (cnn::rand01() < unk_prob / (unk_prob + counts[w.form_id]))
							w.pos_useid = kUNK_POS;
						else
							w.pos_useid = w.pos_id;
					}
				}
				ComputationGraph hg;
				bool skip = false;
				vector<pair<int, int>> outsen;
				cnn_train_mode = true;
				double log_p = parser.construct_grad(&hg, sent, outsen);
				hg.incremental_forward();
				if (log_p<= Negative_Infinity) {
					++si;
					continue;
				}

				for (int kk = 0;kk < sent.size();kk++) {
					if (outsen[kk].first == sent[kk].parent)
						right++;
				}

				ttw += sent.size();
				++si;
				hg.backward();
				sgd.update(1.0);
				//sgd.update(log_p);
				llh += exp(log_p);
			}
			sgd.status();
			time_t time_now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
			cerr << "update #" << iter << " (epoch " << (tot_seen / nsentences) 
				<< " |time=" << put_time(localtime(&time_now), "%c") 
				<< "), llh: " << llh << " err: " << (ttw - right) / ttw << endl;
			log_out << "update #" << iter << " (epoch " << (tot_seen / nsentences)
				<< " |time=" << put_time(localtime(&time_now), "%c") 
				<< "), llh: " << llh << " err: " << (ttw - right) / ttw  << endl;
			llh = ttw = right = 0;
			//si = 0;//hack
			//--------------------------------------
			static int logc = 0;
			++logc;
			if (logc % 25 == 1) { // report on dev set
				unsigned dev_size = dataset.deving.sentences.size();
				// dev_size = 100;
				double llh = 0;
				double trs = 0;
				double right = 0;
				double correct_heads = 0;
				double correct_headlabel = 0;
				double total_heads = 0;
				auto t_start = std::chrono::high_resolution_clock::now();
				for (unsigned sii = 0; sii < dev_size; ++sii) {
					sentence_tr& sent = dataset.deving.sentences[sii];
					//sentree_tr&  sentree = dataset.deving.sentrees[sii];
					for (auto& w : sent) {
						if (training_vocab.count(w.form_id) == 0)
							w.form_useid = kUNK;
						else
							w.form_useid = w.form_id;
						w.pos_useid = w.pos_id;
					}

					base_pool->free();
					ComputationGraph hg;
					vector<pair<int, int>> outsen;
					cnn_train_mode = false;
					parser.decode(&hg, sent, outsen);

					for (int k = 1;k < outsen.size();k++) {
						if (sent[k].pos_id == PU_POS)
							continue;
						total_heads++;
						if (outsen[k].first == sent[k].parent) {
							correct_heads++;
							if (outsen[k].second == sent[k].prel_id)
								correct_headlabel++;
						}
					}
				}
				auto t_end = std::chrono::high_resolution_clock::now();
				cerr << "  **dev (iter=" << iter << " epoch=" << (tot_seen / dev_size) 
					<< ")\tllh=" << llh <<  " uas: " << (correct_heads / total_heads) << 
					" las: " << (correct_headlabel / total_heads) <<
					"\t[" << dev_size << " sents in " 
					<< std::chrono::duration<double, std::milli>(t_end - t_start).count() << " ms]" 
					<< endl;
				log_out << "  **dev (iter=" << iter << " epoch=" << (tot_seen / dev_size)
					<< ")\tllh=" << llh << " uas: " << (correct_heads / total_heads) <<
					" las: " << (correct_headlabel / total_heads) <<
					"\t[" << dev_size << " sents in "
					<< std::chrono::duration<double, std::milli>(t_end - t_start).count() << " ms]"
					<< endl;
				if (correct_heads > best_correct_heads) {
					best_correct_heads = correct_heads;
					
					static int save_id = 0;
					char message[1025];
					sprintf(message, "  **total=%.2f,\tepch=%.3f,\tdev_uas=%.5f,\tdev_uas1=%.5f -- save point %d \n",
						tot_seen, (float)(tot_seen / nsentences),
						(float)(correct_heads / total_heads), (float)((correct_headlabel / total_heads)), save_id);
					cerr << message;
					log_out << message;
					log_out.flush();

					sprintf(message, "%s[s%d][e%.3f][U%.3f,L%.3f].bin", fname.c_str(), save_id, (float)(tot_seen / nsentences),
						(float)(correct_heads / total_heads), (float)(correct_headlabel / total_heads));
					ofstream out(message);
					boost::archive::text_oarchive oa(out);
					oa << model;
					save_id++;
				}
			}
			if ((tot_seen / nsentences)> max_epch)
				break;
		}
	} // should do training?

	if (dataset.testing.sentences.size() > 0) {
		unsigned test_size = dataset.testing.sentences.size();
		
		ofstream test_out(conf["out_pred"].as<string>());
		// dev_size = 100;
		double llh = 0;
		double trs = 0;
		double right = 0;
		double correct_heads = 0;
		double correct_headlabel = 0;
		double total_heads = 0;
		auto t_start = std::chrono::high_resolution_clock::now();
		for (unsigned sii = 0; sii < test_size; ++sii) {
			sentence_tr& sent = dataset.testing.sentences[sii];
			//sentree_tr&  sentree = dataset.testing.sentrees[sii];
			for (auto& w : sent) {
				if (training_vocab.count(w.form_id) == 0)
					w.form_useid = kUNK;
				else
					w.form_useid = w.form_id;
				w.pos_useid = w.pos_id;
			}

			base_pool->free();
			ComputationGraph hg;
			vector<pair<int, int>> outsen;
			cnn_train_mode = false;
			parser.decode(&hg, sent, outsen);

			for (int k = 1;k < outsen.size();k++) {
				//1	´÷ÏàÁú	_	NR	NR	_	2	VMOD	_	_
				string word = dataset.words2Int.int2Tokens[sent[k].form_id];
				string pos = dataset.pos2Int.int2Tokens[sent[k].pos_id];
				string prels = dataset.prel2Int.int2Tokens[outsen[k].second];
				test_out << (k) << "\t" << word << "\t_\t" << pos << "\t" << pos << "\t_\t" <<
					outsen[k].first << "\t" << prels << "\t_\t" << "\t_\t" << endl;

				if (sent[k].pos_id == PU_POS)
					continue;
				total_heads++;
				if (outsen[k].first == sent[k].parent) {
					correct_heads++;
					if (outsen[k].second == sent[k].prel_id )
						correct_headlabel++;
				}
			}
			if(sii<test_size -1)
				test_out << endl;

			if (sii == 99) {
				auto t_end = std::chrono::high_resolution_clock::now();
				float sec2sen = std::chrono::duration<double, std::milli>(t_end - t_start).count() / 100;
				char buff[1025];
				sprintf(buff, "./benchmark/milsecond2sentence[%f].log", sec2sen);
				FILE* f1 = fopen(buff, "wb");
				if (f1) fclose(f1);
			}
		}
		auto t_end = std::chrono::high_resolution_clock::now();
		cerr << "  **test llh=" << llh << " uas: " << (correct_heads / total_heads) <<
			" las: " << (correct_headlabel / total_heads) << " sents in "
			<< std::chrono::duration<double, std::milli>(t_end - t_start).count() << " ms]"
			<< endl;
	}
	printf("finished!\n");
}
