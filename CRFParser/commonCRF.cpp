#include "commonCRF.h"
#include "TreeReaderCRF.h"

namespace CRFParser {
unsigned parser_config::LAYERS = 2;
unsigned parser_config::INPUT_DIM = 100;
unsigned parser_config::HIDDEN_DIM = 100;
unsigned parser_config::ACTION_DIM = 20;
unsigned parser_config::PRETRAINED_DIM = 50;
unsigned parser_config::LSTM_INPUT_DIM = 100;
unsigned parser_config::POS_DIM = 10;
unsigned parser_config::REL_DIM = 20;
float parser_config::DROP_OUT = 0.0f;

bool parser_config::USE_POS = true;

char* parser_config::ROOT_SYMBOL = "ROOT";
unsigned parser_config::kROOT_SYMBOL = 0;
unsigned parser_config::ACTION_SIZE = 0;
unsigned parser_config::VOCAB_SIZE = 0;
unsigned parser_config::POS_SIZE = 0;
unsigned parser_config::KBEST = 10;

unsigned parser_config::LABEL_SIZE = -1;
unsigned parser_config::INPUT_DIM_DL = 100;
unsigned parser_config::HIDDEN_DIM_DH = 200;
unsigned parser_config::DISTANCE_DIM = 100;
unsigned parser_config::SEGMENT_DIM = 100;
unsigned parser_config::MAX_DISTANCE = 500;

void save_cfg(string path, DatasetTB& dataset, parser_config& cfg) {
	ofstream out(path);
	out << "parser_config::LAYERS=" << parser_config::LAYERS << endl;
	out << "parser_config::INPUT_DIM=" << parser_config::INPUT_DIM << endl;
	out << "parser_config::HIDDEN_DIM=" << parser_config::HIDDEN_DIM << endl;
	out << "parser_config::ACTION_DIM=" << parser_config::ACTION_DIM << endl;
	out << "parser_config::PRETRAINED_DIM=" << parser_config::PRETRAINED_DIM << endl;
	out << "parser_config::LSTM_INPUT_DIM=" << parser_config::LSTM_INPUT_DIM << endl;
	out << "parser_config::POS_DIM=" << parser_config::POS_DIM << endl;
	out << "parser_config::REL_DIM=" << parser_config::REL_DIM << endl;
	out << "parser_config::DROP_OUT=" << parser_config::DROP_OUT << endl;
	out << "parser_config::USE_POS=" << parser_config::USE_POS << endl;
	out << "parser_config::ROOT_SYMBOL=" << parser_config::ROOT_SYMBOL << endl;
	out << "parser_config::kROOT_SYMBOL=" << parser_config::kROOT_SYMBOL << endl;
	out << "parser_config::ACTION_SIZE=" << parser_config::ACTION_SIZE << endl;
	out << "parser_config::VOCAB_SIZE=" << parser_config::VOCAB_SIZE << endl;
	out << "parser_config::POS_SIZE=" << parser_config::POS_SIZE << endl;
	out << "parser_config::KBEST=" << parser_config::KBEST << endl;
	out << "parser_config::LABEL_SIZE=" << parser_config::LABEL_SIZE << endl;
	out << "parser_config::INPUT_DIM_DL=" << parser_config::INPUT_DIM_DL << endl;
	out << "parser_config::HIDDEN_DIM_DH=" << parser_config::HIDDEN_DIM_DH << endl;
	out << "parser_config::DISTANCE_DIM=" << parser_config::DISTANCE_DIM << endl;
	out << "parser_config::SEGMENT_DIM=" << parser_config::SEGMENT_DIM << endl;
	out << "parser_config::MAX_DISTANCE=" << parser_config::MAX_DISTANCE << endl;
	//--------------------------
	out << "DatasetTB" << endl;
	out << "POSTag" << endl;
	out << "POSSIZE=" << dataset.pos2Int.nTokens << endl;
	for (auto iter : dataset.pos2Int.tokens2Int) {
		out << "P\t" << iter.first << "\t" << iter.second << endl;
	}
	//--------------------------
	out << "WORD" << endl;
	out << "WORDSIZE=" << dataset.words2Int.nTokens << endl;
	for (auto iter : dataset.words2Int.tokens2Int) {
		out << "W\t" << iter.first << "\t" << iter.second << endl;
	}
	//--------------------------
	out << "LABEL" << endl;
	out << "LABELSIZE=" << dataset.prel2Int.nTokens << endl;
	for (auto iter : dataset.prel2Int.tokens2Int) {
		out << "L\t" << iter.first << "\t" << iter.second << endl;
	}
	out << "END_SAVE" << endl;
	//--------------------------
}

void load_cfg(string path, DatasetTB& dataset, parser_config& cfg) {
	dataset.pos2Int.int2Tokens.clear();
	dataset.pos2Int.nTokens = 0;
	dataset.pos2Int.tokens2Int.clear();
	dataset.words2Int.int2Tokens.clear();
	dataset.words2Int.nTokens = 0;
	dataset.words2Int.tokens2Int.clear();
	dataset.prel2Int.int2Tokens.clear();
	dataset.prel2Int.nTokens = 0;
	dataset.prel2Int.tokens2Int.clear();
	//-----------------------------
	ifstream in(path);
	string line;
	int step = 0;
	while (getline(in, line)) {
		if (line.size() == 0)
			continue;
		if (line == "END_SAVE")
			break;
		if (line == "DatasetTB") {
			step = 1;
			continue;
		}
		if (line == "POSTag") {
			step = 2;
			continue;
		}
		if (line == "LABEL") {
			step = 5;
			continue;
		}
		if (line == "WORD") {
			step = 3;
			continue;
		}
		//---------------------------------------
		if (step == 0) {
			int idx = line.rfind('=');
			string name = line.substr(0, idx);
			string vs = line.substr(idx + 1);
			float v = atof(vs.c_str());
			if (name == "parser_config::LAYERS")  parser_config::LAYERS = v;
			if (name == "parser_config::INPUT_DIM")  parser_config::INPUT_DIM = v;
			if (name == "parser_config::HIDDEN_DIM")  parser_config::HIDDEN_DIM = v;
			if (name == "parser_config::ACTION_DIM")  parser_config::ACTION_DIM = v;
			if (name == "parser_config::PRETRAINED_DIM")  parser_config::PRETRAINED_DIM = v;
			if (name == "parser_config::LSTM_INPUT_DIM")  parser_config::LSTM_INPUT_DIM = v;
			if (name == "parser_config::POS_DIM")  parser_config::POS_DIM = v;
			if (name == "parser_config::REL_DIM")  parser_config::REL_DIM = v;
			if (name == "parser_config::DROP_OUT")  parser_config::DROP_OUT=v;
			if (name == "parser_config::USE_POS")  parser_config::USE_POS = v;
			if (name == "parser_config::ROOT_SYMBOL")  parser_config::ROOT_SYMBOL = _strdup(vs.c_str());
			if (name == "parser_config::kROOT_SYMBOL")  parser_config::kROOT_SYMBOL = v;
			if (name == "parser_config::ACTION_SIZE")  parser_config::ACTION_SIZE = v;
			if (name == "parser_config::VOCAB_SIZE")  parser_config::VOCAB_SIZE = v;
			if (name == "parser_config::POS_SIZE")  parser_config::POS_SIZE = v;
			if (name == "parser_config::KBEST")  parser_config::KBEST = v;
			if (name == "parser_config::LABEL_SIZE")  parser_config::LABEL_SIZE = v;
			if (name == "parser_config::INPUT_DIM_DL")  parser_config::INPUT_DIM_DL = v;
			if (name == "parser_config::HIDDEN_DIM_DH")  parser_config::HIDDEN_DIM_DH = v;
			if (name == "parser_config::DISTANCE_DIM")  parser_config::DISTANCE_DIM = v;
			if (name == "parser_config::SEGMENT_DIM")  parser_config::SEGMENT_DIM = v;
			if (name == "parser_config::MAX_DISTANCE")  parser_config::MAX_DISTANCE = v;
		}

		if (step == 2) { //POSTag
			if (line[0] == 'P' && line[1] == '\t') {
				int	idx = line.rfind('\t');
				string name = line.substr(2, idx-2);
				string vs = line.substr(idx + 1);
				float v = atof(vs.c_str());
				dataset.pos2Int.tokens2Int[name] = v;
				dataset.pos2Int.int2Tokens[v] = name;
			}
			else {
				int	idx = line.rfind('=');
				if (idx != -1) {
					string name = line.substr(0, idx);
					string vs = line.substr(idx + 1);
					int v = atoi(vs.c_str());
					assert(name == "POSSIZE");
					dataset.pos2Int.nTokens = v;
				}
				else
					assert(0);
			}
		}
		//-----------------
		if (step == 3) { //WORD
			if (line[0] == 'W' && line[1] == '\t') {
				int	idx = line.rfind('\t');
				string name = line.substr(2, idx-2);
				string vs = line.substr(idx + 1);
				float v = atof(vs.c_str());
				dataset.words2Int.tokens2Int[name] = v;
				dataset.words2Int.int2Tokens[v] = name;
			}
			else {
				int	idx = line.rfind('=');
				if (idx != -1) {
					string name = line.substr(0, idx);
					string vs = line.substr(idx + 1);
					int v = atoi(vs.c_str());
					assert(name == "WORDSIZE");
					dataset.words2Int.nTokens = v;
				}
				else
					assert(0);
			}
		}
		//-----------------
		if (step == 5) { //LABEL
			if (line[0] == 'L' && line[1] == '\t') {
				int	idx = line.rfind('\t');
				string name = line.substr(2, idx-2);
				string vs = line.substr(idx + 1);
				float v = atof(vs.c_str());
				dataset.prel2Int.tokens2Int[name] = v;
				dataset.prel2Int.int2Tokens[v] = name;
			}
			else {
				int	idx = line.rfind('=');
				if (idx != -1) {
					string name = line.substr(0, idx);
					string vs = line.substr(idx + 1);
					int v = atoi(vs.c_str());
					assert(name == "LABELSIZE");
					dataset.prel2Int.nTokens = v;
				}
				else
					assert(0);
			}
		}
	}
	//--------------------------
}
//////////////////////////////////////////////////
void save_embedding(string path, EmbedDict& dict) {
	//typedef unordered_map<unsigned, vector<float>> EmbedDict;
	unordered_map<unsigned, vector<float>>::iterator iter = dict.begin();
	ofstream out(path);
	out << dict.size()<<  endl <<iter->second.size() << endl;
	for (;iter != dict.end();iter++) {
		out << iter->first << "\t";
		for (int i = 0;i < iter->second.size();i++)
			out << iter->second[i] << "\t";
		out << endl;
	}
}

void load_embedding(string path, EmbedDict& dict) {
	dict.clear();
	ifstream in(path);
	string line;
	getline(in, line);
	int size = atoi(line.c_str());
	getline(in, line);
	int dim = atoi(line.c_str());
	char* buffer = new char[1024 * 56];

	while (getline(in, line)) {
		strcpy(buffer, line.c_str());
		vector<char*> eles;
		split2(buffer, '\t',eles);
		int idx = atoi(eles[0]);
		vector<float> vec;
		for (int i = 1;i < eles.size();i++) {
			if (eles[i][0] == 0)
				continue;
			float v = atof(eles[i]);
			vec.push_back(v);
		}
		assert(dim == vec.size());
		dict[idx] = vec;
		size--;
	}
	assert(size == 0);
}


void parser_config::save(string path, DatasetTB* dataset)
{
	save_cfg(path, *dataset, *this);
	save_embedding(path + ".em", *pretrained);
}

void parser_config::load(string path, DatasetTB* dataset)
{
	load_cfg(path, *dataset, *this);
	pretrained = new EmbedDict();
	load_embedding(path + ".em", *pretrained);
}


void save_countsmap(string path, map<unsigned, unsigned>& counts)
{
	map<unsigned, unsigned>::iterator iter = counts.begin();
	ofstream out(path);
	out << counts.size() << endl;
	for (;iter != counts.end();iter++) {
		out << iter->first << "\t"<< iter->second<<endl;
	}

}

void load_countsmap(string path, map<unsigned, unsigned>& counts)
{
	counts.clear();
	ifstream in(path);
	string line;
	getline(in, line);
	int size = atoi(line.c_str());
	char* buffer = new char[1024 * 56];

	while (getline(in, line)) {
		strcpy(buffer, line.c_str());
		vector<char*> eles;
		split2(buffer, '\t', eles);
		int idx = atoi(eles[0]);
		int num = atoi(eles[1]);
		counts[idx] = num;
		size--;
	}
	assert(size == 0);
}



}//namespace CRFParser {