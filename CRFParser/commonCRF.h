#pragma once
#include <cstdlib>
#include <unordered_map>
#include <unordered_set>
#ifdef WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif
//#include "execinfo.h"
#include <signal.h>

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/program_options.hpp>

#include "cnn/training.h"
#include "cnn/cnn.h"
#include "cnn/expr.h"
#include "cnn/nodes.h"
#include "cnn/lstm.h"
#include "cnn/rnn.h"

using namespace cnn::expr;
using namespace cnn;
using namespace std;
namespace CRFParser {
	typedef unordered_map<unsigned, vector<float>> EmbedDict;

class DatasetTB;

typedef unordered_map<unsigned, vector<float>> EmbedDict;
typedef  vector<int> t_unvec;

struct parser_config{
	Model* model;
	EmbedDict* pretrained;

	static unsigned LAYERS;
	static unsigned INPUT_DIM;
	static unsigned HIDDEN_DIM;
	static unsigned ACTION_DIM;
	static unsigned PRETRAINED_DIM;
	static unsigned LSTM_INPUT_DIM;
	static unsigned POS_DIM;
	static unsigned REL_DIM;
	static float DROP_OUT;


	static bool USE_POS;
	static char* ROOT_SYMBOL;
	static unsigned kROOT_SYMBOL;
	static unsigned ACTION_SIZE;
	static unsigned VOCAB_SIZE;
	static unsigned POS_SIZE;
	static unsigned KBEST;

	//CRF
	static unsigned LABEL_SIZE;
	static unsigned INPUT_DIM_DL;
	static unsigned HIDDEN_DIM_DH;
	static unsigned DISTANCE_DIM;
	static unsigned SEGMENT_DIM;
	static unsigned MAX_DISTANCE;

	AlignedMemoryPool<8>* base_pool;

	void save(string path, DatasetTB* dataset);
	void load(string path, DatasetTB* dataset);
};

void save_countsmap(string path, map<unsigned, unsigned>& counts);
void load_countsmap(string path, map<unsigned, unsigned>& counts);


class seninfo {
public:
	t_unvec raw_sent;
	t_unvec sent;
	t_unvec sentPos;

	seninfo(const t_unvec& rs, const t_unvec& s, const t_unvec& sp) : raw_sent(rs), sent(s), sentPos(sp) {}
};

inline void string_split(char* str, char ch, vector<char*>& out)
{
	int len = strlen(str);
	out.clear();
	char* curitm = str;
	out.push_back(curitm);
	for (int i = 0; i<len; i++)
	{
		if (str[i] == ch) {
			str[i] = 0;
			if (i + 1<len) {
				curitm = str + i + 1;
				out.push_back(curitm);
			}
			else
				curitm = 0;
		}
	}
}

inline char* strtrim(char* source)
{
	while (*source != 0 && (*source == ' ' || *source == '\n' || *source == '\t'))
		source++;
	if (*source == 0)
		return source;
	int len = strlen(source);
	char *cur = source + len - 1;
	while (len>0 && (*cur == ' ' || *cur == '\n' || *cur == '\t')) {
		*cur = 0;
		cur--; len--;
	}
	return source;
};

struct score_pair {
	float score;
	int kbest_idx;
	int actidx;
	bool bdone;
	Expression adiste;
	score_pair() : bdone(false) {}
};


}//namespace CRFParser {
//#define lre_active_function
//#define soft_max_neg_scorer