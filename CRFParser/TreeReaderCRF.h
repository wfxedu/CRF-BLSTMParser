#pragma once
#include <string>
#include <iostream>
#include <cassert>
#include <fstream>
#include <sstream>
#include <vector>
#include <set>
#include <unordered_map>
#include <functional>
#include <vector>
#include <map>
#include <string>
#include <assert.h>
#include "commonCRF.h"
using namespace::std;
namespace CRFParser {

struct element_tr {
	int form_id;
	int form_useid;
	int pos_id;
	int pos_useid;
	int parent;
	int prel_id;
	element_tr() : form_id(-1), form_useid(-1), pos_id(-1), pos_useid(-1), parent(-1), prel_id(-1) {}
};

typedef vector<element_tr> sentence_tr;

inline void split2(char* str, char ch, vector<char*>& out)
{
	int len = (int)strlen(str);
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
//////////////////////////////////////////////////////////////////////////////


class dictionary {
public:
	std::map<std::string, unsigned> tokens2Int;
	std::map<unsigned, std::string> int2Tokens;
	unsigned nTokens;

	int  reg(std::string tok, int id) {
		tokens2Int[tok] = id;
		int2Tokens[id] = tok;
		return id;
	}

	int reg(std::string tok) {
		std::map<std::string, unsigned>::iterator iter = tokens2Int.find(tok);
		if (iter == tokens2Int.end()) {
			tokens2Int[tok] = nTokens;
			int2Tokens[nTokens] = tok;
			nTokens++;
			return nTokens - 1;
		} else {
			return iter->second;
		}
	}

	int get(std::string tok,int default_id) {
		std::map<std::string, unsigned>::iterator iter = tokens2Int.find(tok);
		if (iter == tokens2Int.end()) {
			return default_id;
		}
		else {
			return iter->second;
		}
	}

	dictionary() : nTokens(1) {}
};

struct sentence_item {
	vector<int> raw_sent;
	vector<int> sentPos;
	vector<int> goldhead;
	vector<int> goldrel;
	map<int, vector<int>>  goldhead2deps;
};

class treebank {
public:
	vector<sentence_tr> sentences;
	vector<sentence_item> sentences1;

	dictionary* words2Int;
	dictionary* pos2Int;
	dictionary* prel2Int;
	bool block_diction;

	static const char* UNK;
	static const char* UNK_POS;
	static const char* BAD0;
	static const char* ROOT;
	static const char* NONE_POS;
	static const char* NONE_REL;


	void init(dictionary* w2i, dictionary* p2i, dictionary* r2i,bool block) {
		words2Int = w2i;pos2Int = p2i;prel2Int = r2i;
		block_diction = block;
	}

	void load(const char* path) {
		std::ifstream tbFile(path);
		std::string lineS;
		words2Int->reg(BAD0, 0);
		int unk_id = words2Int->reg(UNK);
		int root_id = words2Int->reg(ROOT);
		int none_pos = pos2Int->reg(NONE_POS);
		int none_rel = prel2Int->reg(NONE_REL);
		std::vector<unsigned> current_sent;
		std::vector<unsigned> current_sent_pos;

		char* buffer = (char*)malloc(2048);
		int buffer_size = 2048;
		sentence_tr cur_sen;
		element_tr ele;
		ele.form_id = root_id;
		ele.parent = -1; //root is -1
		ele.pos_id = none_pos;
		ele.prel_id = none_rel;
		cur_sen.push_back(ele);

		while (getline(tbFile, lineS)) {
			if (lineS.size() >= buffer_size) {
				free(buffer);
				buffer_size = 2 * lineS.size();
				buffer = (char*)malloc(buffer_size);
			}
			strcpy(buffer, lineS.c_str());
			//--------------------------------------
			char* ln = strtrim(buffer);
			if (ln[0] == 0 && !cur_sen.empty()) {
				/*for (int i = 0;i < cur_sen.size();i++) {
					if (cur_sen[i].parent == 0)
						cur_sen[i].parent = cur_sen.size();
					else
						cur_sen[i].parent--;
				}*/
				
				sentences.push_back(cur_sen);
				cur_sen.clear();

				element_tr ele;
				ele.form_id = root_id;
				ele.parent = -1; //root is -1
				ele.pos_id = none_pos;
				ele.prel_id = none_rel;
				cur_sen.push_back(ele);
				continue;
			}
			if (ln[0] == 0)
				continue;

			//element
			vector<char*> eles;
			split2(ln, '\t', eles);
			//1	´÷ÏàÁú	_	NR	NR	_	2	VMOD	_	_	
			element_tr ele;
			ele.parent = atoi(eles[6]); //root is 0
			if (block_diction) {
				ele.form_id = words2Int->get(eles[1],unk_id);
				ele.pos_id = pos2Int->get(eles[3],-1);
				ele.prel_id = prel2Int->get(eles[7], - 1);
				if (ele.pos_id == -1)
					ele.pos_id = none_pos;
			}
			else {
				ele.form_id = words2Int->reg(eles[1]);
				ele.pos_id = pos2Int->reg(eles[3]);
				ele.prel_id = prel2Int->reg(eles[7]);
			}	
			cur_sen.push_back(ele);
		}
		if (!cur_sen.empty()) {
			sentences.push_back(cur_sen);
			cur_sen.clear();
		}
		free(buffer);

		fill_sentence1();
	}

	void fill_sentence1() {
		sentences1.resize(sentences.size());
		for (int i = 0;i < sentences.size();i++) {
			sentence_tr& sent = sentences[i];
			sentence_item& si = sentences1[i];
			si.raw_sent.resize(sent.size());
			si.sentPos.resize(sent.size());
			si.goldhead.resize(sent.size());
			si.goldrel.resize(sent.size());
			for (int j = 0;j < sent.size();j++) {
				element_tr& wd = sent[j];
				si.raw_sent[j] = wd.form_id;
				si.sentPos[j] = wd.pos_id;
				si.goldhead[j] = wd.parent;
				si.goldrel[j] = wd.prel_id;
				if(wd.parent>=0)
					si.goldhead2deps[wd.parent].push_back(j);
			}
		}
	}
};

class DatasetTB {
public:
	dictionary words2Int;
	dictionary pos2Int;
	dictionary prel2Int;
	treebank training;
	treebank deving;
	treebank testing;

	void load_train(std::string path);
	void load_test(std::string path);
	void load_dev(std::string path);
};

}//namespace CRFParser {