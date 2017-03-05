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


inline unsigned UTF8Len(unsigned char x) {
	if (x < 0x80) return 1;
	else if ((x >> 5) == 0x06) return 2;
	else if ((x >> 4) == 0x0e) return 3;
	else if ((x >> 3) == 0x1e) return 4;
	else if ((x >> 2) == 0x3e) return 5;
	else if ((x >> 1) == 0x7e) return 6;
	else return 0;
}

class Corpus
{
public:
	bool USE_SPELLING = false;

	std::map<int, std::vector<unsigned>> correct_act_sent;
	std::map<int, std::vector<unsigned>> sentences;
	std::map<int, std::vector<unsigned>> sentencesPos;

	std::map<int, std::vector<unsigned>> correct_act_sentDev;
	std::map<int, std::vector<unsigned>> sentencesDev;
	std::map<int, std::vector<unsigned>> sentencesPosDev;
	std::map<int, std::vector<std::string>> sentencesStrDev;
	unsigned nsentencesDev;
	unsigned nsentences;

	unsigned nwords;
	unsigned nactions;
	unsigned npos;

	int max;
	int maxPos;

	std::map<std::string, unsigned> wordsToInt;
	std::map<unsigned, std::string> intToWords;
	std::vector<std::string> actions;

	std::map<std::string, unsigned> posToInt;
	std::map<unsigned, std::string> intToPos;

	/*int maxChars;
	std::map<std::string, unsigned> charsToInt;
	std::map<unsigned, std::string> intToChars;*/

	// String literals
	static const char* UNK;
	static const char* BAD0;
public:
	Corpus();
	~Corpus();

	void save_config(const char* path);

	void load_config(const char* path);

	void load_correct_actions(std::string file);

	unsigned get_or_add_word(const std::string& word);

	unsigned get_word(const std::string& word);

	void load_correct_actionsDev(std::string file);

	void load_Test(std::string file);
};

