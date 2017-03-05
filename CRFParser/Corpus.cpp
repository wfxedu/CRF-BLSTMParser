#include "Corpus.h"

#include <string.h>
#include <algorithm>
#define LABEL_HACK




const char* Corpus::UNK = "UNK";
const char* Corpus::BAD0 = "<BAD0>";

void ReplaceStringInPlace(std::string& subject, const std::string& search,
	const std::string& replace) {
	size_t pos = 0;
	while ((pos = subject.find(search, pos)) != std::string::npos) {
		subject.replace(pos, search.length(), replace);
		pos += replace.length();
	}
}


Corpus::Corpus()
{
	max = 0;
	maxPos = 0;
	//maxChars = 0; //Miguel
}


Corpus::~Corpus()
{
}

void write_string2unsigned(std::map<std::string, unsigned>& map1, std::ofstream& outf) {
	outf << map1.size() << std::endl;
	std::map<std::string, unsigned>::iterator iter1 = map1.begin();
	for (; iter1 != map1.end(); iter1++) {
		outf << iter1->first<<"\3"<< iter1->second<< std::endl;
	}
	outf << "\2end" << std::endl;
}

void write_unsigned2string(std::map<unsigned, std::string>& map1, std::ofstream& outf) {
	outf << map1.size() << std::endl;
	std::map<unsigned, std::string>::iterator iter1 = map1.begin();
	for (; iter1 != map1.end(); iter1++) {
		outf << iter1->first << "\3" << iter1->second << std::endl;
	}
	outf << "\2end" << std::endl;
}

void Corpus::save_config(const char* path)
{
	std::ofstream outf(path);
	outf << nwords << std::endl;
	outf << nactions << std::endl;
	outf << npos << std::endl;

	outf << max << std::endl;
	outf << maxPos << std::endl;
	//outf << maxChars << std::endl;

	outf << "\1wordsToInt" << std::endl;
	write_string2unsigned(wordsToInt, outf);
	outf << "\1intToWords" << std::endl;
	write_unsigned2string(intToWords, outf);

	outf << "\1posToInt" << std::endl;
	write_string2unsigned(posToInt, outf);
	outf << "\1intToPos" << std::endl;
	write_unsigned2string(intToPos, outf);

	/*outf << "\1charsToInt" << std::endl;
	write_string2unsigned(charsToInt, outf);
	outf << "\1intToChars" << std::endl;
	write_unsigned2string(intToChars, outf);*/

	outf << "\1actions" << std::endl;
	outf << actions.size() << std::endl;
	for (int i = 0; i < actions.size(); i++) {
		outf << actions[i] << std::endl;
	}
	outf << "\2actions" << std::endl;
}

void split_string(char* org, char** p1, char** p2,bool rev = false) {
	char* fid = 0;
	if (rev) 
		fid = strrchr(org, '\3');
	else
		fid = strchr(org, '\3');
	*p1 = org;
	if (fid == 0) {
		*p2 = 0;
		return;
	}
	*fid = 0;
	*p2 = fid + 1;
}

void read_string2unsigned(std::map<std::string, unsigned>& map1, std::ifstream& inf, char* buffer) {
	map1.clear();
	size_t sz = 0;
	inf >> sz;

	while (!inf.eof()) {
		inf.getline(buffer, 1024 * 33);
		if (buffer[0] == 0)
			continue;
		if (buffer[0] == '\2' && !strcmp(buffer + 1, "end"))
			break;
		char* key = 0;
		char* value = 0;
		split_string(buffer, &key, &value,true);
		map1[key] = atoi(value);
	}
	assert(map1.size() == sz);
}

void read_unsigned2string(std::map<unsigned, std::string>& map1, std::ifstream& inf, char* buffer) {
	map1.clear();
	size_t sz = 0;
	inf >> sz;

	while (!inf.eof()) {
		inf.getline(buffer, 1024 * 33);
		if (buffer[0] == 0)
			continue;
		if (buffer[0] == '\2' && !strcmp(buffer + 1, "end"))
			break;
		char* key = 0;
		char* value = 0;
		split_string(buffer, &key, &value, false);
		map1[atoi(key)] = value;
	}
	assert(map1.size() == sz);
}

void read_vecstring(std::vector<std::string>& actions, std::ifstream& inf, char* buffer) {
	actions.clear();
	size_t sz = 0;
	inf >> sz;

	while (!inf.eof()) {
		inf.getline(buffer, 1024 * 33);
		if (buffer[0] == 0)
			continue;
		if (buffer[0] == '\2' && !strcmp(buffer + 1, "actions"))
			break;
		actions.push_back(buffer);
	}
	assert(actions.size() == sz);
}


void Corpus::load_config(const char* path)
{
	std::ifstream inf(path);
	inf >> nwords;
	inf >> nactions;
	inf >> npos;

	inf >> max;
	inf >> maxPos;

	char* buffer = new char[1024 * 33];
	while (!inf.eof()) {
		inf.getline(buffer, 1024 * 33);
		if (buffer[0] == '\1') {
			if (!strcmp(buffer + 1, "wordsToInt")) {
				read_string2unsigned(wordsToInt, inf, buffer);
			}
			if (!strcmp(buffer + 1, "intToWords")) {
				read_unsigned2string(intToWords, inf, buffer);
			}

			if (!strcmp(buffer + 1, "posToInt")) {
				read_string2unsigned(posToInt, inf, buffer);
			}
			if (!strcmp(buffer + 1, "intToPos")) {
				read_unsigned2string(intToPos, inf, buffer);
			}

			if (!strcmp(buffer + 1, "actions")) {
				read_vecstring(actions, inf, buffer);
			}
		}
	}
}


void Corpus::load_correct_actions(std::string file)
{
	std::ifstream actionsFile(file);
	std::string lineS;

	int count = -1;
	int sentence = -1;
	bool initial = false;
	bool first = true;
	wordsToInt[Corpus::BAD0] = 0;
	intToWords[0] = Corpus::BAD0;
	wordsToInt[Corpus::UNK] = 1; // unknown symbol
	intToWords[1] = Corpus::UNK;
	assert(max == 0);
	assert(maxPos == 0);
	max = 2;
	maxPos = 1;

	//charsToInt[BAD0] = 1;
	//intToChars[1] = "BAD0";
	//maxChars = 1;

	std::vector<unsigned> current_sent;
	std::vector<unsigned> current_sent_pos;
	while (getline(actionsFile, lineS)){
		ReplaceStringInPlace(lineS, "-RRB-", "_RRB_");
		ReplaceStringInPlace(lineS, "-LRB-", "_LRB_");
		if (lineS.empty()) {
			count = 0;
			if (!first) {
				sentences[sentence] = current_sent;
				sentencesPos[sentence] = current_sent_pos;
			}

			sentence++;
			nsentences = sentence;

			initial = true;
			current_sent.clear();
			current_sent_pos.clear();
		}
		else if (count == 0) {
			first = false;
			count = 1;
			if (initial) {
				// the initial line in each sentence may look like:
				// [][the-det, cat-noun, is-verb, on-adp, the-det, mat-noun, ,-punct, ROOT-ROOT]
				// first, get rid of the square brackets.
				lineS = lineS.substr(3, lineS.size() - 4);
				// read the initial line, token by token "the-det," "cat-noun," ...
				std::istringstream iss(lineS);
				do {
					std::string word;
					iss >> word;
					if (word.size() == 0) { continue; }
					// remove the trailing comma if need be.
					if (word[word.size() - 1] == ',') {
						word = word.substr(0, word.size() - 1);
					}
					// split the string (at '-') into word and POS tag.
					size_t posIndex = word.rfind('-');
					if (posIndex == std::string::npos) {
						std::cerr << "cant find the dash in '" << word << "'" << std::endl;
					}
					assert(posIndex != std::string::npos);
					std::string pos = word.substr(posIndex + 1);
					word = word.substr(0, posIndex);
					// new POS tag
					if (posToInt[pos] == 0) {
						posToInt[pos] = maxPos;
						intToPos[maxPos] = pos;
						npos = maxPos;
						maxPos++;
					}

					// new word
					if (wordsToInt[word] == 0) {
						wordsToInt[word] = max;
						intToWords[max] = word;
						nwords = max;
						max++;

						/*unsigned j = 0;
						while (j < word.length()) {
							std::string wj = "";
							for (unsigned h = j; h < j + UTF8Len(word[j]); h++) {
								wj += word[h];
							}
							if (charsToInt[wj] == 0) {
								charsToInt[wj] = maxChars;
								intToChars[maxChars] = wj;
								maxChars++;
							}
							j += UTF8Len(word[j]);
						}*/
					}

					current_sent.push_back(wordsToInt[word]);
					current_sent_pos.push_back(posToInt[pos]);
				} while (iss);
			}
			initial = false;
		}
		else if (count == 1){
			//hack label
#ifdef LABEL_HACK
			auto hyp_rel = lineS;
			int first_char_in_rel = hyp_rel.find('(');
			if (first_char_in_rel >= 0) {
				lineS = lineS.substr(0, first_char_in_rel) + "(NONE)";
			}
#endif
			//------------------------------
			int i = 0;
			bool found = false;
			for (auto a : actions) {
				if (a == lineS) {
					std::vector<unsigned> a = correct_act_sent[sentence];
					a.push_back(i);
					correct_act_sent[sentence] = a;
					found = true;
				}
				i++;
			}
			if (!found) {
				actions.push_back(lineS);
				std::vector<unsigned> a = correct_act_sent[sentence];
				a.push_back(actions.size() - 1);
				correct_act_sent[sentence] = a;
			}
			count = 0;
		}
	}

	// Add the last sentence.
	if (current_sent.size() > 0) {
		sentences[sentence] = current_sent;
		sentencesPos[sentence] = current_sent_pos;
		sentence++;
		nsentences = sentence;
	}

	actionsFile.close();

	nactions = actions.size();
	std::cerr << "nactions:" << nactions << "\n";
	std::cerr << "nwords:" << nwords << "\n";
	std::cerr << "npos:" << npos << "\n";
}

unsigned Corpus::get_or_add_word(const std::string& word) {
	unsigned& id = wordsToInt[word];
	if (id == 0) {
		id = max;
		++max;
		intToWords[id] = word;
		nwords = max;
	}
	return id;
}

unsigned Corpus::get_word(const std::string& word) {
	unsigned& id = wordsToInt[word];
	return id;
}

void Corpus::load_correct_actionsDev(std::string file) {
	std::ifstream actionsFile(file);
	std::string lineS;

	assert(maxPos > 1);
	assert(max > 3);
	int count = -1;
	int sentence = -1;
	bool initial = false;
	bool first = true;
	std::vector<unsigned> current_sent;
	std::vector<unsigned> current_sent_pos;
	std::vector<std::string> current_sent_str;
	while (getline(actionsFile, lineS)) {
		ReplaceStringInPlace(lineS, "-RRB-", "_RRB_");
		ReplaceStringInPlace(lineS, "-LRB-", "_LRB_");
		if (lineS.empty()) {
			// an empty line marks the end of a sentence.
			count = 0;
			if (!first) {
				sentencesDev[sentence] = current_sent;
				sentencesPosDev[sentence] = current_sent_pos;
				sentencesStrDev[sentence] = current_sent_str;
			}

			sentence++;
			nsentencesDev = sentence;

			initial = true;
			current_sent.clear();
			current_sent_pos.clear();
			current_sent_str.clear();
		}
		else if (count == 0) {
			first = false;
			//stack and buffer, for now, leave it like this.
			count = 1;
			if (initial) {
				// the initial line in each sentence may look like:
				// [][the-det, cat-noun, is-verb, on-adp, the-det, mat-noun, ,-punct, ROOT-ROOT]
				// first, get rid of the square brackets.
				lineS = lineS.substr(3, lineS.size() - 4);
				// read the initial line, token by token "the-det," "cat-noun," ...
				std::istringstream iss(lineS);
				do {
					std::string word;
					iss >> word;
					if (word.size() == 0) { continue; }
					// remove the trailing comma if need be.
					if (word[word.size() - 1] == ',') {
						word = word.substr(0, word.size() - 1);
					}
					// split the string (at '-') into word and POS tag.
					size_t posIndex = word.rfind('-');
					assert(posIndex != std::string::npos);
					std::string pos = word.substr(posIndex + 1);
					word = word.substr(0, posIndex);
					// new POS tag
					if (posToInt[pos] == 0) {
						posToInt[pos] = maxPos;
						intToPos[maxPos] = pos;
						npos = maxPos;
						maxPos++;
					}
					// add an empty string for any token except OOVs (it is easy to 
					// recover the surface form of non-OOV using intToWords(id)).
					current_sent_str.push_back("");
					// OOV word
					if (wordsToInt[word] == 0) {
						if (USE_SPELLING) {
							max = nwords + 1;
							//std::cerr<< "max:" << max << "\n";
							wordsToInt[word] = max;
							intToWords[max] = word;
							nwords = max;
						}
						else {
							// save the surface form of this OOV before overwriting it.
							current_sent_str[current_sent_str.size() - 1] = word;
							word = Corpus::UNK;
						}
					}
					current_sent.push_back(wordsToInt[word]);
					current_sent_pos.push_back(posToInt[pos]);
				} while (iss);
			}
			initial = false;
		}
		else if (count == 1) {
			//hack label
#ifdef LABEL_HACK
			auto hyp_rel = lineS;
			int first_char_in_rel = hyp_rel.find('(');
			if (first_char_in_rel >= 0) {
				lineS = lineS.substr(0, first_char_in_rel) + "(NONE)";
			}
#endif
			//------------------------------

			auto actionIter = std::find(actions.begin(), actions.end(), lineS);
			if (actionIter != actions.end()) {
				unsigned actionIndex = std::distance(actions.begin(), actionIter);
				correct_act_sentDev[sentence].push_back(actionIndex);
			}
			else {
				// TODO: right now, new actions which haven't been observed in training
				// are not added to correct_act_sentDev. This may be a problem if the
				// training data is little.
			}
			count = 0;
		}
	}

	// Add the last sentence.
	if (current_sent.size() > 0) {
		sentencesDev[sentence] = current_sent;
		sentencesPosDev[sentence] = current_sent_pos;
		sentencesStrDev[sentence] = current_sent_str;
		sentence++;
		nsentencesDev = sentence;
	}

	actionsFile.close();
}

void split_string(std::vector<std::string>& vec, std::string v, std::string tok) {
	size_t pos = 0;
	size_t newpos = 0;
	while ((newpos = v.find(tok, pos)) != std::string::npos) {
		std::string sub = v.substr(pos, newpos - pos);
		vec.push_back(sub);
		pos = newpos+1;
	}
	std::string sub = v.substr(pos);
	if (sub.length()>0)
		vec.push_back(sub);
}

void Corpus::load_Test(std::string file)
{
	std::ifstream actionsFile(file);
	std::string lineS;
	int sentence = -1;
	std::vector<unsigned> current_sent;
	std::vector<unsigned> current_sent_pos;
	std::vector<std::string> current_sent_str;
	while (getline(actionsFile, lineS)) {
		ReplaceStringInPlace(lineS, "-RRB-", "_RRB_");
		ReplaceStringInPlace(lineS, "-LRB-", "_LRB_");

		std::vector<std::string> sen;
		split_string(sen, lineS, "\t");
		for (int i = 0; i < sen.size(); i++) {
			std::string word = sen[i];
			if (word.size() == 0) { continue; }
			if (word[word.size() - 1] == ',') {
				word = word.substr(0, word.size() - 1);
			}
			// split the string (at '-') into word and POS tag.
			size_t posIndex = word.rfind('_');
			assert(posIndex != std::string::npos);
			std::string pos = word.substr(posIndex + 1);
			word = word.substr(0, posIndex);
			// new POS tag
			if (posToInt[pos] == 0) {
				posToInt[pos] = maxPos;
				intToPos[maxPos] = pos;
				npos = maxPos;
				maxPos++;
			}
			current_sent_str.push_back("");
			// OOV word
			if (wordsToInt[word] == 0) {
				current_sent_str[current_sent_str.size() - 1] = word;
				word = Corpus::UNK;
			}
			current_sent.push_back(wordsToInt[word]);
			current_sent_pos.push_back(posToInt[pos]);
		}

		sentencesDev[sentence] = current_sent;
		sentencesPosDev[sentence] = current_sent_pos;
		sentencesStrDev[sentence] = current_sent_str;

		sentence++;
		nsentencesDev = sentence;

		current_sent.clear();
		current_sent_pos.clear();
		current_sent_str.clear();
	}

	actionsFile.close();
}