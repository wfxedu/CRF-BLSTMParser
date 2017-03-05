#pragma once
#include "TreeReaderCRF.h"
using namespace CRFParser;
class util
{
public:
	static bool load_EmbedDict(DatasetTB& sr_corpus, EmbedDict& pretrained, const char* path, bool isorg = true);
};

inline unsigned compute_correct(const map<int, int>& ref, const map<int, int>& hyp, unsigned len) {
	unsigned res = 0;
	for (unsigned i = 0; i < len; ++i) {
		auto ri = ref.find(i);
		auto hi = hyp.find(i);
		assert(ri != ref.end());
		assert(hi != hyp.end());
		if (ri->second == hi->second) ++res;
	}
	return res;
}


inline void output_conll(const vector<unsigned>& sentence, const vector<unsigned>& pos,
	const vector<string>& sentenceUnkStrings,
	const map<unsigned, string>& intToWords,
	const map<unsigned, string>& intToPos,
	const map<int, int>& hyp, const map<int, string>& rel_hyp, unsigned unk_tok)
{
	for (unsigned i = 0; i < (sentence.size() - 1); ++i) {
		auto index = i + 1;
		assert(i < sentenceUnkStrings.size() &&
			((sentence[i] == unk_tok &&
			sentenceUnkStrings[i].size() > 0) ||
			(sentence[i] != unk_tok &&
			sentenceUnkStrings[i].size() == 0 &&
			intToWords.find(sentence[i]) != intToWords.end())));
		string wit = (sentenceUnkStrings[i].size() > 0) ?
			sentenceUnkStrings[i] : intToWords.find(sentence[i])->second;
		auto pit = intToPos.find(pos[i]);
		assert(hyp.find(i) != hyp.end());
		auto hyp_head = hyp.find(i)->second + 1;
		if (hyp_head == (int)sentence.size()) hyp_head = 0;
		auto hyp_rel_it = rel_hyp.find(i);
		assert(hyp_rel_it != rel_hyp.end());
		auto hyp_rel = hyp_rel_it->second;
		size_t first_char_in_rel = hyp_rel.find('(') + 1;
		size_t last_char_in_rel = hyp_rel.rfind(')') - 1;
		hyp_rel = hyp_rel.substr(first_char_in_rel, last_char_in_rel - first_char_in_rel + 1);
		cout << index << '\t'       // 1. ID 
			<< wit << '\t'         // 2. FORM
			<< "_" << '\t'         // 3. LEMMA 
			<< "_" << '\t'         // 4. CPOSTAG 
			<< pit->second << '\t' // 5. POSTAG
			<< "_" << '\t'         // 6. FEATS
			<< hyp_head << '\t'    // 7. HEAD
			<< hyp_rel << '\t'     // 8. DEPREL
			<< "_" << '\t'         // 9. PHEAD
			<< "_" << endl;        // 10. PDEPREL
	}
	cout << endl;
}


