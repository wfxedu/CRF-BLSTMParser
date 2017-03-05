#include "util.h"

bool util::load_EmbedDict(DatasetTB& sr_corpus, EmbedDict& pretrained, const char* path, bool isorg)
{
	const unsigned kUNK = sr_corpus.words2Int.get(treebank::UNK, 0);
	pretrained[kUNK] = vector<float>(parser_config::PRETRAINED_DIM, 0);
	cerr << "Loading from " << path << " with" << parser_config::PRETRAINED_DIM << " dimensions\n";

	if (isorg) {
		ifstream in(path);
		string line;
		getline(in, line);
		vector<float> v(parser_config::PRETRAINED_DIM, 0);
		string word;
		while (getline(in, line)) {
			istringstream lin(line);
			lin >> word;
			for (unsigned i = 0; i < parser_config::PRETRAINED_DIM; ++i) lin >> v[i];
			int id = sr_corpus.words2Int.get(word, -1);
			if (id > 0)
				pretrained[id] = v;
		}
	}
	else {
		FILE *f;
		long long words, size, a, b;
		float *M;
		char *vocab;
		f = fopen(path, "rb");
		if (f == NULL) {
			printf("Input file not found\n");
			return false;
		}
		fread(&words, sizeof(long long), 1, f);
		fread(&size, sizeof(long long), 1, f);
		if (parser_config::PRETRAINED_DIM != size) {
			printf("ERROR: PRETRAINED_DIM[%d] != size[%lld]\n", parser_config::PRETRAINED_DIM, size);
			return false;
		}

		char vocab_temp[1023];
		vector<float> v(parser_config::PRETRAINED_DIM, 0);

		for (b = 0; b < words; b++) {
			a = 0;
			char* cur_vocab = vocab_temp;
			int len;
			fread(&len, sizeof(int), 1, f);
			fread(cur_vocab, 1, len, f);
			cur_vocab[len] = 0;

			fread(&len, sizeof(int), 1, f);
			if (len != 777) {
				printf("error r1!\n");
				return false;
			}
			if (b % 100000 == 0)
				printf("%s\n", cur_vocab);

			for (a = 0; a < size; a++) fread(&(v[a]), sizeof(float), 1, f);
			fread(&len, sizeof(int), 1, f);
			if (len != 777) {
				printf("error r2!\n");
				return false;
			}

			int id = sr_corpus.words2Int.get(cur_vocab,-1);
			if (id > 0)
				pretrained[id] = v;
		};
	}
}
