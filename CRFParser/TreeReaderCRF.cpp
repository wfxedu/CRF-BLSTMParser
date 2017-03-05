#include "TreeReaderCRF.h"
namespace CRFParser {
const char* treebank::UNK = "UNK";
const char* treebank::UNK_POS = "UNK_POS";
const char* treebank::BAD0 = "<BAD0>";
const char* treebank::ROOT = "ROOT";
const char* treebank::NONE_POS = "NONE_POS";
const char* treebank::NONE_REL = "NONE_REL";

void DatasetTB::load_train(std::string path)
{
	training.init(&words2Int, &pos2Int, &prel2Int,false);
	training.load(path.c_str());
}

void DatasetTB::load_test(std::string path) {
	testing.init(&words2Int, &pos2Int, &prel2Int,true);
	testing.load(path.c_str());
}

void DatasetTB::load_dev(std::string path) {
	deving.init(&words2Int, &pos2Int, &prel2Int, true);
	deving.load(path.c_str());
}

}//namespace CRFParser {