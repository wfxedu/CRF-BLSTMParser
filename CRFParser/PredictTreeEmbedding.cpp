#include "PredictTreeEmbedding.h"
#include "SRParser.h"
#include <algorithm>
#include <functional>

void elements_layer::init(Model* model, word_layer* wl, int word_dim, int pos_dim) {
	int ele_len = pos_dim + word_dim;
	elements_fwrnn = new LSTMBuilder(parser_config::LAYERS, parser_config::HIDDEN_DIM, parser_config::HIDDEN_DIM , model);
	elements_bwrnn = new LSTMBuilder(parser_config::LAYERS, parser_config::HIDDEN_DIM, parser_config::HIDDEN_DIM, model);
	elements_fwrnn->set_dropout(parser_config::DROP_OUT);
	elements_bwrnn->set_dropout(parser_config::DROP_OUT);

	p_w = wl->p_w;
	p_p = wl->p_p;
	p_Wwp = model->add_parameters(Dim(parser_config::HIDDEN_DIM, ele_len));
	p_Bwp = model->add_parameters(Dim(parser_config::HIDDEN_DIM, 1));
	for (int i = 0;i < 2;i++) {
		p_start_sen[i] = model->add_parameters(Dim(parser_config::HIDDEN_DIM, 1));
		p_end_sen[i] = model->add_parameters(Dim(parser_config::HIDDEN_DIM, 1));
	}
}

void elements_layer::init_expression(ComputationGraph* hg) {
	Wwp = parameter(*hg, p_Wwp);
	Bwp = parameter(*hg, p_Bwp);
	for (int i = 0;i < 2;i++) {
		start_sen[i] = parameter(*hg, p_start_sen[i]);
		end_sen[i] = parameter(*hg, p_end_sen[i]);
	}
	elements_fwrnn->new_graph(*hg);
	elements_bwrnn->new_graph(*hg);
	elements_fwrnn->start_new_sequence();
	elements_bwrnn->start_new_sequence();
}

vector<Expression>& elements_layer::build(sentence_tr& sen, ComputationGraph* hg)
{
	int sen_size = sen.size();
	vector<Expression> raw_t(sen_size);
	for (int i = 0;i < sen_size;i++) { //excluding root
		element_tr& ele = sen[i];
		Expression t1 = concatenate({ lookup(*hg, p_w, ele.form_useid),lookup(*hg, p_p, ele.pos_id) });
		raw_t[i] = tanh(affine_transform({ Bwp ,Wwp ,t1 }));
	}
	//-------------------------
	vector<Expression> fwd_t(sen_size);
	for (int i = 0;i < sen_size;i++) {
		elements_fwrnn->add_input(raw_t[i]);
		fwd_t[i] = elements_fwrnn->back();
	}
	//-------------------------
	vector<Expression> bwd_t(sen_size);
	for (int i = sen_size-1;i >= 0;i--) {
		elements_bwrnn->add_input(raw_t[i]);
		bwd_t[i] = elements_bwrnn->back();
	}
	//-------------------------
	embeddings_sentence.resize(sen_size);
	for (int i = 0;i < sen_size;i++) {
		embeddings_sentence[i] = fwd_t[i]+ bwd_t[i];
	}
	return embeddings_sentence;
}
///////////////////////////////////////////////////
bool less_second( int  m1,  int  m2) {
	return m1 < m2;
}

bool great_second( int  m1,  int  m2) {
	return m1 > m2;
}

void build_sentree(sentence_tr& sen, sentree_tr& sentree) {
	sentree.clear();
	sentree.resize(sen.size());
	for (int i = 0;i < sen.size();i++) {
		element_tr& ele = sen[i];
		sentree[i].par_idx = i; //referent current node
		if (ele.parent == -1)
			continue;
		if (ele.parent > i) //left child
			sentree[ele.parent].left_children.push_back(i);
		else
			sentree[ele.parent].right_children.push_back(i);
	}

	for (int i = 0;i < sen.size();i++) {
		tree_item& ele = sentree[i];
		std::sort(ele.left_children.begin(), ele.left_children.end(), great_second);
		std::sort(ele.right_children.begin(), ele.right_children.end(), less_second);
	}
}
///////////////////////////////////////////////////

PredictTreeEmbedding::PredictTreeEmbedding(Model* model) :
	tree_rnnleft1(parser_config::HIDDEN_DIM, parser_config::HIDDEN_DIM, model),
	tree_rnnright1(parser_config::HIDDEN_DIM, parser_config::HIDDEN_DIM, model),
	tree_rnnleft2(parser_config::HIDDEN_DIM, parser_config::HIDDEN_DIM, model),
	tree_rnnright2(parser_config::HIDDEN_DIM, parser_config::HIDDEN_DIM, model),
	p_Wtreel(model->add_parameters(Dim(parser_config::HIDDEN_DIM, parser_config::HIDDEN_DIM * 2+ parser_config::REL_DIM))),
	p_Wtree(model->add_parameters(Dim(parser_config::HIDDEN_DIM, parser_config::HIDDEN_DIM * 2))),
	p_Btree(model->add_parameters(Dim(parser_config::HIDDEN_DIM, 1)))
{
	tree_rnnleft1.set_dropout(parser_config::DROP_OUT);
	tree_rnnright1.set_dropout(parser_config::DROP_OUT);
	tree_rnnleft2.set_dropout(parser_config::DROP_OUT);
	tree_rnnright2.set_dropout(parser_config::DROP_OUT);
}

void PredictTreeEmbedding::Init_layer(parser_config& cfg, word_layer* wl, LookupParameters* p) {
	Model* model = cfg.model;
	//////////////////////////////////////////////////////////////////////////
	p_r = p;
	base_input.init(model, wl, parser_config::INPUT_DIM, parser_config::POS_DIM);
}


void compound_treenode(int curI, sentree_tr& sentree, sentence_tr& sen, 
	PredictTreeEmbedding* pte, Expression& Wtreel, Expression& Btree, ComputationGraph* hg) {
	tree_item& item = sentree[curI];
	for (int lc = 0;lc < item.left_children.size();lc++) {
		int nd = item.left_children[lc];
		if (sentree[nd].pend_exp_enc_base.pg == 0)
			compound_treenode(nd, sentree, sen, pte, Wtreel, Btree, hg);

		pte->tree_rnnleft1.add_input_init(item.pend_exp_left0, sentree[nd].pend_exp_enc_base);
		pte->tree_rnnleft2.add_input_init(item.pend_exp_left, item.pend_exp_left0.H);
	}
	for (int rc = 0;rc < item.right_children.size();rc++) {
		int nd = item.right_children[rc];
		if (sentree[nd].pend_exp_enc_base.pg == 0)
			compound_treenode(nd, sentree, sen, pte, Wtreel, Btree, hg);

		pte->tree_rnnright1.add_input_init(item.pend_exp_right0, sentree[nd].pend_exp_enc_base);
		pte->tree_rnnright2.add_input_init(item.pend_exp_right, item.pend_exp_right0.H);
	}

	Expression er = lookup(*hg, pte->p_r, sen[item.par_idx].prel_id);
	Expression clr = concatenate({ item.pend_exp_left.H, item.pend_exp_right.H,er });
	item.pend_exp_enc_base = tanh(affine_transform({ Btree ,Wtreel,clr }));
}

vector<Expression> PredictTreeEmbedding::build(sentence_tr& sen, ComputationGraph* hg)
{
	sentree_tr sentree;
	build_sentree(sen, sentree);
	tree_rnnleft1.new_graph(*hg);
	tree_rnnright1.new_graph(*hg);
	tree_rnnleft2.new_graph(*hg);
	tree_rnnright2.new_graph(*hg);
	base_input.init_expression(hg);
	Expression Btree = parameter(*hg, p_Btree);
	Expression Wtreel = parameter(*hg, p_Wtreel);
	Expression Wtree = parameter(*hg, p_Wtree);

	vector<Expression>& embeddings = base_input.build(sen, hg);
	//----------------------------------------------------------
	hg->incremental_forward();

	for (int i = 0;i < sentree.size();i++) {
		tree_item& item = sentree[i];
	
		tree_rnnleft1.add_input_init(item.pend_exp_left0, embeddings[i]);
		tree_rnnleft2.add_input_init(item.pend_exp_left, item.pend_exp_left0.H);
		tree_rnnright1.add_input_init(item.pend_exp_right0, embeddings[i]);
		tree_rnnright2.add_input_init(item.pend_exp_right, item.pend_exp_right0.H);

		if (item.left_children.empty() && item.right_children.empty()) {
			Expression clr = concatenate({ item.pend_exp_left.H, item.pend_exp_right.H });
			item.pend_exp_enc_base = tanh(affine_transform({ Btree ,Wtree,clr }));
		}
	}
	//----------------------------------------------------------
	compound_treenode(sentree.size() - 1, sentree, sen, this, Wtreel, Btree, hg);

	vector<Expression> results(sentree.size());
	for (int i = 0;i < sentree.size();i++) {
		int idx = sentree[i].par_idx;
		if (idx == -1)
			continue;
		results[idx] = sentree[idx].pend_exp_enc_base;
	}
	hg->incremental_forward();
	return results;
}

