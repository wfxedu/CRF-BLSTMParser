#include "layersCRF.h"
#include "TreeReaderCRF.h"
namespace CRFParser {
void word_layer::Init_layer(parser_config& cfg)
{
	layer::Init_layer(cfg);
	Model* model = cfg.model;
	const unordered_map<unsigned, vector<float>>& pretrained = *cfg.pretrained;
	//////////////////////////////////////////////////////////////////////////
	p_w = model->add_lookup_parameters(parser_config::VOCAB_SIZE, Dim(parser_config::INPUT_DIM, 1));
	p_w2l = model->add_parameters(Dim(parser_config::LSTM_INPUT_DIM, parser_config::INPUT_DIM));
	p_pre2l = model->add_parameters(Dim(parser_config::LSTM_INPUT_DIM, parser_config::HIDDEN_DIM));
	p_ib = model->add_parameters(Dim(parser_config::LSTM_INPUT_DIM, 1));
	if (parser_config::USE_POS) {
		p_p = model->add_lookup_parameters(parser_config::POS_SIZE, Dim(parser_config::POS_DIM, 1));
		p_p2l = model->add_parameters(Dim(parser_config::LSTM_INPUT_DIM, parser_config::POS_DIM));
	}
	if (pretrained.size() > 0) {
		p_t = model->add_lookup_parameters(parser_config::VOCAB_SIZE, Dim(parser_config::PRETRAINED_DIM, 1));
		for (auto it : pretrained)
			p_t->Initialize(it.first, it.second);
		p_t2l = model->add_parameters(Dim(parser_config::LSTM_INPUT_DIM, parser_config::PRETRAINED_DIM));
	}
	else {
		p_t = nullptr;
		p_t2l = nullptr;
	}
}

void word_layer::Init_Graph(ComputationGraph* hg)
{
	layer::Init_Graph(hg);
	//for word layer
	ib = parameter(*hg, p_ib);
	w2l = parameter(*hg, p_w2l);
	pre2l = parameter(*hg, p_pre2l);
	if (parser_config::USE_POS)
		p2l = parameter(*hg, p_p2l);
	if (p_t2l)
		t2l = parameter(*hg, p_t2l);
}

Expression word_layer::build(unsigned word_id, unsigned pos_id, unsigned orgword_id, Expression pre)
{
	//assert(sent[i] < VOCAB_SIZE);
	Expression w = lookup(*m_hg, p_w, word_id);

	vector<Expression> args = { ib, w2l, w,pre2l, pre }; // learn embeddings
	if (parser_config::USE_POS) { // learn POS tag?
		Expression p = lookup(*m_hg, p_p, pos_id);
		args.push_back(p2l);
		args.push_back(p);
	}
	if (p_t && m_cfg->pretrained->count(orgword_id)) {  // include fixed pretrained vectors?
		Expression t = const_lookup(*m_hg, p_t, orgword_id);
		args.push_back(t2l);
		args.push_back(t);
	}
	return rectify(affine_transform(args));
}
//----------------------------------------------


void elements_layer::init(Model* model, word_layer* wl, int word_dim, int pos_dim) {
	elements_fwrnn = new LSTMBuilder(parser_config::LAYERS, parser_config::INPUT_DIM, parser_config::INPUT_DIM_DL, model);
	elements_bwrnn = new LSTMBuilder(parser_config::LAYERS, parser_config::INPUT_DIM, parser_config::INPUT_DIM_DL, model);
	elements_fwrnn->set_dropout(parser_config::DROP_OUT);
	elements_bwrnn->set_dropout(parser_config::DROP_OUT);

	p_w = wl->p_w;
	p_p = wl->p_p;
	p_t = wl->p_t;
	int ele_len = pos_dim + word_dim;
	if (p_t)
		ele_len += parser_config::PRETRAINED_DIM;
	else
		printf("Missing pretrain embedding!\n");
	p_Wwp = model->add_parameters(Dim(parser_config::INPUT_DIM, ele_len));
	p_Bwp = model->add_parameters(Dim(parser_config::INPUT_DIM, 1));
}

void elements_layer::init_expression(ComputationGraph* hg) {
	Wwp = parameter(*hg, p_Wwp);
	Bwp = parameter(*hg, p_Bwp);
	elements_fwrnn->new_graph(*hg);
	elements_bwrnn->new_graph(*hg);
	elements_fwrnn->start_new_sequence();
	elements_bwrnn->start_new_sequence();
}

vector<Expression>& elements_layer::build(sentence_tr& sen, ComputationGraph* hg)
{
	;

	int sen_size = sen.size();
	vector<Expression> raw_t(sen_size);
	for (int i = 0;i < sen_size;i++) { //excluding root
		element_tr& ele = sen[i];
		Expression t1;
		if(p_t==0)
			t1= concatenate({ lookup(*hg, p_w, ele.form_useid),lookup(*hg, p_p, ele.pos_useid) });
		else
			t1 = concatenate( { lookup(*hg, p_w, ele.form_useid),
								lookup(*hg, p_p, ele.pos_useid),
								const_lookup(*hg, p_t, ele.form_id) });
		//raw_t[i] = tanh(affine_transform({ Bwp ,Wwp ,t1 }));
		raw_t[i] = rectify(affine_transform({ Bwp ,Wwp ,t1 }));
	}
	//-------------------------
	vector<Expression> fwd_t(sen_size);
	for (int i = 0;i < sen_size;i++) {
		elements_fwrnn->add_input(raw_t[i]);
		fwd_t[i] = elements_fwrnn->back();
	}
	//-------------------------
	vector<Expression> bwd_t(sen_size);
	for (int i = sen_size - 1;i >= 0;i--) {
		elements_bwrnn->add_input(raw_t[i]);
		bwd_t[i] = elements_bwrnn->back();
	}
	//-------------------------
	embeddings_sentence.resize(sen_size);
	for (int i = 0;i < sen_size;i++) {
		embeddings_sentence[i] = fwd_t[i] + bwd_t[i];
	}
	return embeddings_sentence;
}

}//namespace CRFParser {
////////////////////////////////////////////////////////////////////////////////////
