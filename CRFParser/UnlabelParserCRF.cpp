#include "UnlabelParserCRF.h"


namespace CRFParser {

#define parser_config_LABEL_SIZE_LOC 1
ParserBuilderUnlabel::ParserBuilderUnlabel(parser_config& cfg) :
	p_r(cfg.model->add_lookup_parameters(parser_config_LABEL_SIZE_LOC, Dim(parser_config::REL_DIM, 1))),
	p_dist(cfg.model->add_lookup_parameters(parser_config::MAX_DISTANCE, Dim(parser_config::DISTANCE_DIM, 1)))
{
	base_pool = cfg.base_pool;
	wordlayer.Init_layer(cfg);
	inputlayer.init(cfg.model, &wordlayer, parser_config::INPUT_DIM, parser_config::POS_DIM);
	segment_fwrnn = new LSTMBuilder(parser_config::LAYERS, parser_config::INPUT_DIM_DL, parser_config::SEGMENT_DIM, cfg.model);
	segment_fwrnn->set_dropout(parser_config::DROP_OUT);
	for (int i = 0;i < 2;i++) {
		p_Wo[i] = cfg.model->add_parameters(Dim(parser_config_LABEL_SIZE_LOC, parser_config::HIDDEN_DIM_DH));
		p_Bo[i] = cfg.model->add_parameters(Dim(parser_config_LABEL_SIZE_LOC, 1));
		p_segment_gard[i] = cfg.model->add_parameters(Dim(parser_config::SEGMENT_DIM, 1));
		//p_Bo[i]->need_regression = false;
		//p_segment_gard[i]->need_regression = false;
	}

	feaarray[FEA_MIN].init(cfg.model, parser_config::INPUT_DIM_DL, parser_config::HIDDEN_DIM_DH);
	feaarray[FEA_MAX].init(cfg.model, parser_config::INPUT_DIM_DL, parser_config::HIDDEN_DIM_DH);
	feaarray[FEA_DIS].init(cfg.model, parser_config::DISTANCE_DIM, parser_config::HIDDEN_DIM_DH);
	feaarray[FEA_S0].init(cfg.model, parser_config::SEGMENT_DIM, parser_config::HIDDEN_DIM_DH);
	feaarray[FEA_S1].init(cfg.model, parser_config::SEGMENT_DIM, parser_config::HIDDEN_DIM_DH);
	feaarray[FEA_S2].init(cfg.model, parser_config::SEGMENT_DIM, parser_config::HIDDEN_DIM_DH);

}

void ParserBuilderUnlabel::evaluate(ComputationGraph* hg, sentence_tr& sen)
{
	inputlayer.init_expression(hg);
	segment_fwrnn->new_graph(*hg);
	segment_fwrnn->start_new_sequence();
	for (int i = 0;i < 6;i++)
		feaarray[i].init_expression(hg);
	Expression Wo[2];
	Expression Bo[2];
	Expression segment_gard[2];
	for (int i = 0;i < 2;i++) {
		Wo[i] = parameter(*hg, p_Wo[i]);
		Bo[i] = parameter(*hg, p_Bo[i]);
		segment_gard[i] = parameter(*hg, p_segment_gard[i]);
	}

	int sen_size = sen.size();
	vector<Expression>& embeddings = inputlayer.build(sen, hg);
	vector<Expression> segment(sen_size);
	for (int i = 0;i < sen_size;i++) {
		segment_fwrnn->add_input(embeddings[i]);
		segment[i] = segment_fwrnn->back();
	}
	//----------------------------------------------------------
	hg->incremental_forward();

	long scores_size = sen_size*sen_size*parser_config_LABEL_SIZE_LOC;
	temp_scores = (double*)base_pool->allocate(scores_size * sizeof(double));
	double* scores_label = temp_scores;
	for (int i = 0;i < scores_size;i++)
		scores_label[i] = DOUBLE_LARGENEG_P1;
	exp_scores.clear();
	exp_scores.resize(sen_size*sen_size, 0);
	//double* scores = (double*)base_pool->allocate(sen_size*sen_size * sizeof(double));
	//----------------------------------------------------------
	for (int h = 0;h < sen_size;h++) {
		for (int m = 1;m < sen_size;m++) {
			if (m == h)
				continue;
			int min_w, max_w, d;
			if (h > m) {
				min_w = m;
				max_w = h;
				d = 1; //left arc
			}
			else {
				min_w = h;
				max_w = m;
				d = 0; //right arc
			}
			int dist = max_w - min_w;

			Expression out_exp = feaarray[FEA_MIN].build(embeddings[min_w], d);
			out_exp = out_exp + feaarray[FEA_MAX].build(embeddings[max_w], d);
			out_exp = out_exp + feaarray[FEA_DIS].build(lookup(*hg, p_dist, dist), d);
			out_exp = out_exp + feaarray[FEA_S0].build(min_w>0 ? segment[min_w - 1] : segment_gard[0], d);
			out_exp = out_exp + feaarray[FEA_S1].build(segment[max_w] - segment[min_w], d);
			out_exp = out_exp + feaarray[FEA_S2].build(
				max_w< sen_size - 1 ? segment[sen_size - 1] - segment[max_w] : segment_gard[1], d);
			//Expression out_exp_a = (cube(out_exp) + out_exp);
			Expression out_exp_a = tanh(cube(out_exp) + out_exp); 
			Expression res = affine_transform({ Bo[d] ,Wo[d],out_exp_a });

			int idx = get_index2(sen_size, h, m);
			exp_scores[idx] = (Expression*)base_pool->allocate(sizeof(Expression));
			*exp_scores[idx] = res;

			hg->incremental_forward();
			//vector<float> tv = as_vector(res.value());
			scores_label[idx] = as_scalar(res.value());;

			/*for (int l = 0;l < parser_config::LABEL_SIZE;l++) {
			int idx = get_index2(sen_size, h, m, l, parser_config::LABEL_SIZE);
			scores_label[idx] = tv[l];
			}*/
		}
	}

}

void ParserBuilderUnlabel::decode(ComputationGraph* hg, sentence_tr& sen, vector<pair<int, int>>& outsen)
{
	outsen.resize(sen.size(), pair<int, int>(-1, -1));
	evaluate(hg, sen);
	double* scores_label = temp_scores;
	//TMP1_get_sumlabel(sen_size*sen_size, LABEL_SIZE, scores_label, scores);
	vector<int>* res = decodeProjective(sen.size(), scores_label);
	for (int i = 0;i < sen.size();i++) {
		outsen[i].first = (*res)[i];
	}
	delete res;
}

double ParserBuilderUnlabel::construct_grad(ComputationGraph* hg, sentence_tr& sen, vector<pair<int, int>>& outsen)
{
	int length = sen.size();
	outsen.resize(length, pair<int, int>(-1, -1));
	//double* marginals = (double*)base_pool->allocate(length*length*parser_config::LABEL_SIZE * sizeof(double));

	evaluate(hg, sen);
	double* scores_label = temp_scores;
	//------------------
	vector<int>* res = decodeProjective(length, scores_label);
	for (int i = 0;i < sen.size();i++) {
		outsen[i].first = (*res)[i];
	}
	delete res;
	//------------------
	double* marginals = encodeMarginals(length, scores_label);
	double log_p = 0;
	//int key_assign = get_index2(length, i, j);
	vector<Expression> args;
	for (int h = 0;h < length;h++) {
		for (int m = 1;m < length;m++) {
			if (m == h)
				continue;
			int idx = get_index2(length, h, m);
			if (exp_scores[idx] == 0) {
				printf("ERROR!");
				continue;
			}
			double gs = 0;
			if (sen[m].parent == h) {
				gs = -1;
				log_p += scores_label[idx];
			}
			args.push_back((gs + marginals[idx])*(*exp_scores[idx])/*pick(*exp_scores[idxl], l)*/);

		}
	}
	if (args.empty()) {
		printf("args.empty len %d\n", length);
		return Negative_Infinity;
	}
	delete[] marginals;
	Expression result = sum(args);
	return log_p;;
}

}//namespace CRFParser {
//
//double ParserBuilderUnlabel::margin_loss(ComputationGraph* hg, sentence_tr& sen, vector<pair<int, int>>& outsen)
//{
//	int length = sen.size();
//	outsen.resize(length, pair<int, int>(-1, -1));
//	double* marginals = (double*)base_pool->allocate(length*length*parser_config::LABEL_SIZE * sizeof(double));
//
//	evaluate(hg, sen);
//	double* scores_label = temp_scores;
//	decodeProjectiveL(length, scores_label, parser_config::LABEL_SIZE, outsen);
//	//---------------------
//	double loss = 0;
//	for (int i = 1;i < sen.size();i++) {
//		if (sen[i].parent == outsen[i].first) {
//			if (sen[i].prel_id != outsen[i].second)
//				loss += 0.5;
//		}
//		else
//			loss += 1;
//	}
//	//----------------------
//	double score_tt = 0;
//	vector<Expression> args, arg2;
//	for (int i = 1;i < sen.size();i++) {
//		int gold_idx = get_index2(length, sen[i].parent, i, sen[i].prel_id, parser_config::LABEL_SIZE);
//		int gold_idxl = get_index2(length, sen[i].parent, i);
//		score_tt -= scores_label[gold_idx];
//		assert(exp_scores[gold_idxl]);
//		assert(sen[i].prel_id >= 0);
//
//		int pre_idx = get_index2(length, outsen[i].first, i, outsen[i].second, parser_config::LABEL_SIZE);
//		int pre_idxl = get_index2(length, outsen[i].first, i);
//		score_tt += scores_label[pre_idx];
//		assert(exp_scores[pre_idxl]);
//		assert(outsen[i].second >= 0);
//
//		args.push_back(pick(*exp_scores[gold_idxl], sen[i].prel_id));
//		for (int l = 0;l < parser_config::LABEL_SIZE;l++) {
//			arg2.push_back(pick(*exp_scores[pre_idxl], l));
//			if (l != sen[i].prel_id)
//				arg2.push_back(pick(*exp_scores[gold_idxl], l));
//
//		}
//	}
//	if (args.empty()) {
//		//printf("args.empty len %d\n", length);
//		return Negative_Infinity;
//	}
//	Expression out = sum(arg2) / arg2.size() - sum(args) / args.size() + 1;
//	return score_tt + 1;
//}

