#include "SRParser.h"
#include "TreeReader.h"

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
void parser_state::Init_layer(parser_config& cfg)
{
	layer::Init_layer(cfg);
	Model* model = cfg.model;
	const EmbedDict& pretrained = *cfg.pretrained;
	//////////////////////////////////////////////////////////////////////////
	stack_lstm = LSTMBuilder(parser_config::LAYERS, parser_config::LSTM_INPUT_DIM, parser_config::HIDDEN_DIM, model);
	buffer_lstm = LSTMBuilder(parser_config::LAYERS, parser_config::LSTM_INPUT_DIM, parser_config::HIDDEN_DIM, model);
	action_lstm = LSTMBuilder(parser_config::LAYERS, parser_config::ACTION_DIM, parser_config::HIDDEN_DIM, model);
	stack_lstm.set_dropout(parser_config::DROP_OUT);
	buffer_lstm.set_dropout(parser_config::DROP_OUT);
	action_lstm.set_dropout(parser_config::DROP_OUT);

	p_action_start = model->add_parameters(Dim(parser_config::ACTION_DIM, 1));
	p_buffer_guard =model->add_parameters(Dim(parser_config::LSTM_INPUT_DIM, 1));
	p_stack_guard = model->add_parameters(Dim(parser_config::LSTM_INPUT_DIM, 1));
}

void parser_state::Init_Graph(ComputationGraph* hg)
{
	layer::Init_Graph(hg);

	stack_lstm.new_graph(*hg);
	buffer_lstm.new_graph(*hg);
	action_lstm.new_graph(*hg);

	stack_lstm.start_new_sequence();
	buffer_lstm.start_new_sequence();
	action_lstm.start_new_sequence();
	////////////////////////////////////////////////////////////////
	action_start = parameter(*hg, p_action_start);
}

void parser_state::Init_state(const t_unvec& raw_sent, const t_unvec& sent, 
	const t_unvec& sentPos, word_layer& layer, vector<Expression>& pre_embedding)
{
	int sen_sz = (int)sent.size();
	action_lstm.add_input(action_start);
	//------------------------------------------------------------------------
	buffer.resize(sen_sz + 1);  // variables representing word embeddings (possibly including POS info)
	bufferi.resize(sen_sz + 1);
	word2inner_mean.resize(sen_sz + 1);
	for (unsigned i = 0; i < sent.size(); ++i) {
		assert(sent[i] < parser_config::VOCAB_SIZE);
		buffer[sent.size() - i] = layer.build(sent[i], sentPos[i], raw_sent[i], pre_embedding[i]);
		bufferi[sent.size() - i] = i;
		word2inner_mean[i] = buffer[sent.size() - i];
	}
	// dummy symbol to represent the empty buffer
	buffer[0] = parameter(*m_hg, p_buffer_guard);
	bufferi[0] = -999;
	for (auto& b : buffer)
		buffer_lstm.add_input(b);
	//------------------------------------------------------------------------
	stack.push_back(parameter(*m_hg, p_stack_guard));
	stacki.push_back(-999); // not used for anything
	stack_lstm.add_input(stack.back());

}

void pstate_layer::Init_layer(parser_config& cfg)
{
	layer::Init_layer(cfg);
	Model* model = cfg.model;
	////////////////////////////////////////////////
	p_pbias = model->add_parameters(Dim(parser_config::HIDDEN_DIM, 1));
	p_A = model->add_parameters(Dim(parser_config::HIDDEN_DIM, parser_config::HIDDEN_DIM));
	p_B = model->add_parameters(Dim(parser_config::HIDDEN_DIM, parser_config::HIDDEN_DIM));
	p_S = model->add_parameters(Dim(parser_config::HIDDEN_DIM, parser_config::HIDDEN_DIM));
}

void pstate_layer::Init_Graph(ComputationGraph* hg)
{
	layer::Init_Graph(hg);

	//for parser state
	pbias = parameter(*hg, p_pbias);
	S = parameter(*hg, p_S);
	B = parameter(*hg, p_B);
	A = parameter(*hg, p_A);
}

Expression pstate_layer::build(parser_state& state)
{
	// p_t = pbias + S * slstm + B * blstm + A * almst
	Expression p_t = affine_transform({ pbias, S, state.stack_lstm.back(), B, state.buffer_lstm.back(), A, state.action_lstm.back() });
	return rectify(p_t);
}


void action_layer::Init_layer(parser_config& cfg)
{
	layer::Init_layer(cfg);
	Model* model = cfg.model;
	////////////////////////////////////////////////
	p_p2a = model->add_parameters(Dim(parser_config::ACTION_SIZE, parser_config::HIDDEN_DIM));
	p_abias = model->add_parameters(Dim(parser_config::ACTION_SIZE, 1));
	p_a = model->add_lookup_parameters(parser_config::ACTION_SIZE, Dim(parser_config::ACTION_DIM, 1));
}

void action_layer::Init_Graph(ComputationGraph* hg)
{
	layer::Init_Graph(hg);

	p2a = parameter(*hg, p_p2a);
	abias = parameter(*hg, p_abias);
}

Expression action_layer::build(Expression pt, vector<unsigned>& current_valid_actions)
{
	Expression r_t = affine_transform({ abias, p2a, pt });
	// adist = log_softmax(r_t, current_valid_actions)
	return log_softmax(r_t, current_valid_actions);
}

void composition_layer::Init_layer(parser_config& cfg)
{
	layer::Init_layer(cfg);
	Model* model = cfg.model;
	////////////////////////////////////////////////
	p_r = (model->add_lookup_parameters(parser_config::ACTION_SIZE, Dim(parser_config::REL_DIM, 1)));
	p_H = (model->add_parameters(Dim(parser_config::LSTM_INPUT_DIM, parser_config::LSTM_INPUT_DIM)));
	p_D = (model->add_parameters(Dim(parser_config::LSTM_INPUT_DIM, parser_config::LSTM_INPUT_DIM)));
	p_R = (model->add_parameters(Dim(parser_config::LSTM_INPUT_DIM, parser_config::REL_DIM)));
	p_cbias = (model->add_parameters(Dim(parser_config::LSTM_INPUT_DIM, 1)));

}
void composition_layer::Init_Graph(ComputationGraph* hg)
{
	layer::Init_Graph(hg);
	H = parameter(*hg, p_H);
	D = parameter(*hg, p_D);
	R = parameter(*hg, p_R);
	cbias = parameter(*hg, p_cbias);
}
// composed = cbias + H * head + D * dep + R * relation
Expression composition_layer::build(Expression head, Expression dep, Expression relation)
{
	Expression composed = affine_transform({ cbias, H, head, D, dep, R, relation });
	return tanh(composed);
}


void ShiftReduceParser::build_setOfActions(dictionary * prel2Int)
{
	//vector<unsigned>	possible_actions;
	//vector<string>		setOfActions;
	int idx = 0;
	setOfActions.push_back("SHIFT");
	possible_actions.push_back(idx++);
	setOfActions.push_back("REDUCE");
	possible_actions.push_back(idx++);

	char buf[512];
	for(int i=0;i<prel2Int->nTokens;i++) {
		sprintf(buf, "LEFT_%d", i);
		setOfActions.push_back(buf);
		possible_actions.push_back(idx++);

		sprintf(buf, "RIGHT_%d", i);
		setOfActions.push_back(buf);
		possible_actions.push_back(idx++);
	}
}

////////////////////////////////////////////////////////////////////////
ShiftReduceParser::ShiftReduceParser(parser_config& cfg) : nn_cfg(cfg), nn_pre(cfg.model)
{
	//nn_cfg.model = model;
	//nn_cfg.pretrained = &pretrained;
	nn_words.Init_layer(nn_cfg);
	nn_parser.Init_layer(nn_cfg);
	nn_pstate.Init_layer(nn_cfg);
	nn_actions.Init_layer(nn_cfg);
	nn_composition.Init_layer(nn_cfg);
	nn_pre.Init_layer(nn_cfg, &nn_words, nn_composition.p_r);
}
////////////////////////////////////////////////////////////////////////
vector<pair<int, int>> ShiftReduceParser::log_prob_parser(
	ComputationGraph* hg,
	const vector<int>& raw_sent,  // raw sentence
	const vector<int>& sent,  // sent with oovs replaced
	const vector<int>& sentPos,
	const vector<int>& goldhead,
	const vector<int>& goldrel,
	map<int, vector<int>>&  goldhead2deps,
	sentence_tr& sentinfo,
	bool btrain, double *right,double iter_num) 
{
	nn_words.Init_Graph(hg);
	nn_parser.Init_Graph(hg);
	nn_pstate.Init_Graph(hg);
	nn_actions.Init_Graph(hg);
	nn_composition.Init_Graph(hg);

	LSTMBuilder& stack_lstm = nn_parser.stack_lstm; // (layers, input, hidden, trainer)
	LSTMBuilder& buffer_lstm = nn_parser.buffer_lstm;
	LSTMBuilder& action_lstm = nn_parser.action_lstm;
	nn_parser.clear();
	nn_tree2embedding.clear();
	vector<Expression>& buffer = nn_parser.buffer;
	vector<int>& bufferi = nn_parser.bufferi;
	vector<Expression>& stack = nn_parser.stack;  // variables representing subtree embeddings
	vector<int>& stacki = nn_parser.stacki; //

	vector<Expression>& pre_embedding = nn_pre.build(sentinfo, hg);
	nn_parser.Init_state(raw_sent, sent, sentPos, nn_words, pre_embedding);

	int size_sent = sent.size();
	map<int, vector<int>>  head2deps;
	map<int, int> modify2head;
	vector<pair<int,int>> results(size_sent, pair<int, int>(-1,-1));
	//////////////////////////////////////////////////////////////////////
	const bool build_training_graph = btrain;
	unsigned action_count = 0;  // incremented at each prediction
	while (stack.size() > 1 || buffer.size() > 2) {
		// get list of possible actions for the current parser state
		map<int, bool> legal_transitions;
		Oracle_ArcEager::legal(bufferi, stacki, head2deps, modify2head, goldhead, legal_transitions);

		vector<unsigned> current_valid_actions;
		for (auto a : possible_actions) {
			const string& actionString = setOfActions[a];
			const char ac = actionString[0];
			const char ac2 = actionString[1];
			if (ac == 'S' && ac2 == 'H') {  // SHIFT
				if(legal_transitions[shift_action])
					current_valid_actions.push_back(a);
			}
			else if (ac == 'R' && ac2 == 'E') { // REDUCE
				if (legal_transitions[reduce_action])
					current_valid_actions.push_back(a);
			}
			else if (ac == 'L') { // LEFT 
				if (legal_transitions[left_action])
					current_valid_actions.push_back(a);
			}
			else if (ac == 'R') {// RIGHT
				if (legal_transitions[right_action])
					current_valid_actions.push_back(a);
			}
		}
		//------------------------------
		Expression r_t = nn_pstate.build(nn_parser);
		Expression adiste = nn_actions.build(r_t, current_valid_actions);
		vector<float> adist = as_vector(hg->incremental_forward());
		//------------------------------
		double best_score = adist[current_valid_actions[0]];
		unsigned best_a = current_valid_actions[0];
		for (unsigned i = 1; i < current_valid_actions.size(); ++i) {
			if (adist[current_valid_actions[i]] > best_score) {
				best_score = adist[current_valid_actions[i]];
				best_a = current_valid_actions[i];
			}
		}
		unsigned action = best_a;
		if (build_training_graph) {  // if we have reference actions (for training) use the reference action
			map<int, bool> options;
			int s = stacki.back();
			int b = bufferi.back();
			Oracle_ArcEager::dyn_oracle(bufferi, stacki, goldhead2deps, 
				head2deps, modify2head, goldhead, options);
			//---------------
			if (options.empty()) {
				printf("ERR");
			}
			vector<unsigned> current_oracle_actions;
			for (auto a : possible_actions) {
				const string& actionString = setOfActions[a];
				const char ac = actionString[0];
				const char ac2 = actionString[1];
				if (ac == 'S' && ac2 == 'H') {  // SHIFT
					if (options[shift_action])
						current_oracle_actions.push_back(a);
				}
				else if (ac == 'R' && ac2 == 'E') { // REDUCE
					if (options[reduce_action])
						current_oracle_actions.push_back(a);
				}
				else if (ac == 'L') { // LEFT 
					if (!options[left_action])
						continue;
					int rel = goldrel[s];
					int ifd = actionString.rfind('_');
					string idx = actionString.substr(ifd+1);
					if (atoi(idx.c_str()) != rel)
						continue;
					current_oracle_actions.push_back(a);
				}
				else if (ac == 'R') {// RIGHT
					if (!options[right_action])
						continue;
					int rel = goldrel[b];
					int ifd = actionString.rfind('_');
					string idx = actionString.substr(ifd+1);
					if (atoi(idx.c_str()) != rel)
						continue;
					current_oracle_actions.push_back(a);
				}
			}
			//---------------
			assert(current_oracle_actions.size() > 0);
			int find_action = -1;
			float best_src = -1000;
			bool bfind_best_a = false;
			for (int i = 0;i < current_oracle_actions.size();i++) {
				if (best_a == current_oracle_actions[i])
					bfind_best_a = true;
				if (adist[current_oracle_actions[i]] > best_src) {
					find_action = current_oracle_actions[i];
					best_src = adist[current_oracle_actions[i]];
				}
			}
			/*if (bfind_best_a)
				action = best_a;
			else*/
				action = find_action;

			if (best_a == action) 
				(*right)++; 
			else/* if(!bfind_best_a)*/ {
				if (iter_num >= 0 && rand() / ((float)RAND_MAX) < 0.5f)
					action = best_a;
			}
		}
		++action_count;
		
		//------------------------------
		nn_parser.accum(adiste, action);

		Expression actione = nn_actions.lookup_act(action);
		action_lstm.add_input(actione);
		//------------------------------
		Expression relation = nn_composition.lookup_rel(action);
		// do action
		const string& actionString = setOfActions[action];
		const char ac = actionString[0];
		const char ac2 = actionString[1];
		int s = stacki.back();
		int b = bufferi.back();
		if (ac == 'S' && ac2 == 'H') {  // SHIFT
			assert(buffer.size() > 1); // dummy symbol means > 1 (not >= 1)

			stack.push_back(buffer.back());
			stack_lstm.add_input(buffer.back());
			buffer.pop_back();
			buffer_lstm.rewind_one_step();
			stacki.push_back(bufferi.back());
			bufferi.pop_back();
		}
		else if (ac == 'R' && ac2 == 'E'){ //reduce --- Miguel
			stack.pop_back();
			stacki.pop_back();
			stack_lstm.rewind_one_step();
		}
		else if (ac == 'R') { // LEFT or RIGHT
			assert(stack.size() > 1 && buffer.size() > 1);
			unsigned depi = bufferi.back(), headi = stacki.back();
			Expression dep=buffer.back(), head= stack.back();
			buffer.pop_back();
			bufferi.pop_back();
			buffer_lstm.rewind_one_step();

			Expression nlcomposed = nn_composition.build(head, dep, relation)- pre_embedding[headi];

			//reflesh head embedding to nlcomposed
			stack.pop_back(); 
			stacki.pop_back();
			stack_lstm.rewind_one_step();
			stack.push_back(nlcomposed);
			stacki.push_back(headi);
			stack_lstm.add_input(nlcomposed);

			//push dep to stack
			stack.push_back(dep);
			stacki.push_back(depi);
			stack_lstm.add_input(dep);

			head2deps[headi].push_back(depi);
			modify2head[depi] = headi;

			int ifd = actionString.rfind('_');
			int rel_idx = atoi(actionString.substr(ifd+1).c_str());
			results[depi].first = headi;
			results[depi].second = rel_idx;
		}
		else if (ac == 'L') {
			assert(stack.size() > 1 && buffer.size() > 1);
			unsigned depi = stacki.back(), headi = bufferi.back();
			Expression dep = stack.back(), head = buffer.back();
			stack.pop_back();
			stacki.pop_back();
			stack_lstm.rewind_one_step();
			buffer.pop_back();
			bufferi.pop_back();
			buffer_lstm.rewind_one_step();

			Expression nlcomposed = nn_composition.build(head, dep, relation) - pre_embedding[headi];

			//reflesh head embedding to nlcomposed
			buffer.push_back(nlcomposed);
			bufferi.push_back(headi);
			buffer_lstm.add_input(nlcomposed);

			head2deps[headi].push_back(depi);
			modify2head[depi] = headi;

			int ifd = actionString.rfind('_');
			int rel_idx = atoi(actionString.substr(ifd + 1).c_str());
			results[depi].first = headi;
			results[depi].second = rel_idx;
		}
	}
	assert(stack.size() == 1); // guard symbol, root
	assert(stacki.size() == 1);
	assert(buffer.size() == 2); // guard symbol
	assert(bufferi.size() == 2);
	Expression tot_neglogprob = nn_parser.log_p_total();
	assert(tot_neglogprob.pg != nullptr);
	return results;
}
