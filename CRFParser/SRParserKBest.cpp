#include "SRParserKBest.h"
#include "freelist.h"

//----------------------------------------------
void parser_stateKBest::Init_layer(parser_config& cfg)
{
	layer::Init_layer(cfg);
	Model* model = cfg.model;
	const EmbedDict& pretrained = *cfg.pretrained;
	//////////////////////////////////////////////////////////////////////////
	stack_lstm = KBestRLSTM(parser_config::LAYERS, parser_config::LSTM_INPUT_DIM, parser_config::HIDDEN_DIM, model, parser_config::KBEST);
	buffer_lstm = KBestRLSTM(parser_config::LAYERS, parser_config::LSTM_INPUT_DIM, parser_config::HIDDEN_DIM, model, parser_config::KBEST);
	action_lstm = KBestRLSTM(parser_config::LAYERS, parser_config::ACTION_DIM, parser_config::HIDDEN_DIM, model, parser_config::KBEST);

	p_action_start = model->add_parameters(Dim(parser_config::ACTION_DIM, 1));
	p_buffer_guard =model->add_parameters(Dim(parser_config::LSTM_INPUT_DIM, 1));
	p_stack_guard = model->add_parameters(Dim(parser_config::LSTM_INPUT_DIM, 1));
}

void parser_stateKBest::Init_Graph(ComputationGraph* hg)
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

void parser_stateKBest::Init_state(const t_unvec& raw_sent, const t_unvec& sent, const t_unvec& sentPos, word_layer& layer)
{
	int sen_sz = (int)sent.size();
	action_lstm.add_input_all(action_start);
	kbest_stores.resize(parser_config::KBEST);
	for (int kbi = 0; kbi < parser_config::KBEST; kbi++) {
		kbest_stores[kbi].buffer.resize(sen_sz + 1);  // variables representing word embeddings (possibly including POS info)
		kbest_stores[kbi].bufferi.resize(sen_sz + 1);
		if (kbi == 0) {
			//------------------------------------------------------------------------
			for (unsigned i = 0; i < sent.size(); ++i) {
				assert(sent[i] < parser_config::VOCAB_SIZE);
				kbest_stores[kbi].buffer[sent.size() - i] = layer.build(sent[i], sentPos[i], raw_sent[i]);
				kbest_stores[kbi].bufferi[sent.size() - i] = i;
			}
			kbest_stores[kbi].buffer[0] = parameter(*m_hg, p_buffer_guard);
			kbest_stores[kbi].bufferi[0] = -999;
			for (auto& b : kbest_stores[kbi].buffer)
				buffer_lstm.add_input_all(b);
			//------------------------------------------------------------------------
			kbest_stores[kbi].stack.push_back(parameter(*m_hg, p_stack_guard));
			stack_lstm.add_input_all(kbest_stores[kbi].stack.back());
		}
		else {
			//------------------------------------------------------------------------
			for (unsigned i = 0; i < sent.size(); ++i) {
				kbest_stores[kbi].buffer[sent.size() - i] = kbest_stores[0].buffer[sent.size() - i];
				kbest_stores[kbi].bufferi[sent.size() - i] = i;
			}
			kbest_stores[kbi].buffer[0] = kbest_stores[0].buffer[0];
			kbest_stores[kbi].bufferi[0] = -999;
			//------------------------------------------------------------------------
			kbest_stores[kbi].stack.push_back(kbest_stores[0].stack[0]);
		}
		kbest_stores[kbi].stacki.push_back(-999); // not used for anything
	}
}
////////////////////////////////////////////////////////////////////////
void pstate_layerKBest::Init_layer(parser_config& cfg)
{
	layer::Init_layer(cfg);
	Model* model = cfg.model;
	////////////////////////////////////////////////
	p_pbias = model->add_parameters(Dim(parser_config::HIDDEN_DIM, 1));
	p_A = model->add_parameters(Dim(parser_config::HIDDEN_DIM, parser_config::HIDDEN_DIM));
	p_B = model->add_parameters(Dim(parser_config::HIDDEN_DIM, parser_config::HIDDEN_DIM));
	p_S = model->add_parameters(Dim(parser_config::HIDDEN_DIM, parser_config::HIDDEN_DIM));
}

void pstate_layerKBest::Init_Graph(ComputationGraph* hg)
{
	layer::Init_Graph(hg);

	//for parser state
	pbias = parameter(*hg, p_pbias);
	S = parameter(*hg, p_S);
	B = parameter(*hg, p_B);
	A = parameter(*hg, p_A);
}

Expression pstate_layerKBest::build(parser_stateKBest& state, int kbi)
{
	// p_t = pbias + S * slstm + B * blstm + A * almst
	Expression p_t = affine_transform({ pbias, S, state.stack_lstm.back(kbi), B, state.buffer_lstm.back(kbi), A, state.action_lstm.back(kbi) });
	return rectify(p_t);
}


////////////////////////////////////////////////////////////////////////
ShiftReduceParserKBest::ShiftReduceParserKBest(parser_config& cfg) : nn_cfg(cfg)
{
	//nn_cfg.model = model;
	//nn_cfg.pretrained = &pretrained;
	nn_words.Init_layer(nn_cfg);
	nn_parser.Init_layer(nn_cfg);
	nn_pstate.Init_layer(nn_cfg);
	nn_actions.Init_layer(nn_cfg);
	nn_composition.Init_layer(nn_cfg);
}
////////////////////////////////////////////////////////////////////////

void ShiftReduceParserKBest::take_action(score_pair& act, const vector<string>& setOfActions, unsigned kbi)
{
	KBestRLSTM& action_lstm = nn_parser.action_lstm;
	//------------------------------
	nn_parser.accum(act.adiste, act.actidx, kbi);
	Expression actione = nn_actions.lookup_act(act.actidx);
	action_lstm.add_input(actione, kbi);
	//-----------------------------
	vector<Expression>& buffer = nn_parser.kbest_stores[kbi].buffer;
	vector<int>& bufferi = nn_parser.kbest_stores[kbi].bufferi;
	vector<Expression>& stack = nn_parser.kbest_stores[kbi].stack;
	vector<int>& stacki = nn_parser.kbest_stores[kbi].stacki;

	Expression relation = nn_composition.lookup_rel(act.actidx);
	// do action
	const string& actionString = setOfActions[act.actidx];
	const char ac = actionString[0];
	const char ac2 = actionString[1];

	if (ac == 'S' && ac2 == 'H') {  // SHIFT
		assert(buffer.size() > 1); // dummy symbol means > 1 (not >= 1)

		stack.push_back(buffer.back());
		nn_parser.stack_lstm.add_input(buffer.back(),kbi);
		buffer.pop_back();
		nn_parser.buffer_lstm.rewind_one_step(kbi);
		stacki.push_back(bufferi.back());
		bufferi.pop_back();
	}
	else if (ac == 'S' && ac2 == 'W'){ //SWAP --- Miguel
		assert(stack.size() > 2); // dummy symbol means > 2 (not >= 2)

		Expression toki, tokj;
		unsigned ii = 0, jj = 0;
		tokj = stack.back();
		jj = stacki.back();
		stack.pop_back();
		stacki.pop_back();

		toki = stack.back();
		ii = stacki.back();
		stack.pop_back();
		stacki.pop_back();

		buffer.push_back(toki);
		bufferi.push_back(ii);

		nn_parser.stack_lstm.rewind_one_step(kbi);
		nn_parser.stack_lstm.rewind_one_step(kbi);

		nn_parser.buffer_lstm.add_input(buffer.back(),kbi);

		stack.push_back(tokj);
		stacki.push_back(jj);

		nn_parser.stack_lstm.add_input(stack.back(),kbi);
	}
	else { // LEFT or RIGHT
		assert(stack.size() > 2); // dummy symbol means > 2 (not >= 2)
		assert(ac == 'L' || ac == 'R');
		Expression dep, head;
		unsigned depi = 0, headi = 0;
		(ac == 'R' ? dep : head) = stack.back();
		(ac == 'R' ? depi : headi) = stacki.back();
		stack.pop_back();
		stacki.pop_back();
		(ac == 'R' ? head : dep) = stack.back();
		(ac == 'R' ? headi : depi) = stacki.back();
		stack.pop_back();
		stacki.pop_back();
		//if (headi == sent.size() - 1) rootword = intToWords.find(sent[depi])->second;
		Expression nlcomposed = nn_composition.build(head, dep, relation);
		nn_tree2embedding[headi] = nlcomposed;
		nn_parser.stack_lstm.rewind_one_step(kbi);
		nn_parser.stack_lstm.rewind_one_step(kbi);
		nn_parser.stack_lstm.add_input(nlcomposed,kbi);
		stack.push_back(nlcomposed);
		stacki.push_back(headi);
	}
}

static FreeList<score_pair> pool_scorepair(1023 * 1023);

bool cmp_score_pair(score_pair* a, score_pair* b) {
	return a->score > b->score;
}

struct item_info {
	float score;
	unsigned current_actions;
	bool bstop;
	item_info() : bstop(false), score(0) {}
};

vector<unsigned> ShiftReduceParserKBest::log_prob_parser(ComputationGraph* hg,
	const vector<unsigned>& raw_sent,  // raw sentence
	const vector<unsigned>& sent,  // sent with oovs replaced
	const vector<unsigned>& sentPos,
	const vector<unsigned>& correct_actions,
	const vector<string>& setOfActions,
	const map<unsigned, std::string>& intToWords,
	double *right, int KBEST )
{
	nn_words.Init_Graph(hg);
	nn_parser.Init_Graph(hg);
	nn_pstate.Init_Graph(hg);
	nn_actions.Init_Graph(hg);
	nn_composition.Init_Graph(hg);

	KBestRLSTM& stack_lstm = nn_parser.stack_lstm; // (layers, input, hidden, trainer)
	KBestRLSTM& buffer_lstm = nn_parser.buffer_lstm;
	KBestRLSTM& action_lstm = nn_parser.action_lstm;
	nn_parser.clear();
	nn_parser.Init_state(raw_sent, sent, sentPos, nn_words);
	const bool build_training_graph = correct_actions.size() > 0;
	if (build_training_graph) KBEST = 1;
	//--------------------------------------------------------------
	unsigned action_count = 0;  // incremented at each prediction
	score_pair* best_idx = 0;
	vector<score_pair*> all_scores;
	vector<std::pair<int, float> > all_scores_res;
	score_pair best_res;
	std::vector<parser_store> &kbest_stores = nn_parser.kbest_stores;
	map<std::pair<vector<unsigned>, int>, score_pair* > scorepair_map;
	vector<item_info> iteminfo; iteminfo.resize(KBEST);
	pool_scorepair.free();


	while (true) {
		pool_scorepair.free();
		all_scores.clear();
		scorepair_map.clear();
		for (int kbi = 0; kbi < KBEST; kbi++) {
			vector<Expression>& buffer = kbest_stores[kbi].buffer;
			vector<int>& stacki        = kbest_stores[kbi].stacki;
			//-----------------------------
			if (!(stacki.size() > 2 || buffer.size() > 1)) continue;
			item_info& info = iteminfo[kbi];
			vector<unsigned> current_valid_actions;
			for (auto a : possible_actions) {
				if (IsActionForbidden(setOfActions[a], buffer.size(), stacki.size(), stacki))
					continue;
				current_valid_actions.push_back(a);
			}
			//------------------------------
			Expression r_t = nn_pstate.build(nn_parser, kbi);
			Expression adiste = nn_actions.build(r_t, current_valid_actions);
			vector<float> adist = as_vector(hg->incremental_forward());

			std::pair<vector<unsigned>, int> key = std::pair<vector<unsigned>, int>(kbest_stores[kbi].results, -1);
			for (int i = 0; i < current_valid_actions.size(); i++) {
				key.second = current_valid_actions[i];
				if (scorepair_map.find(key) != scorepair_map.end())
					continue;

				score_pair* sp = pool_scorepair.alloc(1); sp->bdone = false;
				sp->adiste = adiste; sp->kbest_idx = kbi; sp->actidx = current_valid_actions[i]; sp->score = info.score + adist[current_valid_actions[i]];
				all_scores.push_back(sp);

				scorepair_map[key] = sp;
			}
		}
		//------------------------------
		if (all_scores.empty())
			break;
		//------------------------------
		//stop node
		for (int kbi = 0; kbi < KBEST; kbi++) {
			item_info& info = iteminfo[kbi];
			if (!info.bstop) continue;
			std::pair<vector<unsigned>, int> key = std::pair<vector<unsigned>, int>(kbest_stores[kbi].results, -1);
			if (scorepair_map.find(key) != scorepair_map.end())
				continue;

			score_pair* sp = pool_scorepair.alloc(1);
			sp->actidx = info.current_actions; sp->kbest_idx = kbi;  sp->score = info.score; sp->bdone = true;
			all_scores.push_back(sp);

			scorepair_map[key] = sp;
		}
		std::sort(all_scores.begin(), all_scores.end(), cmp_score_pair);
		while (all_scores.size() < KBEST) {
			bool bfind = false;
			for (int fdx = 0; fdx < all_scores.size(); fdx++) 
				if (all_scores[fdx]->bdone == false) {
					score_pair* sp = pool_scorepair.alloc(1);
					*sp = *all_scores[fdx]; 
					all_scores.push_back(sp); bfind = true;
					break;
				}
			assert(bfind);
		}
		//------------------------------
		if (build_training_graph) {
			unsigned action = correct_actions[action_count];
			if (all_scores[0]->actidx == action) { (*right)++; }
			all_scores[0]->actidx = action;
		}
		best_idx = all_scores[0];
		++action_count;
		all_scores.resize(KBEST);
		//------------------------------
		//copy state
		vector<int> bdone, id2kbloc; bdone.resize(KBEST, 0); id2kbloc.resize(KBEST, -1);
		if (KBEST > 1) {
			for (int j = 0; j < KBEST; j++) {
				int cur_kbi = all_scores[j]->kbest_idx;
				if (!all_scores[j]->bdone ) continue;
				assert(!bdone[cur_kbi]);
				id2kbloc[j] = cur_kbi; bdone[cur_kbi] = 1;
			}
			for (int j = 0; j < KBEST; j++) {
				int cur_kbi = all_scores[j]->kbest_idx;
				if (bdone[cur_kbi]) continue;
				id2kbloc[j] = cur_kbi; bdone[cur_kbi] = 1;
			}
			int from_idx = 0;
			for (int to = 0; to < KBEST; to++) {
				if (bdone[to]) continue;
				if (id2kbloc[from_idx] != -1) {
					from_idx++; to--;
					continue;
				}
				if (from_idx > KBEST) {
					printf("ERROR:copy(from %d, to %d)\n", from_idx, to);
				}
				int from = all_scores[from_idx]->kbest_idx;
				kbest_stores[to].copy(kbest_stores[from]);
				stack_lstm.copy(from, to);
				buffer_lstm.copy(from, to);
				action_lstm.copy(from, to);
				all_scores[from_idx]->kbest_idx = to;
				from_idx++;
			}
		}
		//------------------------------
		for (int j = 0; j < KBEST; j++) {
			int cur_kbi = all_scores[j]->kbest_idx;
			if (all_scores[j]->bdone) continue;

			take_action(*all_scores[j], setOfActions, cur_kbi);

			if (all_scores[j]->kbest_idx == best_idx->kbest_idx && build_training_graph) {
				//correct_actions_check.push_back(all_scores[j]->actidx);
				if (action_count>0 && all_scores[j]->actidx != correct_actions[action_count - 1])
					assert(0);
			}
		}
		//------------------------------
		best_res = *best_idx;
		//------------------------------	
		int fininsh_cn = 0;
		for (int j = 0; j < KBEST; j++)  {
			int kbi = all_scores[j]->kbest_idx;
			item_info& info = iteminfo[kbi];
			info.score = all_scores[j]->score; info.current_actions = all_scores[j]->actidx;
			info.bstop = !(kbest_stores[kbi].stacki.size() > 2 || kbest_stores[kbi].bufferi.size() > 1);
			if (info.bstop) fininsh_cn++;
		}
		//if (fininsh_cn > KBEST*0.5) break;

		//------------------------------
		if (build_training_graph && action_count >= correct_actions.size()) break;
	}
	//------------------------------
	assert(kbest_stores[best_res.kbest_idx].stack.size() == 2); // guard symbol, root
	assert(kbest_stores[best_res.kbest_idx].stacki.size() == 2);
	assert(kbest_stores[best_res.kbest_idx].buffer.size() == 1); // guard symbol
	assert(kbest_stores[best_res.kbest_idx].bufferi.size() == 1);
	Expression tot_neglogprob = kbest_stores[best_res.kbest_idx].log_p_total();
	assert(tot_neglogprob.pg != nullptr);
	return kbest_stores[best_res.kbest_idx].results;
}
