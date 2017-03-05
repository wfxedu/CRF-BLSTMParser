#pragma once
#include <map>
#include <vector>
using namespace::std;
enum ACTION_TYPE {
	shift_action = 0,
	reduce_action = 1,
	right_action = 2,
	left_action = 3
};


class Oracle_ArcEager
{
public:
	static bool zero_cost_shift(
		const vector<int>& bufferi, 
		const vector<int>& stacki,
		const map<int, vector<int>>&  head2deps, 
		const map<int, int>& modify2head, 
		const vector<int>& goldhead)
	{
		/*
		Is a shift zero cost ?
		Moving b onto stack means that b will not be able to acquire any head or dependents in S.Cost
		is number of gold arcs of form(k, b) or (b, k) such that k in S

		: param conf : Working config
		: param gold_conf : Gold config
		: return : Is the cost zero
		*/
		if (bufferi.size() <= 2) //buffer guard -999
			return false;
		int b = bufferi.back();
		for (auto si : stacki) {
			if (si < 0) continue;//guard -999
			if (goldhead[si] == b || goldhead[b] == si)
				return false;
		}
		return true;
	}


	static void convert2map(const vector<int>& vec, map<int, int>& omap, int c = 1) {
		omap.clear();
		for (int i = 0;i < vec.size();i++)
			omap[vec[i]] = c;
	}

	static bool zero_cost_right(
		const vector<int>& bufferi, 
		const vector<int>& stacki,
		map<int, vector<int>>&  goldhead2deps, 
		const map<int, int>& modify2head,
		const vector<int>& goldhead)
	{
		/*
		Adding the arc (s, b) and pushing b onto the stack means that b will not be able to acquire any head in
		S or B, nor any dependents in S.  The cost is the number of gold arcs of form (k, b) such that k in S or B,
		(b, k) such that k in S and no arc (x, k) in working conf.  Cost zero for (s, b) in gold arcs but also
		where s is not the gold head of b but the real head not in S or B and no gold dependents of b in S.
		We return a boolean to identify if right-arc will be zero cost

		:param conf: working configuration (A_c)
		:param gold_conf: gold configuration
		:return: True if zero-cost, false otherwise
		*/
		if (bufferi.size() == 1 || stacki.size() == 1) //guard -999
			return false;
		auto s = stacki.back();
		auto b = bufferi.back();
		// (k, b)
		auto k = goldhead[b];
		if (k == s)
			return true;
		map<int, int> mbufferi, mstacki;
		convert2map(bufferi, mbufferi);
		convert2map(stacki, mstacki);
		//(k, b) and k in S or B
		bool k_b_costs = mstacki.find(k) != mstacki.end() ||
			mbufferi.find(k) != mbufferi.end();
		//(h, d) => k_heads[d] = h
		const map<int, int>& k_heads = modify2head;
		//(b, k)
		vector<int>& b_deps = goldhead2deps[b];
		//(b, k) and k in S
		vector<int> b_k_in_stack;
		for (auto tv : b_deps)
			if (mstacki.find(tv) != mstacki.end())
				b_k_in_stack.push_back(tv);

		// s is not gold head but real head(k) not in stack or buffer
		// and no gold deps of b in S-- (b, k) doesnt exist on stack
		if (mstacki.find(k) == mstacki.end() &&
			mbufferi.find(k) == mbufferi.end() && b_k_in_stack.size() == 0)
			return true;

		if (k_b_costs)
			return false;

		vector<int> b_k_final;
		for (auto tv : b_k_in_stack)
			if (k_heads.find(tv) == k_heads.end())
				b_k_final.push_back(tv);
		return b_k_final.size() == 0;
	}

	static bool zero_cost_left(
		const vector<int>& bufferi,
		const vector<int>& stacki,
		map<int, vector<int>>&  head2deps, 
		const map<int, int>& modify2head,
		const vector<int>& goldhead)
	{
		/*
		Is the cost of a left arc going to be zero?  Adding the arc (b, s) and popping s from the stack
		means that s will not be able to acquire any head or dependents in B.  The cost is the number of gold_arcs
		(k, s) or (s, k) where k in B.

		Cost of the arc found in the gold_arcs is 0, as well as the case where b is not the gold head, but the
		real head is not in B.

		:param conf: The working configuration
		:param gold_conf: The gold arcs
		:return: True if a left-arc would be zero-cost, False otherwise
		*/

		if (stacki.size() == 1 || bufferi.size() == 1) //guard -999
			return false;

		int s = stacki.back();
		int b = bufferi.back();

		for (int bi = b;bi < goldhead.size();bi++) {
			if (goldhead[bi] == s)
				return false;
			if (b != bi && goldhead[s] == bi)
				return false;
		}
		return true;
	}

	static bool zero_cost_reduce(
		const vector<int>& bufferi, 
		const vector<int>& stacki,
		map<int, vector<int>>&  head2deps, 
		const map<int, int>& modify2head,
		const vector<int>& goldhead)
	{
		if (stacki.size() == 1 || bufferi.size() == 1) //guard -999
			return false;

		int s = stacki.back();
		int b = bufferi.back();

		for (int bi = b;bi < goldhead.size();bi++)
			if (goldhead[bi] == s)
				return false;
		return true;
	}

	static void legal(
		const vector<int>& bufferi, 
		const vector<int>& stacki,
		map<int, vector<int>>&  head2deps, 
		const map<int, int>& modify2head,
		const vector<int>& goldhead, 
		map<int, bool>& legal_transitions)
	{

		bool shift_ok = true;
		bool right_ok = true;
		bool left_ok = true;
		bool reduce_ok = true;

		if (bufferi.size() == 2) //guard -999
			right_ok = shift_ok = false;

		int s = -1;
		if (stacki.size() == 1) //guard -999
			left_ok = right_ok = reduce_ok = false;
		else
			s = stacki.back();

		//if the s is already a dependent, we cannot left-arc
		if (modify2head.find(s) != modify2head.end())
			left_ok = false;
		else
			reduce_ok = false;
		legal_transitions[shift_action] = shift_ok;
		legal_transitions[right_action] = right_ok;
		legal_transitions[left_action] = left_ok;
		legal_transitions[reduce_action] = reduce_ok;
	}

	static void dyn_oracle(
		const vector<int>& bufferi,
		const vector<int>& stacki,
		map<int, vector<int>>&  goldhead2deps,
		map<int, vector<int>>&  head2deps,
		const map<int, int>& modify2head,
		const vector<int>& goldhead,
		map<int, bool>& options)
	{
		map<int, bool> legal_transitions;
		legal(bufferi, stacki, head2deps, modify2head, goldhead, legal_transitions);
		options.clear();
		bool allow_shift = zero_cost_shift(bufferi, stacki, head2deps, modify2head, goldhead);
		bool allow_right = zero_cost_right(bufferi, stacki, goldhead2deps, modify2head, goldhead);
		bool allow_left = zero_cost_left(bufferi, stacki, head2deps, modify2head, goldhead);
		bool allow_reduce = zero_cost_reduce(bufferi, stacki, head2deps, modify2head, goldhead);

		if (legal_transitions[shift_action] && allow_shift)
			options[shift_action]=true;
		if (legal_transitions[right_action] && allow_right)
			options[right_action] = true;
		if (legal_transitions[left_action] && allow_left)
			options[left_action] = true;
		if (legal_transitions[reduce_action] && allow_reduce)
			options[reduce_action] = true;
	}
};
