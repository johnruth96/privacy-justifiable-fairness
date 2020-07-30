from typing import List

from privacy.base import BaseAnonymizer


class BayardoAnonymizer(BaseAnonymizer):
    CALL_CACHE = {}

    def _reset_state(self, k):
        super(BayardoAnonymizer, self)._reset_state(k)
        self.CALL_CACHE = {}

    def anonymize(self):
        self.k_anonymize([], self.sigma_all, self.best_cost)

    def k_anonymize(self, head_set, tail_set, best_cost):
        h = head_set
        t = tail_set

        t = self.prune_useless_values(h, t)
        c = min(best_cost, self.compute_cost(h))
        if c < self.best_cost:
            self.best_head = h
            self.best_cost = c

        t = self.prune(h, t, c)
        t = self.reorder_tail(h, t)
        while t:
            v = t[0]
            h_new = h.copy()
            h_new.append(v)
            h_new.sort()
            t.remove(v)
            c = self.k_anonymize(h_new, t, c)
            if c < self.best_cost:
                self.best_head = h_new
                self.best_cost = c
            t = self.prune(h, t, c)
        return c

    def prune_useless_values(self, head_set, tail_set):
        return tail_set

    def prune(self, head_set, tail_set, best_cost: float) -> List[int]:
        call = (tuple(head_set), tuple(tail_set), best_cost)
        if call not in self.CALL_CACHE:
            self.CALL_CACHE[call] = self._prune(head_set, tail_set, best_cost)
        return self.CALL_CACHE[call]

    def _prune(self, head_set, tail_set, best_cost: float) -> List[int]:
        all_set = sorted(head_set + tail_set)
        lower_bound = self.compute_lower_bound(head_set, all_set)
        if lower_bound >= best_cost:
            return []

        t_new = tail_set.copy()
        for v in tail_set:
            h_new = sorted(head_set + [v])
            param_t_new = t_new.copy()
            param_t_new.remove(v)
            if self.prune(h_new, param_t_new, best_cost) == []:
                cost_h_new = self.compute_cost(h_new)
                if cost_h_new > best_cost:
                    t_new.remove(v)
        if t_new != tail_set:
            # prune
            return self.prune(head_set, t_new, best_cost)
        else:
            return t_new

    def reorder_tail(self, head_set, tail_set):
        return tail_set
