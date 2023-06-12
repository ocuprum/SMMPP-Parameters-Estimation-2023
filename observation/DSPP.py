from observation.PoissonProcess import PoissonProcess

class DSPP:
    def __init__(self, lams):
        self.lams = lams


    def next(self, smc_state, prev_S, cur_S):
        lam = self.lams[smc_state]
        poisson_process = PoissonProcess(lam=lam * (cur_S - prev_S))
        self.events_amount = poisson_process.next()

        return self.events_amount