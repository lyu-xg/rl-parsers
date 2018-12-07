from collections import namedtuple

from ply import lex, yacc
from rl_parsers import ParserError
from . import tokrules

import numpy as np


# LEXER

lexer = lex.lex(module=tokrules)


# DPOMDP

DPOMDP = namedtuple(
    'DPOMDP',
    'agents, discount, values, states, actions, observations, start, T, O, R,'
    ' reset'
)


# PARSER


class Parser:
    tokens = tokrules.tokens

    def __init__(self):
        self.discount = None
        self.values = None

        self.agents = None
        self.states = None
        self.actions = None
        self.observations = None

        self.start = None
        self.T = None
        self.O = None
        self.R = None

        self.reset = None

    def p_error(self, p):
        # TODO send all printsto stderr or smth like that
        # print('Parsing Error:', p.lineno, p.lexpos, p.type, p.value)
        raise ParserError(
            f'Parsing Error: {p.lineno} {p.lexpos} {p.type} {p.value}')

    def p_dpomdp(self, p):
        """ dpomdp : preamble structure """
        self.dpomdp = DPOMDP(
            agents=self.agents,
            discount=self.discount,
            values=self.values,
            states=self.states,
            actions=self.actions,
            observations=self.observations,
            start=self.start,
            T=self.T,
            O=self.O,
            R=self.R,
            reset=self.reset,
        )

    ###

    # TODO should probably enforce uniqueness requirement!
    def p_preamble(self, p):
        """ preamble : preamble_list """
        self.T = np.zeros((*self.nactions, self.nstates, self.nstates))
        self.O = np.zeros((*self.nactions, self.nstates, *self.nobservations))
        self.R = np.zeros(
            (*self.nactions, self.nstates, self.nstates, *self.nobservations))
        self.reset = np.zeros((*self.nactions, self.nstates), dtype=np.bool)

    def p_preamble_list(self, p):
        """ preamble_list : preamble_list preamble_item
                          | preamble_item """

    def p_preamble_agents_N(self, p):
        """ preamble_item : AGENTS COLON INT NL """
        N = p[3]
        self.agents = tuple(range(N))
        self.nagents = N

    def p_preamble_agents_names(self, p):
        """ preamble_item : AGENTS COLON id_list NL """
        id_list = p[3]
        self.agents = tuple(id_list)
        self.nagents = len(id_list)

    def p_preamble_discount(self, p):
        """ preamble_item : DISCOUNT COLON unumber NL """
        self.discount = p[3]

    def p_preamble_values(self, p):
        """ preamble_item : VALUES COLON REWARD NL
                          | VALUES COLON COST NL """
        self.values = p[3]

    def p_preamble_states_N(self, p):
        """ preamble_item : STATES COLON INT NL """
        N = p[3]
        self.states = tuple(range(N))
        self.nstates = N

    def p_preamble_states_names(self, p):
        """ preamble_item : STATES COLON id_list NL """
        id_list = p[3]
        self.states = tuple(id_list)
        self.nstates = len(id_list)

    ###

    def p_preamble_actions(self, p):
        """ preamble_item : ACTIONS COLON NL def_list """
        def_list = p[4]
        self.actions = tuple(def_list)
        self.nactions = tuple(map(len, def_list))
        self.astrides = np.cumprod(self.nactions[::-1])[::-1]
        # self.astrides = (np.array(self.nactions)[::-1].cumprod()[::-1] //
        #                  self.nactions)

    def p_preamble_observations(self, p):
        """ preamble_item : OBSERVATIONS COLON NL def_list """
        def_list = p[4]
        self.observations = tuple(def_list)
        self.nobservations = tuple(map(len, def_list))

    def p_preamble_def_list(self, p):
        """ def_list : def_list def """
        p[0] = p[1] + [p[2]]

    def p_preamble_def_base(self, p):
        """ def_list : def """
        p[0] = [p[1]]

    def p_preamble_def_N(self, p):
        """ def : INT NL """
        p[0] = tuple(range(p[1]))

    def p_preamble_def_names(self, p):
        """ def : id_list NL """
        p[0] = tuple(p[1])

    ###

    def p_preamble_start_uniform(self, p):
        """ preamble_item : START COLON NL UNIFORM NL """
        self.start = np.full(self.nstates, 1 / self.nstates)

    def p_preamble_start_dist(self, p):
        """ preamble_item : START COLON NL pvector NL """
        pv = np.array(p[4])
        if not np.isclose(pv.sum(), 1.):
            raise ParserError(f'Start distribution is not normalized (sums to '
                              '{pv.sum()}).')
        self.start = pv

    def p_preamble_start_state(self, p):
        """ preamble_item : START COLON state NL """
        s = p[3]
        self.start = np.zeros(self.nstates)
        self.start[s] = 1

    def p_preamble_start_include(self, p):
        """ preamble_item : START INCLUDE COLON state_list NL """
        slist = p[4]
        self.start = np.zeros(self.nstates)
        self.start[slist] = 1 / len(slist)

    def p_preamble_start_exclude(self, p):
        """ preamble_item : START EXCLUDE COLON state_list NL """
        slist = p[4]
        self.start = np.full(self.nstates, 1 / (self.nstates - len(slist)))
        self.start[slist] = 0

    ###

    def p_id_list(self, p):
        """ id_list : id_list ID """
        p[0] = p[1] + [p[2]]

    def p_id_list_base(self, p):
        """ id_list : ID """
        p[0] = [p[1]]

    ###

    def p_state_list(self, p):
        """ state_list : state_list state """
        p[0] = p[1] + [p[2]]

    def p_state_list_base(self, p):
        """ state_list : state """
        p[0] = [p[1]]

    ###

    def p_jaction(self, p):
        """ jaction : jactions_list """
        actions = p[1]

        if len(actions) == 1:
            action = actions[0]
            if isinstance(action, int):
                jaction = list(action // self.astrides)
            elif action == slice(None):
                jaction = [slice(None)] * self.nagents
        elif len(actions) == self.nagents:
            jaction = list(actions)
            for i, a in enumerate(actions):
                if isinstance(a, str):
                    jaction[i] = self.actions[i].index(a)
        else:
            # TODO
            raise ParserError('Joint action should contain either one or '
                              'enough indices for each agent')

        p[0] = tuple(jaction)

    def p_jactions_list(self, p):
        """ jactions_list : jactions_list action """
        p[0] = p[1] + [p[2]]

    def p_jactions_base(self, p):
        """ jactions_list : action """
        p[0] = [p[1]]

    ###

    def p_action_idx(self, p):
        """ action : INT """
        p[0] = p[1]

    def p_action_id(self, p):
        """ action : ID """
        p[0] = p[1]

    def p_action_all(self, p):
        """ action : ASTERISK """
        p[0] = slice(None)

    ###

    def p_jobservation(self, p):
        """ jobservation : observations """
        observations = p[1]

        if len(observations) == 1 and observations[0] == slice(None):
            jobservation = [slice(None)] * self.nagents
        elif len(observations) == self.nagents:
            jobservation = list(observations)
            for i, o in enumerate(observations):
                if isinstance(o, str):
                    jobservation[i] = self.observations[i].index(o)
        else:
            # TODO
            raise ParserError('BLARG')

        p[0] = tuple(jobservation)

    def p_observations(self, p):
        """ observations : observations observation """
        p[0] = p[1] + [p[2]]

    def p_observations_base(self, p):
        """ observations : observation """
        p[0] = [p[1]]

    def p_observation_idx(self, p):
        """ observation : INT """
        p[0] = p[1]

    def p_observation_id(self, p):
        """ observation : ID """
        p[0] = p[1]

    def p_observation_all(self, p):
        """ observation : ASTERISK """
        p[0] = slice(None)

    ###

    def p_state_idx(self, p):
        """ state : INT """
        p[0] = p[1]

    def p_state_id(self, p):
        """ state : ID """
        p[0] = self.states.index(p[1])

    def p_state_all(self, p):
        """ state : ASTERISK """
        p[0] = slice(None)

    ###

    def p_structure(self, p):
        """ structure : structure_list """

    def p_structure_list(self, p):
        """ structure_list : structure_list structure_item
                           | """

    def p_structure_t_ass(self, p):
        """ structure_item : T COLON jaction COLON state COLON state COLON prob NL """
        ja, s0, s1, prob = p[3], p[5], p[7], p[9]
        self.T[(*ja, s0, s1)] = prob

    def p_structure_t_as_uniform(self, p):
        """ structure_item : T COLON jaction COLON state COLON NL UNIFORM NL """
        ja, s0 = p[3], p[5]
        self.T[(*ja, s0)] = 1 / self.nstates

    def p_structure_t_as_reset(self, p):
        """ structure_item : T COLON jaction COLON state COLON NL RESET NL """
        ja, s0 = p[3], p[5]
        self.T[(*ja, s0)] = self.start
        self.reset[(*ja, s0)] = True

    def p_structure_t_as_dist(self, p):
        """ structure_item : T COLON jaction COLON state COLON NL pvector NL """
        ja, s0, pv = p[3], p[5], p[8]
        pv = np.array(pv)
        # TODO postpone tests to end...?
        if not np.isclose(pv.sum(), 1.):
            raise ParserError(f'Transition distribution (jaction={ja}, '
                              f'state={s0}) is not normalized (sums to '
                              f'{pv.sum()}).')
        self.T[(*ja, s0)] = pv

    def p_structure_t_a_uniform(self, p):
        """ structure_item : T COLON jaction COLON NL UNIFORM NL """
        ja = p[3]
        self.T[ja] = 1 / self.nstates

    def p_structure_t_a_identity(self, p):
        """ structure_item : T COLON jaction COLON NL IDENTITY NL """
        ja = p[3]
        self.T[ja] = np.eye(self.nstates)

    def p_structure_t_a_dist(self, p):
        """ structure_item : T COLON jaction COLON NL pmatrix NL """
        ja, pm = p[3], p[6]
        pm = np.reshape(pm, (self.nstates, self.nstates))
        # TODO postpone tests to end...?
        if not np.isclose(pm.sum(axis=1), 1.).all():
            raise ParserError(f'Transition state distribution (action={ja}) is'
                              ' not normalized;')
        self.T[ja] = pm

    ###

    def p_structure_o_aso(self, p):
        """ structure_item : O COLON jaction COLON state COLON jobservation COLON prob NL """
        ja, s1, jo, pr = p[3], p[5], p[7], p[9]
        self.O[(*ja, s1, *jo)] = pr

    def p_structure_o_as_uniform(self, p):
        """ structure_item : O COLON jaction COLON state COLON NL UNIFORM NL """
        ja, s1 = p[3], p[5]
        self.O[(*ja, s1)] = 1 / np.prod(self.nobservations)

    def p_structure_o_as_dist(self, p):
        """ structure_item : O COLON jaction COLON state COLON NL pvector NL """
        ja, s1, pv = p[3], p[5], p[8]
        # TODO test probability?
        self.O[(*ja, s1)] = np.reshape(pv, self.nobservations)

    def p_structure_o_a_uniform(self, p):
        """ structure_item : O COLON jaction COLON NL UNIFORM NL """
        ja = p[3]
        self.O[ja] = 1 / np.prod(self.nobservations)

    def p_structure_o_a_dist(self, p):
        """ structure_item : O COLON jaction COLON NL pmatrix NL """
        ja, pm = p[3], p[6]
        # TODO test probability?
        self.O[ja] = np.reshape(pm, (self.nstates, *self.nobservations))

    ###

    def p_structure_r_asso(self, p):
        """ structure_item : R COLON jaction COLON state COLON state COLON jobservation COLON number NL """
        ja, s0, s1, jo, r = p[3], p[5], p[7], p[9], p[11]
        self.R[(*ja, s0, s1, *jo)] = r

    def p_structure_r_ass(self, p):
        """ structure_item : R COLON jaction COLON state COLON state COLON NL nvector NL """
        ja, s0, s1, rv = p[3], p[5], p[7], p[10]
        self.R[(*ja, s0, s1)] = np.reshape(rv, self.nobservations)

    def p_structure_r_as(self, p):
        """ structure_item : R COLON jaction COLON state COLON NL nmatrix NL """
        ja, s0, rm = p[3], p[5], p[8]
        self.R[(*ja, s0)] = np.reshape(rm, (self.nstates, *self.nobservations))

    ###

    def p_nmatrix(self, p):
        """ nmatrix : nmatrix NL nvector """
        p[0] = p[1] + [p[3]]

    def p_nmatrix_base(self, p):
        """ nmatrix : nvector """
        p[0] = [p[1]]

    def p_nvector(self, p):
        """ nvector : nvector number """
        p[0] = p[1] + [p[2]]

    def p_nvector_base(self, p):
        """ nvector : number """
        p[0] = [p[1]]

    ###

    def p_pmatrix(self, p):
        """ pmatrix : pmatrix NL pvector """
        p[0] = p[1] + [p[3]]

    def p_pmatrix_base(self, p):
        """ pmatrix : pvector """
        p[0] = [p[1]]

    def p_pvector(self, p):
        """ pvector : pvector prob """
        p[0] = p[1] + [p[2]]

    def p_pvector_base(self, p):
        """ pvector : prob """
        p[0] = [p[1]]

    def p_prob(self, p):
        """ prob : unumber """
        prob = p[1]
        if not 0 <= prob <= 1:
            raise ParserError(f'Probability value ({prob}) out of bounds.')
        p[0] = prob

    ###

    def p_number_signed(self, p):
        """ number : PLUS unumber
                   | MINUS unumber """
        if p[1] == '+':
            p[0] = p[2]
        elif p[1] == '-':
            p[0] = -p[2]
        else:
            assert False

    def p_number_unsigned(self, p):
        """ number : unumber """
        p[0] = p[1]

    def p_unumber(self, p):
        """ unumber : FLOAT
                    | INT """
        p[0] = float(p[1])


def parse(text, **kwargs):
    p = Parser()
    y = yacc.yacc(module=p)
    y.parse(text, lexer=lexer, **kwargs)
    return p.dpomdp
