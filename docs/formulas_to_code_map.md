# Paper formulas → code map

| Symbol / Eq.   | What it is                                       | Code location |
|----------------|---------------------------------------------------|---------------|
| Eq. (1)        | Vertex partition (concept/skill/...)             | `hypergraph.VertexType` |
| Eq. (2)        | Hyperedge `(tail, head, kind)`                    | `hypergraph.Hyperedge` |
| Eq. (3)        | Prerequisite DAG `G_pr`                           | `methodology.MethodologyData.prereq_dag` |
| Eq. (4)        | Belief update `p_{t+1} = σ(logit(p_t)+q_t·G(H)·u)`| `student_model.StudentModel.bayes_update` |
| Eq. (5), (A1)  | Propagation operator `G(H)`                       | `hypergraph.build_propagation_operator` |
| Eq. (6)–(7)    | Mastery/meta dynamics, expected increment         | `environment.SyntheticStudent.update_p_true` |
| Eq. (23)       | Reliability heuristic `q_t`                       | `reliability.compute_reliability` |
| Eq. (27)       | `A_safe`                                          | `policies.ActionType` |
| Eq. (30)–(34)  | `m_mastery`, `m_transfer`, `m_retention`, `m_hint`, `m_robust` | `metrics.py` |
| Eq. (A3)       | Logistic-normal full stochastic specification     | `environment.SyntheticStudent.update_p_true` |
| Eq. (C1)       | Past-LTL invariants                               | `invariants.py` |
| Def. 1         | Pedagogical invariant                             | `invariants.py` docstrings |
| §V.B           | 16-state controller FSM                           | `fsm.ControllerFSM` |
| Lemma 1        | `‖G(H)‖_2 ≤ Σ_κ √(w_κ · max_e w_{κ,e})`           | `tests/test_hypergraph.py::test_boundedness_lemma1` |
| Lemma 2        | `p_{t+1} ∈ (0,1)^K`                                | `tests/test_student_model.py::test_bayes_update_stays_in_interior` |
| H1–H4          | Falsifiable hypotheses                            | `analysis.test_h1..test_h4` |
