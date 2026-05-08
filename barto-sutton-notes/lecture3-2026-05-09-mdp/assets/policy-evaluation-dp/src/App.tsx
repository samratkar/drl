import * as React from 'react';
import { useState, useEffect, useCallback } from 'react';
import { Action, Policy, MDPConfig } from './types';
import './index.css';

const GRID_SIZE = 4;
const ACTIONS: Action[] = ['UP', 'DOWN', 'LEFT', 'RIGHT'];

const DANGER_STATES = new Set([5, 9]);
const REWARD_DANGER = -5;

const CONFIG: MDPConfig = {
  gridSize: GRID_SIZE,
  gamma: 0.9,
  rewardGoal: 10,
  rewardStep: 0,
};

// Define Policies
const uniformPolicy: Policy = {};
const optimisticPolicy: Policy = {};
const suboptimalPolicy: Policy = {};

for (let s = 0; s < GRID_SIZE * GRID_SIZE; s++) {
  uniformPolicy[s] = { UP: 0.25, DOWN: 0.25, LEFT: 0.25, RIGHT: 0.25 };
  optimisticPolicy[s] = { UP: 0.1, DOWN: 0.4, LEFT: 0.1, RIGHT: 0.4 };
  suboptimalPolicy[s] = { UP: 0.4, DOWN: 0.1, LEFT: 0.4, RIGHT: 0.1 };
}

const POLICIES = [
  { id: 1, label: 'Policy 1', name: 'Uniform Random', policy: uniformPolicy },
  { id: 2, label: 'Policy 2', name: 'Optimistic (Down/Right)', policy: optimisticPolicy },
  { id: 3, label: 'Policy 3', name: 'Suboptimal (Up/Left)', policy: suboptimalPolicy },
];

const App: React.FC = () => {
  // Evaluation State
  const [iteration, setIteration] = useState(0);
  const [evalValues, setEvalValues] = useState<number[][]>(POLICIES.map(() => new Array(GRID_SIZE * GRID_SIZE).fill(0)));
  const [isEvalDone, setIsEvalDone] = useState(false);
  const [isEvalPlaying, setIsEvalPlaying] = useState(false);

  // Improvement State (Frozen Copy of Eval results)
  const [frozenEvalValues, setFrozenEvalValues] = useState<number[][] | null>(null);
  const [improvedPolicies, setImprovedPolicies] = useState<(Action | 'NONE')[][] | null>(null);
  const [qValues, setQValues] = useState<Record<string, number>[][] | null>(null);
  const [showImprovement, setShowImprovement] = useState(false);

  // Value Iteration State
  const [viValues, setViValues] = useState<number[]>(new Array(GRID_SIZE * GRID_SIZE).fill(0));
  const [viIteration, setViIteration] = useState(0);
  const [viDone, setViDone] = useState(false);
  const [viPlaying, setViPlaying] = useState(false);
  const [viPolicy, setViPolicy] = useState<(Action | 'NONE')[] | null>(null);
  const [viQValues, setViQValues] = useState<Record<string, number>[]>(
    Array.from({ length: GRID_SIZE * GRID_SIZE }, () => ({ UP: 0, DOWN: 0, LEFT: 0, RIGHT: 0 }))
  );

  // Monte Carlo State
  const [mcQTable, setMcQTable] = useState<Record<string, number>[]>(
    Array.from({ length: GRID_SIZE * GRID_SIZE }, () => ({ UP: 0, DOWN: 0, LEFT: 0, RIGHT: 0 }))
  );
  const [mcReturns, setMcReturns] = useState<Record<string, number[]>[]>(
    Array.from({ length: GRID_SIZE * GRID_SIZE }, () => ({ UP: [], DOWN: [], LEFT: [], RIGHT: [] }))
  );
  const [mcEpisode, setMcEpisode] = useState(0);
  const [mcPlaying, setMcPlaying] = useState(false);
  const [mcDone, setMcDone] = useState(false);
  const [mcLastPath, setMcLastPath] = useState<number[]>([]);
  const [mcTotalReward, setMcTotalReward] = useState(0);

  // SARSA State
  const [sarsaQTable, setSarsaQTable] = useState<Record<string, number>[]>(
    Array.from({ length: GRID_SIZE * GRID_SIZE }, () => ({ UP: 0, DOWN: 0, LEFT: 0, RIGHT: 0 }))
  );
  const [sarsaEpisode, setSarsaEpisode] = useState(0);
  const [sarsaPlaying, setSarsaPlaying] = useState(false);
  const [sarsaDone, setSarsaDone] = useState(false);
  const [sarsaLastPath, setSarsaLastPath] = useState<number[]>([]);
  const [sarsaTotalReward, setSarsaTotalReward] = useState(0);

  // Q-Learning State
  const [qlQTable, setQlQTable] = useState<Record<string, number>[]>(
    Array.from({ length: GRID_SIZE * GRID_SIZE }, () => ({ UP: 0, DOWN: 0, LEFT: 0, RIGHT: 0 }))
  );
  const [qlEpisode, setQlEpisode] = useState(0);
  const [qlPlaying, setQlPlaying] = useState(false);
  const [qlDone, setQlDone] = useState(false);
  const [qlLastPath, setQlLastPath] = useState<number[]>([]);
  const [qlTotalReward, setQlTotalReward] = useState(0);

  // Shared hyperparameters for model-free methods
  const ALPHA = 0.1;
  const EPSILON = 0.2;

  const getNextState = (s: number, a: Action): number => {
    const row = Math.floor(s / GRID_SIZE);
    const col = s % GRID_SIZE;
    let nRow = row;
    let nCol = col;

    if (a === 'UP') nRow = Math.max(0, row - 1);
    if (a === 'DOWN') nRow = Math.min(GRID_SIZE - 1, row + 1);
    if (a === 'LEFT') nCol = Math.max(0, col - 1);
    if (a === 'RIGHT') nCol = Math.min(GRID_SIZE - 1, col + 1);

    return nRow * GRID_SIZE + nCol;
  };

  const getQValue = (s: number, a: Action, values: number[]): number => {
    const nextS = getNextState(s, a);
    let reward = CONFIG.rewardStep;
    if (nextS === GRID_SIZE * GRID_SIZE - 1) reward = CONFIG.rewardGoal;
    else if (DANGER_STATES.has(nextS)) reward = REWARD_DANGER;
    return reward + CONFIG.gamma * values[nextS];
  };

  const runEvaluationStep = useCallback(() => {
    let maxDelta = 0;
    const nextValues = evalValues.map((policyValues, pIdx) => {
      const policy = POLICIES[pIdx].policy;
      const nextPolicyValues = [...policyValues];

      for (let s = 0; s < GRID_SIZE * GRID_SIZE - 1; s++) {
        let newValue = 0;
        for (const a of ACTIONS) {
          newValue += policy[s][a] * getQValue(s, a, policyValues);
        }
        nextPolicyValues[s] = newValue;
        maxDelta = Math.max(maxDelta, Math.abs(newValue - policyValues[s]));
      }
      return nextPolicyValues;
    });

    setEvalValues(nextValues);
    setIteration(prev => prev + 1);

    if (maxDelta < 0.001 && iteration > 0) {
      setIsEvalDone(true);
      setIsEvalPlaying(false);
    }
  }, [evalValues, iteration]);

  useEffect(() => {
    let interval: number;
    if (isEvalPlaying && !isEvalDone) {
      interval = setInterval(runEvaluationStep, 100);
    }
    return () => clearInterval(interval);
  }, [isEvalPlaying, isEvalDone, runEvaluationStep]);

  const startImprovementPhase = () => {
    setFrozenEvalValues(evalValues.map(v => [...v]));
    setShowImprovement(true);
  };

  const runPolicyImprovement = () => {
    if (!frozenEvalValues) return;
    const newImproved: (Action | 'NONE')[][] = [];
    const newQValues: Record<string, number>[][] = [];

    frozenEvalValues.forEach((policyValues) => {
      const bestActions: (Action | 'NONE')[] = [];
      const policyQValues: Record<string, number>[] = [];

      for (let s = 0; s < GRID_SIZE * GRID_SIZE - 1; s++) {
        let bestQ = -Infinity;
        let bestA: Action = 'UP';
        const stateQ: Record<string, number> = {};
        for (const a of ACTIONS) {
          const q = getQValue(s, a, policyValues);
          stateQ[a] = q;
          if (q > bestQ) {
            bestQ = q;
            bestA = a;
          }
        }
        bestActions[s] = bestA;
        policyQValues[s] = stateQ;
      }
      bestActions[GRID_SIZE * GRID_SIZE - 1] = 'NONE';
      policyQValues[GRID_SIZE * GRID_SIZE - 1] = { UP: 0, DOWN: 0, LEFT: 0, RIGHT: 0 };
      newImproved.push(bestActions);
      newQValues.push(policyQValues);
    });

    setImprovedPolicies(newImproved);
    setQValues(newQValues);
  };

  // Value Iteration: V(s) = max_a [r + γ V(s')]
  const runViStep = useCallback(() => {
    let maxDelta = 0;
    const nextValues = [...viValues];
    const nextQValues: Record<string, number>[] = [...viQValues];

    for (let s = 0; s < GRID_SIZE * GRID_SIZE - 1; s++) {
      let bestQ = -Infinity;
      const stateQ: Record<string, number> = {};
      for (const a of ACTIONS) {
        const q = getQValue(s, a, viValues);
        stateQ[a] = q;
        if (q > bestQ) bestQ = q;
      }
      maxDelta = Math.max(maxDelta, Math.abs(bestQ - viValues[s]));
      nextValues[s] = bestQ;
      nextQValues[s] = stateQ;
    }

    setViValues(nextValues);
    setViQValues(nextQValues);
    setViIteration(prev => prev + 1);

    if (maxDelta < 0.001 && viIteration > 0) {
      setViDone(true);
      setViPlaying(false);
    }
  }, [viValues, viIteration, viQValues]);

  useEffect(() => {
    let interval: number;
    if (viPlaying && !viDone) {
      interval = setInterval(runViStep, 100);
    }
    return () => clearInterval(interval);
  }, [viPlaying, viDone, runViStep]);

  const extractViPolicy = () => {
    const policy: (Action | 'NONE')[] = [];
    for (let s = 0; s < GRID_SIZE * GRID_SIZE - 1; s++) {
      let bestQ = -Infinity;
      let bestA: Action = 'UP';
      for (const a of ACTIONS) {
        const q = getQValue(s, a, viValues);
        if (q > bestQ) {
          bestQ = q;
          bestA = a;
        }
      }
      policy[s] = bestA;
    }
    policy[GRID_SIZE * GRID_SIZE - 1] = 'NONE';
    setViPolicy(policy);
  };

  // Check if Q-table has converged (max delta below threshold)
  const qTableMaxDelta = (oldQ: Record<string, number>[], newQ: Record<string, number>[]): number => {
    let maxDelta = 0;
    for (let s = 0; s < GRID_SIZE * GRID_SIZE - 1; s++) {
      for (const a of ACTIONS) {
        maxDelta = Math.max(maxDelta, Math.abs(newQ[s][a] - oldQ[s][a]));
      }
    }
    return maxDelta;
  };

  // Epsilon-greedy action selection from a Q-table
  const epsilonGreedyAction = (qTable: Record<string, number>[], s: number, eps: number): Action => {
    if (Math.random() < eps) {
      return ACTIONS[Math.floor(Math.random() * ACTIONS.length)];
    }
    const stateQ = qTable[s];
    let bestA: Action = 'UP';
    let bestQ = -Infinity;
    for (const a of ACTIONS) {
      if (stateQ[a] > bestQ) { bestQ = stateQ[a]; bestA = a; }
    }
    return bestA;
  };

  // Get reward for transitioning to nextS
  const getReward = (nextS: number): number => {
    if (nextS === GRID_SIZE * GRID_SIZE - 1) return CONFIG.rewardGoal;
    if (DANGER_STATES.has(nextS)) return REWARD_DANGER;
    return CONFIG.rewardStep;
  };

  // Generate a full episode (for Monte Carlo)
  const generateEpisode = (qTable: Record<string, number>[], eps: number): { states: number[]; actions: Action[]; rewards: number[] } => {
    const states: number[] = [];
    const actions: Action[] = [];
    const rewards: number[] = [];
    let s = Math.floor(Math.random() * (GRID_SIZE * GRID_SIZE - 1));
    let steps = 0;
    const maxSteps = 200;

    while (s !== GRID_SIZE * GRID_SIZE - 1 && steps < maxSteps) {
      states.push(s);
      const a = epsilonGreedyAction(qTable, s, eps);
      actions.push(a);
      const nextS = getNextState(s, a);
      const r = getReward(nextS);
      rewards.push(r);
      s = nextS;
      steps++;
    }
    states.push(s);
    return { states, actions, rewards };
  };

  // Monte Carlo: First-visit MC with epsilon-greedy
  const runMcEpisode = useCallback(() => {
    const { states, actions, rewards } = generateEpisode(mcQTable, EPSILON);
    const newQTable = mcQTable.map(q => ({ ...q }));
    const newReturns = mcReturns.map(r => ({ UP: [...r.UP], DOWN: [...r.DOWN], LEFT: [...r.LEFT], RIGHT: [...r.RIGHT] }));

    let G = 0;
    const visited = new Set<string>();

    for (let t = states.length - 2; t >= 0; t--) {
      G = CONFIG.gamma * G + rewards[t];
      const key = `${states[t]}-${actions[t]}`;
      if (!visited.has(key)) {
        visited.add(key);
        newReturns[states[t]][actions[t]].push(G);
        const returns = newReturns[states[t]][actions[t]];
        newQTable[states[t]][actions[t]] = returns.reduce((a, b) => a + b, 0) / returns.length;
      }
    }

    const delta = qTableMaxDelta(mcQTable, newQTable);
    setMcQTable(newQTable);
    setMcReturns(newReturns);
    setMcEpisode(prev => prev + 1);
    setMcLastPath(states);
    setMcTotalReward(rewards.reduce((a, b) => a + b, 0));

    if (delta < 0.01 && mcEpisode > 50) {
      setMcDone(true);
      setMcPlaying(false);
    }
  }, [mcQTable, mcReturns, mcEpisode]);

  useEffect(() => {
    let interval: number;
    if (mcPlaying && !mcDone) {
      interval = setInterval(runMcEpisode, 150);
    }
    return () => clearInterval(interval);
  }, [mcPlaying, mcDone, runMcEpisode]);

  // SARSA: On-policy TD control
  const runSarsaEpisode = useCallback(() => {
    const newQTable = sarsaQTable.map(q => ({ ...q }));
    let s = Math.floor(Math.random() * (GRID_SIZE * GRID_SIZE - 1));
    let a = epsilonGreedyAction(newQTable, s, EPSILON);
    const path: number[] = [s];
    let totalR = 0;
    let steps = 0;

    while (s !== GRID_SIZE * GRID_SIZE - 1 && steps < 200) {
      const nextS = getNextState(s, a);
      const r = getReward(nextS);
      totalR += r;
      const nextA = epsilonGreedyAction(newQTable, nextS, EPSILON);

      // SARSA update: Q(s,a) += α[r + γQ(s',a') - Q(s,a)]
      newQTable[s][a] += ALPHA * (r + CONFIG.gamma * newQTable[nextS][nextA] - newQTable[s][a]);

      s = nextS;
      a = nextA;
      path.push(s);
      steps++;
    }

    const delta = qTableMaxDelta(sarsaQTable, newQTable);
    setSarsaQTable(newQTable);
    setSarsaEpisode(prev => prev + 1);
    setSarsaLastPath(path);
    setSarsaTotalReward(totalR);

    if (delta < 0.01 && sarsaEpisode > 50) {
      setSarsaDone(true);
      setSarsaPlaying(false);
    }
  }, [sarsaQTable, sarsaEpisode]);

  useEffect(() => {
    let interval: number;
    if (sarsaPlaying && !sarsaDone) {
      interval = setInterval(runSarsaEpisode, 150);
    }
    return () => clearInterval(interval);
  }, [sarsaPlaying, sarsaDone, runSarsaEpisode]);

  // Q-Learning: Off-policy TD control
  const runQlEpisode = useCallback(() => {
    const newQTable = qlQTable.map(q => ({ ...q }));
    let s = Math.floor(Math.random() * (GRID_SIZE * GRID_SIZE - 1));
    const path: number[] = [s];
    let totalR = 0;
    let steps = 0;

    while (s !== GRID_SIZE * GRID_SIZE - 1 && steps < 200) {
      const a = epsilonGreedyAction(newQTable, s, EPSILON);
      const nextS = getNextState(s, a);
      const r = getReward(nextS);
      totalR += r;

      // Q-Learning update: Q(s,a) += α[r + γ max_a' Q(s',a') - Q(s,a)]
      const maxNextQ = Math.max(...ACTIONS.map(na => newQTable[nextS][na]));
      newQTable[s][a] += ALPHA * (r + CONFIG.gamma * maxNextQ - newQTable[s][a]);

      s = nextS;
      path.push(s);
      steps++;
    }

    const delta = qTableMaxDelta(qlQTable, newQTable);
    setQlQTable(newQTable);
    setQlEpisode(prev => prev + 1);
    setQlLastPath(path);
    setQlTotalReward(totalR);

    if (delta < 0.01 && qlEpisode > 50) {
      setQlDone(true);
      setQlPlaying(false);
    }
  }, [qlQTable, qlEpisode]);

  useEffect(() => {
    let interval: number;
    if (qlPlaying && !qlDone) {
      interval = setInterval(runQlEpisode, 150);
    }
    return () => clearInterval(interval);
  }, [qlPlaying, qlDone, runQlEpisode]);

  const resetAll = () => {
    setIteration(0);
    setEvalValues(POLICIES.map(() => new Array(GRID_SIZE * GRID_SIZE).fill(0)));
    setIsEvalDone(false);
    setIsEvalPlaying(false);
    setFrozenEvalValues(null);
    setImprovedPolicies(null);
    setQValues(null);
    setShowImprovement(false);
    setViValues(new Array(GRID_SIZE * GRID_SIZE).fill(0));
    setViQValues(Array.from({ length: GRID_SIZE * GRID_SIZE }, () => ({ UP: 0, DOWN: 0, LEFT: 0, RIGHT: 0 })));
    setViIteration(0);
    setViDone(false);
    setViPlaying(false);
    setViPolicy(null);
    setMcQTable(Array.from({ length: GRID_SIZE * GRID_SIZE }, () => ({ UP: 0, DOWN: 0, LEFT: 0, RIGHT: 0 })));
    setMcReturns(Array.from({ length: GRID_SIZE * GRID_SIZE }, () => ({ UP: [], DOWN: [], LEFT: [], RIGHT: [] })));
    setMcEpisode(0);
    setMcPlaying(false);
    setMcDone(false);
    setMcLastPath([]);
    setMcTotalReward(0);
    setSarsaQTable(Array.from({ length: GRID_SIZE * GRID_SIZE }, () => ({ UP: 0, DOWN: 0, LEFT: 0, RIGHT: 0 })));
    setSarsaEpisode(0);
    setSarsaPlaying(false);
    setSarsaDone(false);
    setSarsaLastPath([]);
    setSarsaTotalReward(0);
    setQlQTable(Array.from({ length: GRID_SIZE * GRID_SIZE }, () => ({ UP: 0, DOWN: 0, LEFT: 0, RIGHT: 0 })));
    setQlEpisode(0);
    setQlPlaying(false);
    setQlDone(false);
    setQlLastPath([]);
    setQlTotalReward(0);
  };

  const actionArrow = (a: Action) => {
    switch (a) {
      case 'UP': return '↑';
      case 'DOWN': return '↓';
      case 'LEFT': return '←';
      case 'RIGHT': return '→';
    }
  };

  return (
    <div className="container">
      <header style={{textAlign: 'center', marginBottom: '2rem'}}>
        <h1>Dynamic Programming: Policy Iteration vs. Value Iteration</h1>
      </header>

      <section className="phase-container" style={{border: '2px solid #2196f3', padding: '20px', borderRadius: '12px'}}>
        <h2 style={{color: '#2196f3'}}>The MDP Environment</h2>

        <div style={{display: 'flex', gap: '2.5rem', alignItems: 'center', flexWrap: 'wrap'}}>
          <div className="grid-container grid-large" style={{borderColor: '#2196f3'}}>
            {Array.from({ length: GRID_SIZE * GRID_SIZE }).map((_, sIdx) => {
              const isTerminal = sIdx === GRID_SIZE * GRID_SIZE - 1;
              const isDanger = DANGER_STATES.has(sIdx);
              return (
                <div
                  key={sIdx}
                  className={`cell ${isTerminal ? 'terminal' : ''} ${isDanger ? 'danger' : ''}`}
                  style={{
                    backgroundColor: isTerminal ? '#1a4d2e' : isDanger ? '#4d1a1a' : '#2a2a3a',
                    position: 'relative',
                  }}
                >
                  <span className="cell-id">{sIdx}</span>
                  {isTerminal ? (
                    <div style={{textAlign: 'center'}}>
                      <div style={{fontSize: '1.2rem'}}>🏁</div>
                      <div style={{fontSize: '0.6rem', color: '#4caf50'}}>R={CONFIG.rewardGoal}</div>
                    </div>
                  ) : isDanger ? (
                    <div style={{textAlign: 'center'}}>
                      <div style={{fontSize: '1.2rem'}}>💀</div>
                      <div style={{fontSize: '0.6rem', color: '#f44336'}}>R={REWARD_DANGER}</div>
                    </div>
                  ) : (
                    <div className="action-arrows">
                      <span className="arrow arrow-up">↑</span>
                      <span className="arrow arrow-left">←</span>
                      <span className="arrow arrow-right">→</span>
                      <span className="arrow arrow-down">↓</span>
                    </div>
                  )}
                </div>
              );
            })}
          </div>

          <div style={{display: 'flex', flexDirection: 'column', gap: '0.5rem'}}>
            <p style={{margin: '0 0 0.5rem 0', fontSize: '0.85rem', maxWidth: '220px'}}>4×4 gridworld. Reach the 🏁 Goal while avoiding 💀 danger zones.</p>

            <div style={{background: '#1a1a2e', padding: '6px 12px', borderRadius: '6px', border: '1px solid #2196f3', fontSize: '0.8rem'}}>
              <strong>S:</strong> {'{'} 0 .. 15 {'}'}
            </div>
            <div style={{background: '#1a1a2e', padding: '6px 12px', borderRadius: '6px', border: '1px solid #2196f3', fontSize: '0.8rem'}}>
              <strong>A:</strong> {'{'} ↑ ↓ ← → {'}'}
            </div>
            <div style={{background: '#1a1a2e', padding: '6px 12px', borderRadius: '6px', border: '1px solid #2196f3', fontSize: '0.8rem'}}>
              <strong>R:</strong> +{CONFIG.rewardGoal} goal, {REWARD_DANGER} danger, {CONFIG.rewardStep} step
            </div>
            <div style={{background: '#1a1a2e', padding: '6px 12px', borderRadius: '6px', border: '1px solid #2196f3', fontSize: '0.8rem'}}>
              <strong>γ:</strong> {CONFIG.gamma}
            </div>
            <div style={{background: '#1a1a2e', padding: '6px 12px', borderRadius: '6px', border: '1px solid #2196f3', fontSize: '0.8rem'}}>
              <strong>Dynamics:</strong> Deterministic
            </div>
          </div>
        </div>
      </section>

      <div style={{width: '100%', textAlign: 'center', padding: '10px', background: 'linear-gradient(90deg, #ffa500, #4caf50)', borderRadius: '8px'}}>
        <h2 style={{margin: 0, color: '#000'}}>METHOD 1: Policy Iteration</h2>
        <p style={{margin: '4px 0 0', color: '#222', fontSize: '0.85rem'}}>Evaluate fully, then improve. Repeat until optimal.</p>
      </div>

      <section className="phase-container" style={{border: '2px solid #333', padding: '20px', borderRadius: '12px'}}>
        <h2 style={{color: '#ffa500'}}>STEP 1: Policy Evaluation</h2>
        <div className="equation">
          V<sub>π</sub>(s) = Σ<sub>a</sub> π(a|s) [r + γ V<sub>π</sub>(s')]
        </div>
        <p>Iteratively update state values until they converge for the given policies. <span style={{color: '#888', fontSize: '0.8rem'}}>Convergence: max|ΔV| &lt; 0.001</span></p>
        
        <div className="controls">
          <button onClick={() => setIsEvalPlaying(!isEvalPlaying)} disabled={isEvalDone}>
            {isEvalPlaying ? 'Pause Evaluation' : 'Start Evaluation'}
          </button>
          <button onClick={runEvaluationStep} disabled={isEvalDone || isEvalPlaying}>Single Step</button>
          <button onClick={resetAll}>Reset All</button>
          <span style={{marginLeft: '10px'}}>Iteration: {iteration}</span>
          {isEvalDone && <span style={{ color: '#4caf50', fontWeight: 'bold' }}> - CONVERGED!</span>}
        </div>

        <div className="dashboard">
          {POLICIES.map((p, pIdx) => (
            <div key={p.label} className="policy-view">
              <h3>{p.label}<br/><span style={{fontSize: '0.8em', opacity: 0.7}}>{p.name}</span></h3>
              <div className="grid-container">
                {evalValues[pIdx].map((v, sIdx) => {
                  const isTerminal = sIdx === GRID_SIZE * GRID_SIZE - 1;
                  const isDanger = DANGER_STATES.has(sIdx);
                  let bgColor: string | undefined;
                  if (!isTerminal) {
                    if (v >= 0) bgColor = `rgba(255, 165, 0, ${Math.min(v / 10, 1)})`;
                    else bgColor = `rgba(244, 67, 54, ${Math.min(Math.abs(v) / 5, 0.8)})`;
                  }
                  return (
                    <div
                      key={sIdx}
                      className={`cell ${isTerminal ? 'terminal' : ''} ${isDanger ? 'danger' : ''}`}
                      style={{ backgroundColor: bgColor }}
                    >
                      <span className="cell-id">{isDanger ? '💀' : sIdx}</span>
                      <span className="cell-value">{v.toFixed(2)}</span>
                    </div>
                  );
                })}
              </div>
            </div>
          ))}
        </div>

        <div className="policy-tables">
          {POLICIES.map((p, pIdx) => (
            <div key={p.label + '_table'}>
              <h4>{p.label}<br/><span style={{fontSize: '0.8em', opacity: 0.7}}>{p.name}</span></h4>
              <table>
                <thead>
                  <tr>
                    <th>State</th>
                    <th>UP</th>
                    <th>DOWN</th>
                    <th>LEFT</th>
                    <th>RIGHT</th>
                    <th style={{backgroundColor: '#1a3a3a'}}>V<sub>π</sub></th>
                  </tr>
                </thead>
                <tbody>
                  {Array.from({ length: GRID_SIZE * GRID_SIZE }).map((_, sIdx) => (
                    <tr key={sIdx}>
                      <td>{sIdx}</td>
                      <td>{p.policy[sIdx].UP.toFixed(2)}</td>
                      <td>{p.policy[sIdx].DOWN.toFixed(2)}</td>
                      <td>{p.policy[sIdx].LEFT.toFixed(2)}</td>
                      <td>{p.policy[sIdx].RIGHT.toFixed(2)}</td>
                      <td style={{fontWeight: 'bold', color: isEvalDone ? '#4caf50' : '#ffa500'}}>
                        {evalValues[pIdx][sIdx].toFixed(2)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ))}
        </div>

        {isEvalDone && !showImprovement && (
          <div style={{textAlign: 'center', marginTop: '20px'}}>
            <button 
              onClick={startImprovementPhase}
              style={{padding: '15px 30px', fontSize: '1.2rem', backgroundColor: '#4caf50', border: 'none', borderRadius: '8px', color: 'white'}}
            >
              Unlock Step 2: Policy Improvement
            </button>
          </div>
        )}
      </section>

      {showImprovement && (
        <section className="phase-container" style={{border: '2px solid #4caf50', padding: '20px', borderRadius: '12px', background: '#111'}}>
          <h2 style={{color: '#4caf50'}}>STEP 2: Policy Improvement (Manual)</h2>
          <p>
            The values below are <strong>FROZEN</strong> from the evaluation above. 
            Now, click "Improve Policy" to apply the <strong>argmax</strong> rule and transform the stochastic policy into a greedy optimal one.
          </p>
          
          <div className="equation" style={{background: '#000', border: '1px solid #4caf50'}}>
            π'(s) = argmax<sub>a</sub> [r + γ V<sub>π</sub>(s')]
          </div>

          <div style={{textAlign: 'center', marginBottom: '1rem'}}>
            <button
              onClick={runPolicyImprovement}
              disabled={!!improvedPolicies}
              style={{backgroundColor: improvedPolicies ? '#333' : '#4caf50', padding: '10px 24px', border: 'none', borderRadius: '8px', color: '#fff', fontSize: '1rem'}}
            >
              {improvedPolicies ? 'Policy Improved!' : 'Improve Policy (Run argmax)'}
            </button>
          </div>

          <div className="dashboard">
            {POLICIES.map((p, pIdx) => (
              <div key={p.label + '_improved'} className="policy-view">
                <h3>{p.label}<br/><span style={{fontSize: '0.8em', opacity: 0.7}}>{p.name} (Result)</span></h3>
                <div className="grid-container" style={{borderColor: '#4caf50'}}>
                  {frozenEvalValues![pIdx].map((v, sIdx) => {
                    const isTerminal = sIdx === GRID_SIZE * GRID_SIZE - 1;
                    const isDanger = DANGER_STATES.has(sIdx);
                    let bgColor: string | undefined;
                    if (!isTerminal) {
                      if (v >= 0) bgColor = `rgba(76, 175, 80, ${Math.min(v / 10, 0.4)})`;
                      else bgColor = `rgba(244, 67, 54, ${Math.min(Math.abs(v) / 5, 0.6)})`;
                    }
                    return (
                      <div
                        key={sIdx}
                        className={`cell ${isTerminal ? 'terminal' : ''} ${isDanger ? 'danger' : ''}`}
                        style={{ backgroundColor: bgColor }}
                      >
                        <span className="cell-id">{isDanger ? '💀' : sIdx}</span>
                        <span className="cell-value" style={{opacity: 0.6}}>{v.toFixed(2)}</span>
                        {improvedPolicies && !isTerminal && (
                          <div style={{color: '#fff', fontWeight: 'bold', fontSize: '0.9rem', backgroundColor: '#2e7d32', padding: '2px 4px', borderRadius: '4px'}}>
                            {improvedPolicies[pIdx][sIdx]}
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>
            ))}
          </div>

          <div className="policy-tables">
            {POLICIES.map((p, pIdx) => (
              <div key={p.label + '_table_improved'}>
                <h4>{p.label}<br/><span style={{fontSize: '0.8em', opacity: 0.7}}>{p.name} — Improved π'</span></h4>
                <table>
                  <thead>
                    <tr>
                      <th rowSpan={2}>State</th>
                      <th colSpan={4}>π'(a|s) / Q(s,a)</th>
                      <th rowSpan={2} style={{backgroundColor: '#1a3a3a'}}>V<sub>π</sub></th>
                    </tr>
                    <tr>
                      <th>UP</th>
                      <th>DOWN</th>
                      <th>LEFT</th>
                      <th>RIGHT</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Array.from({ length: GRID_SIZE * GRID_SIZE }).map((_, sIdx) => {
                      const bestAction = improvedPolicies ? improvedPolicies[pIdx][sIdx] : null;
                      const origPolicy = p.policy[sIdx];
                      const stateQ = qValues ? qValues[pIdx][sIdx] : null;
                      return (
                        <tr key={sIdx}>
                          <td>{sIdx}</td>
                          {bestAction && stateQ ? (
                            <>
                              {ACTIONS.map(a => (
                                <td key={a} style={{color: bestAction === a ? '#4caf50' : '#888', fontWeight: bestAction === a ? 'bold' : 'normal'}}>
                                  <div>{bestAction === a ? '1.00' : '0.00'}</div>
                                  <div style={{fontSize: '0.65rem', color: bestAction === a ? '#81c784' : '#666', marginTop: '2px'}}>
                                    Q={stateQ[a].toFixed(2)}
                                  </div>
                                </td>
                              ))}
                            </>
                          ) : (
                            <>
                              <td>{origPolicy.UP.toFixed(2)}</td>
                              <td>{origPolicy.DOWN.toFixed(2)}</td>
                              <td>{origPolicy.LEFT.toFixed(2)}</td>
                              <td>{origPolicy.RIGHT.toFixed(2)}</td>
                            </>
                          )}
                          <td style={{background: '#0a0a0a'}}>{frozenEvalValues![pIdx][sIdx].toFixed(2)}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            ))}
          </div>
        </section>
      )}

      {/* VALUE ITERATION */}
      <div style={{width: '100%', textAlign: 'center', padding: '10px', background: 'linear-gradient(90deg, #9c27b0, #e91e63)', borderRadius: '8px', marginTop: '2rem'}}>
        <h2 style={{margin: 0, color: '#fff'}}>METHOD 2: Value Iteration</h2>
        <p style={{margin: '4px 0 0', color: '#eee', fontSize: '0.85rem'}}>Combine evaluation + improvement in every sweep. No separate policy needed.</p>
      </div>

      <section className="phase-container" style={{border: '2px solid #9c27b0', padding: '20px', borderRadius: '12px'}}>
        <h2 style={{color: '#ce93d8'}}>Value Iteration</h2>
        <div className="equation" style={{border: '1px solid #9c27b0'}}>
          V(s) = max<sub>a</sub> [r + γ V(s')]
        </div>
        <p>At each sweep, every state takes the <strong>max</strong> over all actions — evaluation and improvement happen simultaneously. <span style={{color: '#888', fontSize: '0.8rem'}}>Convergence: max|ΔV| &lt; 0.001</span></p>

        <div className="controls">
          <button onClick={() => setViPlaying(!viPlaying)} disabled={viDone} style={{borderColor: '#9c27b0'}}>
            {viPlaying ? 'Pause' : 'Start Value Iteration'}
          </button>
          <button onClick={runViStep} disabled={viDone || viPlaying}>Single Step</button>
          <button onClick={resetAll}>Reset All</button>
          <span style={{marginLeft: '10px'}}>Iteration: {viIteration}</span>
          {viDone && <span style={{ color: '#ce93d8', fontWeight: 'bold' }}> - CONVERGED!</span>}
        </div>

        <div style={{display: 'flex', gap: '2rem', flexWrap: 'wrap', justifyContent: 'center', alignItems: 'flex-start'}}>
          <div>
            <h3 style={{textAlign: 'center'}}>V*(s)</h3>
            <div className="grid-container" style={{borderColor: '#9c27b0'}}>
              {viValues.map((v, sIdx) => {
                const isTerminal = sIdx === GRID_SIZE * GRID_SIZE - 1;
                const isDanger = DANGER_STATES.has(sIdx);
                let bgColor: string | undefined;
                if (!isTerminal) {
                  if (v >= 0) bgColor = `rgba(206, 147, 216, ${Math.min(v / 10, 1)})`;
                  else bgColor = `rgba(244, 67, 54, ${Math.min(Math.abs(v) / 5, 0.8)})`;
                }
                return (
                  <div
                    key={sIdx}
                    className={`cell ${isTerminal ? 'terminal' : ''} ${isDanger ? 'danger' : ''}`}
                    style={{ backgroundColor: bgColor }}
                  >
                    <span className="cell-id">{isDanger ? '💀' : sIdx}</span>
                    <span className="cell-value">{v.toFixed(2)}</span>
                  </div>
                );
              })}
            </div>
          </div>

          {viDone && (
            <div>
              <h3 style={{textAlign: 'center'}}>π*(s) — Extracted Policy</h3>
              {!viPolicy && (
                <div style={{textAlign: 'center', marginBottom: '1rem'}}>
                  <button onClick={extractViPolicy} style={{backgroundColor: '#9c27b0', border: 'none', color: '#fff', padding: '8px 16px', borderRadius: '6px'}}>
                    Extract Policy from V*
                  </button>
                </div>
              )}
              <div className="grid-container" style={{borderColor: '#e91e63'}}>
                {viValues.map((v, sIdx) => {
                  const isTerminal = sIdx === GRID_SIZE * GRID_SIZE - 1;
                  const isDanger = DANGER_STATES.has(sIdx);
                  let bgColor: string | undefined;
                  if (!isTerminal) {
                    if (v >= 0) bgColor = `rgba(206, 147, 216, ${Math.min(v / 10, 0.4)})`;
                    else bgColor = `rgba(244, 67, 54, ${Math.min(Math.abs(v) / 5, 0.4)})`;
                  }
                  return (
                    <div
                      key={sIdx}
                      className={`cell ${isTerminal ? 'terminal' : ''} ${isDanger ? 'danger' : ''}`}
                      style={{ backgroundColor: bgColor }}
                    >
                      <span className="cell-id">{isDanger ? '💀' : sIdx}</span>
                      {viPolicy && !isTerminal ? (
                        <div style={{color: '#fff', fontWeight: 'bold', fontSize: '1.1rem'}}>
                          {actionArrow(viPolicy[sIdx] as Action)}
                        </div>
                      ) : isTerminal ? (
                        <span style={{fontSize: '1rem'}}>🏁</span>
                      ) : (
                        <span className="cell-value" style={{opacity: 0.5}}>?</span>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>

        {viIteration > 0 && (
          <div className="policy-tables" style={{marginTop: '1.5rem'}}>
            <div>
              <h4>Value Iteration<br/><span style={{fontSize: '0.8em', opacity: 0.7}}>Q(s,a) → max → V(s) | Iteration {viIteration}</span></h4>
              <table>
                <thead>
                  <tr>
                    <th rowSpan={2}>State</th>
                    <th colSpan={4}>Q(s,a) &amp; π(a|s)</th>
                    <th rowSpan={2} style={{backgroundColor: '#2a1a3a'}}>V(s) = max Q</th>
                  </tr>
                  <tr>
                    <th>UP</th>
                    <th>DOWN</th>
                    <th>LEFT</th>
                    <th>RIGHT</th>
                  </tr>
                </thead>
                <tbody>
                  {Array.from({ length: GRID_SIZE * GRID_SIZE }).map((_, sIdx) => {
                    const stateQ = viQValues[sIdx];
                    const maxQ = Math.max(...ACTIONS.map(a => stateQ[a]));
                    const isTerminal = sIdx === GRID_SIZE * GRID_SIZE - 1;
                    return (
                      <tr key={sIdx}>
                        <td>{sIdx}</td>
                        {ACTIONS.map(a => {
                          const isBest = !isTerminal && stateQ[a] === maxQ;
                          return (
                            <td key={a} style={{color: isBest ? '#ce93d8' : '#888', fontWeight: isBest ? 'bold' : 'normal'}}>
                              <div style={{fontSize: '0.7rem', color: isBest ? '#ce93d8' : '#666'}}>
                                Q={stateQ[a].toFixed(2)}
                              </div>
                              <div>
                                {isTerminal ? '-' : isBest ? '1.00' : '0.00'}
                              </div>
                            </td>
                          );
                        })}
                        <td style={{background: '#1a0a2a', fontWeight: 'bold', color: '#ce93d8'}}>
                          {viValues[sIdx].toFixed(2)}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </section>

      {/* COMPARISON */}
      {(isEvalDone || viDone) && (
        <section className="phase-container" style={{border: '2px solid #ffeb3b', padding: '20px', borderRadius: '12px', background: '#1a1a0a'}}>
          <h2 style={{color: '#ffeb3b'}}>Comparison: Policy Iteration vs. Value Iteration</h2>

          <div style={{display: 'flex', gap: '2rem', justifyContent: 'center', flexWrap: 'wrap'}}>
            <div style={{background: '#111', border: '1px solid #ffa500', borderRadius: '8px', padding: '16px', minWidth: '280px'}}>
              <h3 style={{color: '#ffa500', marginTop: 0}}>Policy Iteration</h3>
              <table style={{width: '100%'}}>
                <tbody>
                  <tr><td>Eval Iterations</td><td style={{fontWeight: 'bold', color: '#ffa500'}}>{iteration}</td></tr>
                  <tr><td>Improvement Steps</td><td style={{fontWeight: 'bold', color: '#4caf50'}}>1</td></tr>
                  <tr><td>Total Sweeps</td><td style={{fontWeight: 'bold'}}>{iteration + 1}</td></tr>
                  <tr><td style={{paddingTop: '8px'}}>Update Rule</td><td style={{paddingTop: '8px', fontSize: '0.75rem'}}>V = Σ π(a|s) · Q(s,a)</td></tr>
                  <tr><td>Needs Policy?</td><td style={{color: '#ffa500'}}>Yes (π given)</td></tr>
                  <tr><td>When to use</td><td style={{fontSize: '0.75rem'}}>Few iterations to converge each eval</td></tr>
                </tbody>
              </table>
            </div>

            <div style={{background: '#111', border: '1px solid #9c27b0', borderRadius: '8px', padding: '16px', minWidth: '280px'}}>
              <h3 style={{color: '#ce93d8', marginTop: 0}}>Value Iteration</h3>
              <table style={{width: '100%'}}>
                <tbody>
                  <tr><td>Iterations</td><td style={{fontWeight: 'bold', color: '#ce93d8'}}>{viIteration}</td></tr>
                  <tr><td>Improvement Steps</td><td style={{fontWeight: 'bold', color: '#4caf50'}}>Every sweep</td></tr>
                  <tr><td>Total Sweeps</td><td style={{fontWeight: 'bold'}}>{viIteration}</td></tr>
                  <tr><td style={{paddingTop: '8px'}}>Update Rule</td><td style={{paddingTop: '8px', fontSize: '0.75rem'}}>V = max_a Q(s,a)</td></tr>
                  <tr><td>Needs Policy?</td><td style={{color: '#ce93d8'}}>No (implicit max)</td></tr>
                  <tr><td>When to use</td><td style={{fontSize: '0.75rem'}}>Direct path to V*, no policy overhead</td></tr>
                </tbody>
              </table>
            </div>
          </div>

          <div style={{marginTop: '1.5rem', textAlign: 'center', fontSize: '0.9rem', color: '#ccc', maxWidth: '700px', margin: '1.5rem auto 0'}}>
            <strong style={{color: '#ffeb3b'}}>Key Insight:</strong> Both converge to the same V* and π*. Policy Iteration separates "evaluate" and "improve" cleanly. Value Iteration merges them — each Bellman update already takes the max, so it's both evaluating and improving at once. Value Iteration typically needs fewer total sweeps but each sweep does more work (max instead of weighted sum).
          </div>
        </section>
      )}

      {/* MONTE CARLO */}
      <div style={{width: '100%', textAlign: 'center', padding: '10px', background: 'linear-gradient(90deg, #00bcd4, #009688)', borderRadius: '8px', marginTop: '2rem'}}>
        <h2 style={{margin: 0, color: '#fff'}}>METHOD 3: Monte Carlo (Model-Free)</h2>
        <p style={{margin: '4px 0 0', color: '#eee', fontSize: '0.85rem'}}>Learn from complete episodes. No model needed — just experience.</p>
      </div>

      <section className="phase-container" style={{border: '2px solid #00bcd4', padding: '20px', borderRadius: '12px'}}>
        <h2 style={{color: '#4dd0e1'}}>Monte Carlo — First-Visit</h2>
        <div className="equation" style={{border: '1px solid #00bcd4'}}>
          Q(s,a) = average(Returns(s,a))
        </div>
        <p>Generate full episodes using ε-greedy policy, then update Q-values with the <strong>average return</strong> from first visit to each (s, a) pair. <span style={{color: '#888', fontSize: '0.8rem'}}>Convergence: max|ΔQ| &lt; 0.01 after 50+ episodes</span></p>

        <div className="controls">
          <button onClick={() => setMcPlaying(!mcPlaying)} disabled={mcDone} style={{borderColor: '#00bcd4'}}>
            {mcPlaying ? 'Pause' : 'Start MC'}
          </button>
          <button onClick={runMcEpisode} disabled={mcPlaying || mcDone}>Run Episode</button>
          <button onClick={resetAll}>Reset All</button>
          <span style={{marginLeft: '10px'}}>Episodes: {mcEpisode}</span>
          <span style={{marginLeft: '10px', color: '#4dd0e1'}}>Last reward: {mcTotalReward.toFixed(1)}</span>
          {mcDone && <span style={{ color: '#4caf50', fontWeight: 'bold' }}> - CONVERGED!</span>}
        </div>

        <div style={{display: 'flex', gap: '2rem', flexWrap: 'wrap', justifyContent: 'center', alignItems: 'flex-start'}}>
          <div>
            <h3 style={{textAlign: 'center'}}>Greedy Policy from Q</h3>
            <div className="grid-container" style={{borderColor: '#00bcd4'}}>
              {mcQTable.map((stateQ, sIdx) => {
                const isTerminal = sIdx === GRID_SIZE * GRID_SIZE - 1;
                const isDanger = DANGER_STATES.has(sIdx);
                const maxQ = Math.max(...ACTIONS.map(a => stateQ[a]));
                const bestA = ACTIONS.find(a => stateQ[a] === maxQ) || 'UP';
                const onPath = mcLastPath.includes(sIdx);
                let bgColor: string | undefined;
                if (!isTerminal) {
                  if (maxQ >= 0) bgColor = `rgba(0, 188, 212, ${Math.min(maxQ / 10, 0.8)})`;
                  else bgColor = `rgba(244, 67, 54, ${Math.min(Math.abs(maxQ) / 5, 0.6)})`;
                }
                return (
                  <div
                    key={sIdx}
                    className={`cell ${isTerminal ? 'terminal' : ''} ${isDanger ? 'danger' : ''}`}
                    style={{ backgroundColor: bgColor, border: onPath ? '2px solid #fff' : undefined }}
                  >
                    <span className="cell-id">{isDanger ? '💀' : sIdx}</span>
                    {isTerminal ? (
                      <span style={{fontSize: '1rem'}}>🏁</span>
                    ) : mcEpisode > 0 ? (
                      <div style={{color: '#fff', fontWeight: 'bold', fontSize: '1rem'}}>
                        {actionArrow(bestA)}
                      </div>
                    ) : (
                      <span className="cell-value" style={{opacity: 0.5}}>?</span>
                    )}
                    {!isTerminal && <span style={{fontSize: '0.55rem', position: 'absolute', bottom: 2, color: '#aaa'}}>{maxQ.toFixed(1)}</span>}
                  </div>
                );
              })}
            </div>
          </div>

          {mcLastPath.length > 0 && (
            <div>
              <h3 style={{textAlign: 'center'}}>Last Episode Path<br/><span style={{fontSize: '0.7em', opacity: 0.7}}>{mcLastPath.length - 1} steps</span></h3>
              <div className="grid-container" style={{borderColor: '#009688'}}>
                {Array.from({ length: GRID_SIZE * GRID_SIZE }).map((_, sIdx) => {
                  const isTerminal = sIdx === GRID_SIZE * GRID_SIZE - 1;
                  const isDanger = DANGER_STATES.has(sIdx);
                  const pathIdx = mcLastPath.indexOf(sIdx);
                  const onPath = pathIdx !== -1;
                  const stateQ = mcQTable[sIdx];
                  const maxQ = Math.max(...ACTIONS.map(a => stateQ[a]));
                  return (
                    <div
                      key={sIdx}
                      className={`cell ${isTerminal ? 'terminal' : ''} ${isDanger ? 'danger' : ''}`}
                      style={{ backgroundColor: onPath ? 'rgba(0, 188, 212, 0.4)' : undefined }}
                    >
                      <span className="cell-id">{isDanger ? '💀' : sIdx}</span>
                      {onPath && <span style={{fontWeight: 'bold', color: '#4dd0e1'}}>{pathIdx === 0 ? 'S' : pathIdx === mcLastPath.length - 1 ? 'E' : pathIdx}</span>}
                      {!isTerminal && <span style={{fontSize: '0.5rem', position: 'absolute', bottom: 2, color: '#aaa'}}>{maxQ.toFixed(1)}</span>}
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>

        {mcEpisode > 0 && (
          <div className="policy-tables" style={{marginTop: '1.5rem'}}>
            <div>
              <h4>Monte Carlo Q-Table<br/><span style={{fontSize: '0.8em', opacity: 0.7}}>Episode {mcEpisode} | ε={EPSILON} | γ={CONFIG.gamma}</span></h4>
              <table>
                <thead>
                  <tr>
                    <th>State</th>
                    <th>Q(s,↑)</th>
                    <th>Q(s,↓)</th>
                    <th>Q(s,←)</th>
                    <th>Q(s,→)</th>
                    <th style={{backgroundColor: '#0a2a2a'}}>Best</th>
                  </tr>
                </thead>
                <tbody>
                  {mcQTable.map((stateQ, sIdx) => {
                    const maxQ = Math.max(...ACTIONS.map(a => stateQ[a]));
                    return (
                      <tr key={sIdx}>
                        <td>{sIdx}</td>
                        {ACTIONS.map(a => (
                          <td key={a} style={{color: stateQ[a] === maxQ && (maxQ !== 0 || ACTIONS.some(act => stateQ[act] !== 0)) ? '#4dd0e1' : '#888', fontWeight: stateQ[a] === maxQ && (maxQ !== 0 || ACTIONS.some(act => stateQ[act] !== 0)) ? 'bold' : 'normal'}}>
                            {stateQ[a].toFixed(2)}
                          </td>
                        ))}
                        <td style={{color: '#4dd0e1', fontWeight: 'bold'}}>
                          {(maxQ !== 0 || ACTIONS.some(a => stateQ[a] !== 0)) ? actionArrow(ACTIONS.find(a => stateQ[a] === maxQ) as Action) : '-'}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </section>

      {/* TD METHODS: SARSA & Q-LEARNING */}
      <div style={{width: '100%', textAlign: 'center', padding: '10px', background: 'linear-gradient(90deg, #ff5722, #ff9800)', borderRadius: '8px', marginTop: '2rem'}}>
        <h2 style={{margin: 0, color: '#fff'}}>METHOD 4: Temporal Difference (Model-Free)</h2>
        <p style={{margin: '4px 0 0', color: '#eee', fontSize: '0.85rem'}}>Learn from incomplete episodes. Bootstrap from current estimates — no need to wait for episode end.</p>
      </div>

      <section className="phase-container" style={{border: '2px solid #ff5722', padding: '20px', borderRadius: '12px'}}>
        <div style={{display: 'flex', gap: '2rem', flexWrap: 'wrap', justifyContent: 'center'}}>

          {/* SARSA */}
          <div style={{flex: 1, minWidth: '400px'}}>
            <h2 style={{color: '#ff8a65'}}>SARSA (On-Policy)</h2>
            <div className="equation" style={{border: '1px solid #ff5722', fontSize: '1rem'}}>
              Q(s,a) ← Q(s,a) + α[r + γQ(s',a') - Q(s,a)]
            </div>
            <p style={{fontSize: '0.85rem'}}>Updates Q using the <strong>action actually taken</strong> (a') in the next state. Learns the value of the policy it follows. <span style={{color: '#888', fontSize: '0.75rem'}}>Convergence: max|ΔQ| &lt; 0.01 after 50+ episodes</span></p>

            <div className="controls" style={{flexWrap: 'wrap'}}>
              <button onClick={() => setSarsaPlaying(!sarsaPlaying)} disabled={sarsaDone} style={{borderColor: '#ff5722'}}>
                {sarsaPlaying ? 'Pause' : 'Start SARSA'}
              </button>
              <button onClick={runSarsaEpisode} disabled={sarsaPlaying || sarsaDone}>Run Episode</button>
              <span style={{fontSize: '0.8rem'}}>Ep: {sarsaEpisode} | R: {sarsaTotalReward.toFixed(1)}</span>
              {sarsaDone && <span style={{ color: '#4caf50', fontWeight: 'bold', fontSize: '0.8rem' }}> CONVERGED!</span>}
            </div>

            <div style={{display: 'flex', justifyContent: 'center'}}>
              <div className="grid-container" style={{borderColor: '#ff5722'}}>
                {sarsaQTable.map((stateQ, sIdx) => {
                  const isTerminal = sIdx === GRID_SIZE * GRID_SIZE - 1;
                  const isDanger = DANGER_STATES.has(sIdx);
                  const maxQ = Math.max(...ACTIONS.map(a => stateQ[a]));
                  const bestA = ACTIONS.find(a => stateQ[a] === maxQ) || 'UP';
                  const onPath = sarsaLastPath.includes(sIdx);
                  let bgColor: string | undefined;
                  if (!isTerminal) {
                    if (maxQ >= 0) bgColor = `rgba(255, 138, 101, ${Math.min(maxQ / 10, 0.8)})`;
                    else bgColor = `rgba(244, 67, 54, ${Math.min(Math.abs(maxQ) / 5, 0.6)})`;
                  }
                  return (
                    <div
                      key={sIdx}
                      className={`cell ${isTerminal ? 'terminal' : ''} ${isDanger ? 'danger' : ''}`}
                      style={{ backgroundColor: bgColor, border: onPath ? '2px solid #fff' : undefined }}
                    >
                      <span className="cell-id">{isDanger ? '💀' : sIdx}</span>
                      {isTerminal ? (
                        <span style={{fontSize: '0.9rem'}}>🏁</span>
                      ) : sarsaEpisode > 0 ? (
                        <div style={{color: '#fff', fontWeight: 'bold', fontSize: '1rem'}}>
                          {actionArrow(bestA)}
                        </div>
                      ) : (
                        <span className="cell-value" style={{opacity: 0.5}}>?</span>
                      )}
                      {!isTerminal && <span style={{fontSize: '0.5rem', position: 'absolute', bottom: 2, color: '#aaa'}}>{maxQ.toFixed(1)}</span>}
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          {/* Q-LEARNING */}
          <div style={{flex: 1, minWidth: '400px'}}>
            <h2 style={{color: '#ffb74d'}}>Q-Learning (Off-Policy)</h2>
            <div className="equation" style={{border: '1px solid #ff9800', fontSize: '1rem'}}>
              Q(s,a) ← Q(s,a) + α[r + γ max<sub>a'</sub> Q(s',a') - Q(s,a)]
            </div>
            <p style={{fontSize: '0.85rem'}}>Updates Q using the <strong>best possible action</strong> (max) in the next state. Learns optimal policy regardless of behavior. <span style={{color: '#888', fontSize: '0.75rem'}}>Convergence: max|ΔQ| &lt; 0.01 after 50+ episodes</span></p>

            <div className="controls" style={{flexWrap: 'wrap'}}>
              <button onClick={() => setQlPlaying(!qlPlaying)} disabled={qlDone} style={{borderColor: '#ff9800'}}>
                {qlPlaying ? 'Pause' : 'Start Q-Learning'}
              </button>
              <button onClick={runQlEpisode} disabled={qlPlaying || qlDone}>Run Episode</button>
              <span style={{fontSize: '0.8rem'}}>Ep: {qlEpisode} | R: {qlTotalReward.toFixed(1)}</span>
              {qlDone && <span style={{ color: '#4caf50', fontWeight: 'bold', fontSize: '0.8rem' }}> CONVERGED!</span>}
            </div>

            <div style={{display: 'flex', justifyContent: 'center'}}>
              <div className="grid-container" style={{borderColor: '#ff9800'}}>
                {qlQTable.map((stateQ, sIdx) => {
                  const isTerminal = sIdx === GRID_SIZE * GRID_SIZE - 1;
                  const isDanger = DANGER_STATES.has(sIdx);
                  const maxQ = Math.max(...ACTIONS.map(a => stateQ[a]));
                  const bestA = ACTIONS.find(a => stateQ[a] === maxQ) || 'UP';
                  const onPath = qlLastPath.includes(sIdx);
                  let bgColor: string | undefined;
                  if (!isTerminal) {
                    if (maxQ >= 0) bgColor = `rgba(255, 183, 77, ${Math.min(maxQ / 10, 0.8)})`;
                    else bgColor = `rgba(244, 67, 54, ${Math.min(Math.abs(maxQ) / 5, 0.6)})`;
                  }
                  return (
                    <div
                      key={sIdx}
                      className={`cell ${isTerminal ? 'terminal' : ''} ${isDanger ? 'danger' : ''}`}
                      style={{ backgroundColor: bgColor, border: onPath ? '2px solid #fff' : undefined }}
                    >
                      <span className="cell-id">{isDanger ? '💀' : sIdx}</span>
                      {isTerminal ? (
                        <span style={{fontSize: '0.9rem'}}>🏁</span>
                      ) : qlEpisode > 0 ? (
                        <div style={{color: '#fff', fontWeight: 'bold', fontSize: '1rem'}}>
                          {actionArrow(bestA)}
                        </div>
                      ) : (
                        <span className="cell-value" style={{opacity: 0.5}}>?</span>
                      )}
                      {!isTerminal && <span style={{fontSize: '0.5rem', position: 'absolute', bottom: 2, color: '#aaa'}}>{maxQ.toFixed(1)}</span>}
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        </div>

        {/* TD Q-Tables side by side */}
        {(sarsaEpisode > 0 || qlEpisode > 0) && (
          <div className="policy-tables" style={{marginTop: '1.5rem'}}>
            {sarsaEpisode > 0 && (
              <div>
                <h4>SARSA Q-Table<br/><span style={{fontSize: '0.8em', opacity: 0.7}}>α={ALPHA} | ε={EPSILON} | γ={CONFIG.gamma}</span></h4>
                <table>
                  <thead>
                    <tr>
                      <th>State</th>
                      <th>Q(s,↑)</th>
                      <th>Q(s,↓)</th>
                      <th>Q(s,←)</th>
                      <th>Q(s,→)</th>
                      <th style={{backgroundColor: '#2a1a0a'}}>Best</th>
                    </tr>
                  </thead>
                  <tbody>
                    {sarsaQTable.map((stateQ, sIdx) => {
                      const maxQ = Math.max(...ACTIONS.map(a => stateQ[a]));
                      return (
                        <tr key={sIdx}>
                          <td>{sIdx}</td>
                          {ACTIONS.map(a => (
                            <td key={a} style={{color: stateQ[a] === maxQ && (maxQ !== 0 || ACTIONS.some(act => stateQ[act] !== 0)) ? '#ff8a65' : '#888', fontWeight: stateQ[a] === maxQ && (maxQ !== 0 || ACTIONS.some(act => stateQ[act] !== 0)) ? 'bold' : 'normal'}}>
                              {stateQ[a].toFixed(2)}
                            </td>
                          ))}
                          <td style={{color: '#ff8a65', fontWeight: 'bold'}}>
                            {(maxQ !== 0 || ACTIONS.some(a => stateQ[a] !== 0)) ? actionArrow(ACTIONS.find(a => stateQ[a] === maxQ) as Action) : '-'}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
            {qlEpisode > 0 && (
              <div>
                <h4>Q-Learning Q-Table<br/><span style={{fontSize: '0.8em', opacity: 0.7}}>α={ALPHA} | ε={EPSILON} | γ={CONFIG.gamma}</span></h4>
                <table>
                  <thead>
                    <tr>
                      <th>State</th>
                      <th>Q(s,↑)</th>
                      <th>Q(s,↓)</th>
                      <th>Q(s,←)</th>
                      <th>Q(s,→)</th>
                      <th style={{backgroundColor: '#2a1a0a'}}>Best</th>
                    </tr>
                  </thead>
                  <tbody>
                    {qlQTable.map((stateQ, sIdx) => {
                      const maxQ = Math.max(...ACTIONS.map(a => stateQ[a]));
                      return (
                        <tr key={sIdx}>
                          <td>{sIdx}</td>
                          {ACTIONS.map(a => (
                            <td key={a} style={{color: stateQ[a] === maxQ && (maxQ !== 0 || ACTIONS.some(act => stateQ[act] !== 0)) ? '#ffb74d' : '#888', fontWeight: stateQ[a] === maxQ && (maxQ !== 0 || ACTIONS.some(act => stateQ[act] !== 0)) ? 'bold' : 'normal'}}>
                              {stateQ[a].toFixed(2)}
                            </td>
                          ))}
                          <td style={{color: '#ffb74d', fontWeight: 'bold'}}>
                            {(maxQ !== 0 || ACTIONS.some(a => stateQ[a] !== 0)) ? actionArrow(ACTIONS.find(a => stateQ[a] === maxQ) as Action) : '-'}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}

        <div style={{marginTop: '1.5rem', textAlign: 'center', fontSize: '0.85rem', color: '#ccc', maxWidth: '700px', margin: '1.5rem auto 0'}}>
          <strong style={{color: '#ff9800'}}>SARSA vs Q-Learning:</strong> SARSA uses the actual next action (a') — it learns the value of the ε-greedy policy it follows (safer, avoids danger). Q-Learning uses max — it learns the optimal policy regardless of exploration (bolder, may get closer to danger during learning).
        </div>
      </section>

      {/* ALL METHODS COMPARISON */}
      {(mcEpisode > 20 || sarsaEpisode > 20 || qlEpisode > 20) && (
        <section className="phase-container" style={{border: '2px solid #fff', padding: '20px', borderRadius: '12px', background: '#0a0a0a'}}>
          <h2 style={{color: '#fff'}}>All Methods — Final Policies Compared</h2>
          <p style={{fontSize: '0.85rem', color: '#aaa'}}>After sufficient learning, all methods should converge to similar optimal policies for this deterministic environment.</p>

          <div className="dashboard">
            <div className="policy-view">
              <h3 style={{color: '#ce93d8'}}>Value Iteration<br/><span style={{fontSize: '0.7em', opacity: 0.7}}>(DP, model-based)</span></h3>
              <div className="grid-container" style={{borderColor: '#9c27b0'}}>
                {viValues.map((_, sIdx) => {
                  const isTerminal = sIdx === GRID_SIZE * GRID_SIZE - 1;
                  const isDanger = DANGER_STATES.has(sIdx);
                  const bestA = viPolicy ? viPolicy[sIdx] : null;
                  return (
                    <div key={sIdx} className={`cell ${isTerminal ? 'terminal' : ''} ${isDanger ? 'danger' : ''}`}>
                      <span className="cell-id">{isDanger ? '💀' : sIdx}</span>
                      {isTerminal ? <span>🏁</span> : bestA && bestA !== 'NONE' ? (
                        <span style={{fontWeight: 'bold', fontSize: '1rem'}}>{actionArrow(bestA as Action)}</span>
                      ) : <span style={{opacity: 0.5}}>?</span>}
                    </div>
                  );
                })}
              </div>
            </div>

            <div className="policy-view">
              <h3 style={{color: '#4dd0e1'}}>Monte Carlo<br/><span style={{fontSize: '0.7em', opacity: 0.7}}>(model-free, ep {mcEpisode})</span></h3>
              <div className="grid-container" style={{borderColor: '#00bcd4'}}>
                {mcQTable.map((stateQ, sIdx) => {
                  const isTerminal = sIdx === GRID_SIZE * GRID_SIZE - 1;
                  const isDanger = DANGER_STATES.has(sIdx);
                  const maxQ = Math.max(...ACTIONS.map(a => stateQ[a]));
                  const bestA = ACTIONS.find(a => stateQ[a] === maxQ) || 'UP';
                  return (
                    <div key={sIdx} className={`cell ${isTerminal ? 'terminal' : ''} ${isDanger ? 'danger' : ''}`}>
                      <span className="cell-id">{isDanger ? '💀' : sIdx}</span>
                      {isTerminal ? <span>🏁</span> : mcEpisode > 0 ? (
                        <span style={{fontWeight: 'bold', fontSize: '1rem'}}>{actionArrow(bestA)}</span>
                      ) : <span style={{opacity: 0.5}}>?</span>}
                    </div>
                  );
                })}
              </div>
            </div>

            <div className="policy-view">
              <h3 style={{color: '#ff8a65'}}>SARSA<br/><span style={{fontSize: '0.7em', opacity: 0.7}}>(on-policy TD, ep {sarsaEpisode})</span></h3>
              <div className="grid-container" style={{borderColor: '#ff5722'}}>
                {sarsaQTable.map((stateQ, sIdx) => {
                  const isTerminal = sIdx === GRID_SIZE * GRID_SIZE - 1;
                  const isDanger = DANGER_STATES.has(sIdx);
                  const maxQ = Math.max(...ACTIONS.map(a => stateQ[a]));
                  const bestA = ACTIONS.find(a => stateQ[a] === maxQ) || 'UP';
                  return (
                    <div key={sIdx} className={`cell ${isTerminal ? 'terminal' : ''} ${isDanger ? 'danger' : ''}`}>
                      <span className="cell-id">{isDanger ? '💀' : sIdx}</span>
                      {isTerminal ? <span>🏁</span> : sarsaEpisode > 0 ? (
                        <span style={{fontWeight: 'bold', fontSize: '1rem'}}>{actionArrow(bestA)}</span>
                      ) : <span style={{opacity: 0.5}}>?</span>}
                    </div>
                  );
                })}
              </div>
            </div>

            <div className="policy-view">
              <h3 style={{color: '#ffb74d'}}>Q-Learning<br/><span style={{fontSize: '0.7em', opacity: 0.7}}>(off-policy TD, ep {qlEpisode})</span></h3>
              <div className="grid-container" style={{borderColor: '#ff9800'}}>
                {qlQTable.map((stateQ, sIdx) => {
                  const isTerminal = sIdx === GRID_SIZE * GRID_SIZE - 1;
                  const isDanger = DANGER_STATES.has(sIdx);
                  const maxQ = Math.max(...ACTIONS.map(a => stateQ[a]));
                  const bestA = ACTIONS.find(a => stateQ[a] === maxQ) || 'UP';
                  return (
                    <div key={sIdx} className={`cell ${isTerminal ? 'terminal' : ''} ${isDanger ? 'danger' : ''}`}>
                      <span className="cell-id">{isDanger ? '💀' : sIdx}</span>
                      {isTerminal ? <span>🏁</span> : qlEpisode > 0 ? (
                        <span style={{fontWeight: 'bold', fontSize: '1rem'}}>{actionArrow(bestA)}</span>
                      ) : <span style={{opacity: 0.5}}>?</span>}
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          <div style={{marginTop: '1.5rem'}}>
            <table style={{margin: '0 auto', fontSize: '0.8rem'}}>
              <thead>
                <tr>
                  <th>Property</th>
                  <th style={{color: '#ce93d8'}}>DP (Value Iter)</th>
                  <th style={{color: '#4dd0e1'}}>Monte Carlo</th>
                  <th style={{color: '#ff8a65'}}>SARSA</th>
                  <th style={{color: '#ffb74d'}}>Q-Learning</th>
                </tr>
              </thead>
              <tbody>
                <tr><td>Needs model?</td><td>Yes</td><td>No</td><td>No</td><td>No</td></tr>
                <tr><td>Update timing</td><td>Every sweep</td><td>End of episode</td><td>Every step</td><td>Every step</td></tr>
                <tr><td>Bootstrap?</td><td>Yes</td><td>No</td><td>Yes</td><td>Yes</td></tr>
                <tr><td>On/Off policy</td><td>N/A</td><td>On-policy</td><td>On-policy</td><td>Off-policy</td></tr>
                <tr><td>Update target</td><td>max Q</td><td>G (full return)</td><td>Q(s',a')</td><td>max Q(s',a')</td></tr>
                <tr><td>Convergence</td><td>Fast (exact)</td><td>Slow (variance)</td><td>Medium</td><td>Medium</td></tr>
              </tbody>
            </table>
          </div>
        </section>
      )}
    </div>
  );
};

export default App;
