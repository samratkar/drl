export type Action = 'UP' | 'DOWN' | 'LEFT' | 'RIGHT';

export interface State {
  id: number;
  row: number;
  col: number;
  isTerminal: boolean;
}

export type Policy = Record<number, Record<Action, number>>;

export interface MDPConfig {
  gridSize: number;
  gamma: number;
  rewardGoal: number;
  rewardStep: number;
}
