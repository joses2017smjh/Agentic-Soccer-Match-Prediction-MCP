/** Mirrors the gateway's prediction JSON (src/models/predict.py). */

export interface MatchOutcome {
  home: number;
  draw: number;
  away: number;
  conformal_set: ("home" | "draw" | "away")[];
  conformal_alpha: number;
}

export interface Scoreline {
  score: string; // "1-0"
  prob: number;
}

export interface ExactScore {
  top_scorelines: Scoreline[];
  over_under_2_5: { over: number; under: number };
  btts: { yes: number; no: number };
  grid_outcome_probs: { home: number; draw: number; away: number };
}

export interface FirstScorer {
  home_first: number;
  away_first: number;
  no_goals: number;
}

export interface GoalBand {
  band: string; // "0-15"
  home: number;
  away: number;
}

export interface EventSequence {
  first_scorer: FirstScorer;
  goals_by_band: GoalBand[];
  next_goal_from_kickoff: { home: number; away: number; no_more_goals: number };
}

export interface ScenarioGoal {
  minute: number;
  team: "home" | "away";
  scorer?: string;
  assist?: string;
  p_scorer_anytime?: number;
}

export interface HeadlineScenario {
  scoreline: string;
  probability: number;
  goals: ScenarioGoal[];
  penalties?: { winner: "home" | "away"; p_advance: number };
  player_of_the_match?: string;
}

export interface PlayerProp {
  player: string;
  p_anytime_scorer: number;
  p_assist: number;
  goal_lambda: number;
  assist_lambda: number;
}

export interface Suggestion {
  market: string;
  selection: string;
  edge: number;
  ev: number;
  kelly_stake: number;
  tier: string;
  rationale: string;
}

export interface Prediction {
  match_id: string;
  model_version: string;
  match_outcome: MatchOutcome;
  expected_goals: { home: number; away: number };
  exact_score: ExactScore;
  event_sequence: EventSequence;
  headline_scenario?: HeadlineScenario;
  knockout?: { advance: { home: number; away: number } };
  player_props?: { home?: PlayerProp[]; away?: PlayerProp[] };
  market_comparison?: unknown[];
  suggestions?: Suggestion[];
  as_of: string;
}

export interface ToolCallSummary {
  server: string;
  tool: string;
  ok: boolean;
  latency_ms: number;
  error?: string;
}

export type PredictResponse =
  | {
      status: "complete";
      thread_id: string;
      answer: string;
      prediction: Prediction | null;
      degraded: string[];
      tool_calls: ToolCallSummary[];
    }
  | {
      status: "pending_approval";
      thread_id: string;
      approval_request: { suggestions: Suggestion[]; match_id: string };
    };

export interface Health {
  ok: boolean;
  model_version: string | null;
}
