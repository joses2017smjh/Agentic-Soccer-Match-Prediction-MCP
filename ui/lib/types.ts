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

export interface ScorelineGrid {
  max_goals: number;
  probs: number[][]; // probs[home][away], 6x6
  tail_mass: number;
}

export interface ExactScore {
  top_scorelines: Scoreline[];
  over_under_2_5: { over: number; under: number };
  btts: { yes: number; no: number };
  grid_outcome_probs: { home: number; draw: number; away: number };
  scoreline_grid?: ScorelineGrid;
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

// ---- leagues hub ----

export interface LeagueRef {
  id: string;
  name: string;
  country: string;
}

export interface LeagueDirectory {
  regions: { region: string; leagues: LeagueRef[] }[];
  tournaments: { id: string; name: string; type: string; endpoint?: string }[];
}

export interface StandingRow {
  rank: number;
  team: string;
  played: number;
  won: number;
  drawn: number;
  lost: number;
  gf: number;
  ga: number;
  gd: number;
  points: number;
}

export interface MatchResult {
  date: string;
  home_team: string;
  away_team: string;
  home_score: number;
  away_score: number;
}

export interface UpcomingFixture {
  date: string;
  time: string;
  home_team: string;
  away_team: string;
  odds_home?: number;
  odds_draw?: number;
  odds_away?: number;
}

export interface LeagueDetail {
  id: string;
  name: string;
  region: string;
  country: string;
  season: string;
  standings: StandingRow[];
  elo: Record<string, number>;
  recent_results: MatchResult[];
  upcoming_fixtures: UpcomingFixture[];
  fixtures_source?: string;
  teams: string[];
  note: string;
}

// ---- bracket + matchup tie (shared shape from the Elo engine) ----

export interface TieGoal {
  minute: number;
  team: "home" | "away";
  scorer_role: string;
  assist_role: string | null;
}

export interface Tie {
  home: string;
  away: string;
  seeds?: { home: number; away: number };
  expected_goals: { home: number; away: number };
  outcome_90: { home: number; draw: number; away: number };
  advance: Record<string, number>;
  projected_winner: string;
  top_scorelines: { score: string; prob: number }[];
  first_scorer: { home_first: number; away_first: number; no_goals: number };
  headline_scenario: {
    scoreline: string;
    probability: number;
    goals: TieGoal[];
    penalties: { winner: string; p_advance: number } | null;
    player_data?: "statsbomb" | "mixed" | "role-level";
    note: string;
  };
  evidence: {
    home_rating: { elo: number; matches: number; seed: number; last_played: string };
    away_rating: { elo: number; matches: number; seed: number; last_played: string };
    elo_difference: number;
    model_xg: { home: number; away: number };
    dixon_coles_rho: number;
    method: string;
  };
}

export interface Bracket {
  tournament: string;
  disclaimer: string;
  seeding: { rank: number; team: string; elo: number; matches: number }[];
  rounds: { round: string; matches: Tie[] }[];
  champion: string;
  model: Record<string, number | string>;
}

// ---- correlated parlay pricer ----

export interface ParlayLegInput {
  match_id: string;
  leg_type: string;
  selection: string;
  line?: number;
  decimal_odds?: number;
  team_side?: string;
  xg_share?: number;
}

export interface ParlayLegResult {
  match_id: string;
  leg_type: string;
  selection: string;
  marginal_prob: number;
  decimal_odds: number;
  implied_prob: number;
}

export interface ParlayMatchGroup {
  match_id: string;
  joint_prob: number;
  independent_prob: number;
  correlation_factor: number;
  mu_home: number;
  mu_away: number;
  n_legs: number;
}

export interface ParlayResult {
  legs: ParlayLegResult[];
  match_groups: ParlayMatchGroup[];
  joint_probability: number;
  independent_probability: number;
  correlation_factor: number;
  model_fair_odds: number;
  sportsbook_parlay_odds: number;
  edge: number;
  ev: number;
  kelly_fraction: number;
  correlation_direction: string;
  analysis: string;
}

// ---- chat / conversational layer ----

export interface ChatAction {
  type: "navigate";
  route: string;
}

export interface ChatResponse {
  reply: string;
  action?: ChatAction;
  suggested_input?: string;
  intent: string;
}
