export interface WordProbability {
  word: string;
  prob: number;
}

export interface LineageEntry {
  gen: number;
  id: string;
  origin: string;
  parent_ids: string[];
}

export interface Agent {
  id: string;
  typed_text: string;
  score: number;
  scores_breakdown: {
    vocabulary_score?: number;
    grammar_score?: number;
    coherence_score?: number;
    total?: number;
    vocab_bonus?: number;
    known_words_used?: string[];
    reason?: string;  // OpenAI's explanation for the score
  };
  exploration_rate: number;
  vocabulary?: string[];
  vocabulary_size?: number;
  // Lineage tracking (Word Genome only)
  origin?: 'elite' | 'crossover' | 'mutation' | 'random';
  parent_ids?: string[];
  generation_born?: number;
  lineage?: LineageEntry[];
  top_words?: WordProbability[];
  model_type?: string;
}

export interface ConversationTurn {
  agent_text: string;
  openai_response: string;
}

export interface GenerationStats {
  generation: number;
  // Old field names (from evolution)
  best_score?: number;
  avg_score?: number;
  worst_score?: number;
  best_text?: string;
  best_agent_id?: string | null;
  avg_exploration?: number;
  // New field names (from neural network)
  max?: number;
  mean?: number;
  min?: number;
  exploration_rate?: number;
}

export interface BestOutput {
  generation: number;
  conversation?: ConversationTurn[];
  text?: string;
  score: number;
  scores_breakdown?: {
    vocabulary_score: number;
    grammar_score: number;
    coherence_score: number;
  };
}

export interface Milestones {
  first_word: boolean;
  first_sentence: boolean;
  first_coherent: boolean;
  sustained_conversation: boolean;
  achieved: Array<{
    type: string;
    text?: string[];
  }>;
}

export interface Config {
  num_agents: number;
  generation_time: number;
  turns_per_conversation: number;
  mutation_rate: number;
  exploration_decay: number;
}

export interface DiscoveredSentence {
  text: string;
  score: number;
  coherence: number;
  agent_id: string;
  generation: number;
  model_type: string;
  timestamp: number;
}

export interface PatternDiscoveries {
  bigrams: string[];
  trigrams: string[];
  sentences: DiscoveredSentence[];
}
