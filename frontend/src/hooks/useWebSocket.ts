import { useEffect, useRef, useState, useCallback } from 'react';
import type { Agent, GenerationStats, BestOutput, Milestones, ConversationTurn, DiscoveredSentence } from '../types';

export type ModelType = 'genome' | 'char_nn' | 'word_nn' | 'word_genome' | 'concurrent';

// Concurrent mode state for a single model
export interface ConcurrentModelState {
  generation: number;
  agents: Agent[];
  history: GenerationStats[];
  bestOutputs: BestOutput[];
  milestones: Milestones;
  isRunning: boolean;
  conversations: Record<string, ConversationTurn[]>;
}

// Running status for all models
export interface RunningStatus {
  genome: boolean;
  char_nn: boolean;
  word_nn: boolean;
  word_genome: boolean;
}

// Word Genome phases
export type WordGenomePhase = 'STRUCTURE' | 'PATTERNS' | 'SENTENCES' | 'CONVERSATION';

export interface WSMessage {
  type: 'init' | 'generation_start' | 'turn_start' | 'keystroke' | 'keystroke_update' | 'turn_end' | 'generation_end' | 'reset' | 'ping' | 'pong' | 'milestone' | 'model_switch' | 'phase_transition' | 'gpt_first';
  mode?: 'concurrent';  // Present when in concurrent mode
  model_type?: ModelType;  // Which model this message is for
  generation?: number;
  is_running?: boolean;
  running_status?: RunningStatus;  // Running status for all models
  agents?: Agent[];
  history?: GenerationStats[];
  best_outputs?: BestOutput[];
  milestones?: Milestones;
  stats?: GenerationStats;
  step?: number;
  turn?: number;
  total_turns?: number;
  agent_id?: string;
  char?: string;
  text?: string;
  updates?: Array<{
    id?: string;
    agent_id?: string;
    agent_text?: string;
    output?: string;
    openai_response?: string;
    turn?: number;
  }>;
  conversations?: Record<string, ConversationTurn[]>;
  global_vocabulary?: string[];
  current_model?: ModelType;
  // Pattern discoveries
  discovered_bigrams?: string[];
  discovered_trigrams?: string[];
  discovered_sentences?: DiscoveredSentence[];
  // Word Genome phase tracking
  word_genome_phase?: WordGenomePhase;
  word_genome_phase_progress?: number;
  word_genome_phase_number?: number;
  // Phase transition
  old_phase?: string;
  new_phase?: string;
  // GPT first message (Phase 4)
  message?: string;
  // Concurrent mode nested data
  char_nn?: {
    generation?: number;
    agents?: Agent[];
    history?: GenerationStats[];
    best_outputs?: BestOutput[];
    milestones?: Milestones;
    stats?: GenerationStats;
    updates?: Array<{ id: string; output: string; score: number }>;
    is_running?: boolean;
  };
  word_nn?: {
    generation?: number;
    agents?: Agent[];
    history?: GenerationStats[];
    best_outputs?: BestOutput[];
    milestones?: Milestones;
    stats?: GenerationStats;
    updates?: Array<{ id: string; output: string; score: number }>;
    is_running?: boolean;
  };
}

export interface WebSocketState {
  agents: Agent[];
  generation: number;
  isRunning: boolean;  // Current view's running state
  runningStatus: RunningStatus;  // Per-model running states
  history: GenerationStats[];
  bestOutputs: BestOutput[];
  isConnected: boolean;
  currentTurn: number;
  totalTurns: number;
  milestones: Milestones;
  conversations: Record<string, ConversationTurn[]>;
  latestMilestone: string | null;
  globalVocabulary: string[];
  currentModel: ModelType;
  // Per-model state (for concurrent and switching)
  charNN: ConcurrentModelState;
  wordNN: ConcurrentModelState;
  genome: ConcurrentModelState;
  // Pattern discoveries (shared across models)
  discoveredBigrams: string[];
  discoveredTrigrams: string[];
  discoveredSentences: DiscoveredSentence[];
  // Word Genome phase tracking
  wordGenomePhase: WordGenomePhase;
  wordGenomePhaseProgress: number;
  gptFirstMessage: string;
}

// Use environment variable for backend URL, fallback to localhost for development
const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || 'localhost:8000';
const WS_PROTOCOL = BACKEND_URL.includes('localhost') ? 'ws' : 'wss';
const WS_URL = `${WS_PROTOCOL}://${BACKEND_URL}/ws`;

export function useWebSocket() {
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  
  const defaultMilestones: Milestones = {
    first_word: false,
    first_sentence: false,
    first_coherent: false,
    sustained_conversation: false,
    achieved: []
  };

  const defaultConcurrentState: ConcurrentModelState = {
    generation: 0,
    agents: [],
    history: [],
    bestOutputs: [],
    milestones: defaultMilestones,
    isRunning: false,
    conversations: {}
  };

  const defaultRunningStatus: RunningStatus = {
    genome: false,
    char_nn: false,
    word_nn: false,
    word_genome: false
  };

  const [state, setState] = useState<WebSocketState>({
    agents: [],
    generation: 0,
    isRunning: false,
    runningStatus: defaultRunningStatus,
    history: [],
    bestOutputs: [],
    isConnected: false,
    currentTurn: 0,
    totalTurns: 3,
    milestones: defaultMilestones,
    conversations: {},
    latestMilestone: null,
    globalVocabulary: [],
    currentModel: 'char_nn',
    charNN: defaultConcurrentState,
    wordNN: defaultConcurrentState,
    genome: defaultConcurrentState,
    // Pattern discoveries
    discoveredBigrams: [],
    discoveredTrigrams: [],
    discoveredSentences: [],
    // Word Genome phase tracking
    wordGenomePhase: 'STRUCTURE',
    wordGenomePhaseProgress: 0,
    gptFirstMessage: ''
  });

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onopen = () => {
      console.log('WebSocket connected');
      setState(s => ({ ...s, isConnected: true }));
    };

    ws.onclose = () => {
      console.log('WebSocket disconnected');
      setState(s => ({ ...s, isConnected: false }));
      
      reconnectTimeoutRef.current = setTimeout(() => {
        connect();
      }, 2000);
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    ws.onmessage = (event) => {
      try {
        const msg: WSMessage = JSON.parse(event.data);
        
        switch (msg.type) {
          case 'init':
            if (msg.mode === 'concurrent') {
              // Concurrent mode init
              setState(s => ({
                ...s,
                isRunning: msg.is_running ?? false,
                runningStatus: msg.running_status ?? s.runningStatus,
                totalTurns: msg.total_turns ?? s.totalTurns,
                globalVocabulary: msg.global_vocabulary ?? s.globalVocabulary,
                currentModel: 'concurrent',
                // Pattern discoveries
                discoveredBigrams: msg.discovered_bigrams ?? s.discoveredBigrams,
                discoveredTrigrams: msg.discovered_trigrams ?? s.discoveredTrigrams,
                discoveredSentences: msg.discovered_sentences ?? s.discoveredSentences,
                charNN: {
                  ...s.charNN,
                  generation: msg.char_nn?.generation ?? 0,
                  agents: msg.char_nn?.agents ?? [],
                  history: msg.char_nn?.history ?? [],
                  bestOutputs: msg.char_nn?.best_outputs ?? [],
                  milestones: msg.char_nn?.milestones ?? s.charNN.milestones,
                  isRunning: msg.char_nn?.is_running ?? s.charNN.isRunning
                },
                wordNN: {
                  ...s.wordNN,
                  generation: msg.word_nn?.generation ?? 0,
                  agents: msg.word_nn?.agents ?? [],
                  history: msg.word_nn?.history ?? [],
                  bestOutputs: msg.word_nn?.best_outputs ?? [],
                  milestones: msg.word_nn?.milestones ?? s.wordNN.milestones,
                  isRunning: msg.word_nn?.is_running ?? s.wordNN.isRunning
                }
              }));
            } else {
              // Single model init
              const modelType = msg.current_model ?? s.currentModel;
              setState(s => ({
                ...s,
                generation: msg.generation ?? 0,
                isRunning: msg.is_running ?? false,
                runningStatus: msg.running_status ?? s.runningStatus,
                agents: msg.agents ?? [],
                history: msg.history ?? [],
                bestOutputs: msg.best_outputs ?? [],
                milestones: msg.milestones ?? s.milestones,
                totalTurns: msg.total_turns ?? s.totalTurns,
                globalVocabulary: msg.global_vocabulary ?? s.globalVocabulary,
                currentModel: modelType,
                // Pattern discoveries
                discoveredBigrams: msg.discovered_bigrams ?? s.discoveredBigrams,
                discoveredTrigrams: msg.discovered_trigrams ?? s.discoveredTrigrams,
                discoveredSentences: msg.discovered_sentences ?? s.discoveredSentences,
                // Word Genome phase
                wordGenomePhase: msg.word_genome_phase ?? s.wordGenomePhase,
                wordGenomePhaseProgress: msg.word_genome_phase_progress ?? s.wordGenomePhaseProgress,
              }));
            }
            break;
            
          case 'generation_start':
            if (msg.mode === 'concurrent') {
              setState(s => ({
                ...s,
                isRunning: true,
                currentTurn: 0,
                conversations: {},
                charNN: {
                  ...s.charNN,
                  generation: msg.char_nn?.generation ?? s.charNN.generation,
                  agents: msg.char_nn?.agents ?? s.charNN.agents,
                  isRunning: true,
                  conversations: {}
                },
                wordNN: {
                  ...s.wordNN,
                  generation: msg.word_nn?.generation ?? s.wordNN.generation,
                  agents: msg.word_nn?.agents ?? s.wordNN.agents,
                  isRunning: true,
                  conversations: {}
                }
              }));
            } else {
              // Route by model_type for independent training
              const modelType = msg.model_type ?? s.currentModel;
              
              if (modelType === 'char_nn') {
                setState(s => ({
                  ...s,
                  runningStatus: msg.running_status ?? { ...s.runningStatus, char_nn: true },
                  charNN: {
                    ...s.charNN,
                    generation: msg.generation ?? s.charNN.generation,
                    agents: msg.agents ?? s.charNN.agents,
                    isRunning: true,
                    conversations: {}
                  },
                  // Update main state if this is the current view
                  ...(s.currentModel === 'char_nn' ? {
                    generation: msg.generation ?? s.generation,
                    agents: msg.agents ?? s.agents,
                    isRunning: true,
                    currentTurn: 0,
                    conversations: {}
                  } : {})
                }));
              } else if (modelType === 'word_nn') {
                setState(s => ({
                  ...s,
                  runningStatus: msg.running_status ?? { ...s.runningStatus, word_nn: true },
                  wordNN: {
                    ...s.wordNN,
                    generation: msg.generation ?? s.wordNN.generation,
                    agents: msg.agents ?? s.wordNN.agents,
                    isRunning: true,
                    conversations: {}
                  },
                  // Update main state if this is the current view
                  ...(s.currentModel === 'word_nn' ? {
                    generation: msg.generation ?? s.generation,
                    agents: msg.agents ?? s.agents,
                    isRunning: true,
                    currentTurn: 0,
                    conversations: {}
                  } : {})
                }));
              } else if (modelType === 'genome') {
                setState(s => ({
                  ...s,
                  runningStatus: msg.running_status ?? { ...s.runningStatus, genome: true },
                  genome: {
                    ...s.genome,
                    generation: msg.generation ?? s.genome.generation,
                    agents: msg.agents ?? s.genome.agents,
                    isRunning: true,
                    conversations: {}
                  },
                  // Update main state if this is the current view
                  ...(s.currentModel === 'genome' ? {
                    generation: msg.generation ?? s.generation,
                    agents: msg.agents ?? s.agents,
                    isRunning: true,
                    currentTurn: 0,
                    conversations: {}
                  } : {})
                }));
              } else if (modelType === 'word_genome') {
                setState(s => ({
                  ...s,
                  runningStatus: msg.running_status ?? { ...s.runningStatus, word_genome: true },
                  // Word Genome phase tracking
                  wordGenomePhase: msg.word_genome_phase ?? s.wordGenomePhase,
                  wordGenomePhaseProgress: msg.word_genome_phase_progress ?? s.wordGenomePhaseProgress,
                  // Update main state if this is the current view
                  ...(s.currentModel === 'word_genome' ? {
                    generation: msg.generation ?? s.generation,
                    agents: msg.agents ?? s.agents,
                    isRunning: true,
                    currentTurn: 0,
                    conversations: {}
                  } : {})
                }));
              } else {
                // Fallback: update main state
                setState(s => ({
                  ...s,
                  generation: msg.generation ?? s.generation,
                  agents: msg.agents ?? s.agents,
                  isRunning: true,
                  currentTurn: 0,
                  conversations: {}
                }));
              }
            }
            break;
          
          case 'turn_start':
            setState(s => ({
              ...s,
              currentTurn: msg.turn ?? s.currentTurn,
              totalTurns: msg.total_turns ?? s.totalTurns,
            }));
            break;
            
          case 'keystroke':
            // Update individual agent text during typing
            if (msg.agent_id) {
              setState(s => {
                const newAgents = s.agents.map(agent => {
                  if (agent.id === msg.agent_id) {
                    return { ...agent, typed_text: msg.text ?? agent.typed_text };
                  }
                  return agent;
                });
                return { ...s, agents: newAgents };
              });
            }
            break;
          
          case 'keystroke_update':
            if (msg.mode === 'concurrent') {
              // Concurrent mode: update both models
              setState(s => {
                const charUpdates = msg.char_nn?.updates || [];
                const wordUpdates = msg.word_nn?.updates || [];
                
                const newCharAgents = s.charNN.agents.map(agent => {
                  const update = charUpdates.find((u) => u.id === agent.id);
                  if (update) {
                    return { ...agent, typed_text: update.output ?? agent.typed_text };
                  }
                  return agent;
                });
                
                const newWordAgents = s.wordNN.agents.map(agent => {
                  const update = wordUpdates.find((u) => u.id === agent.id);
                  if (update) {
                    return { ...agent, typed_text: update.output ?? agent.typed_text };
                  }
                  return agent;
                });
                
                return {
                  ...s,
                  charNN: { ...s.charNN, agents: newCharAgents },
                  wordNN: { ...s.wordNN, agents: newWordAgents }
                };
              });
            } else if (msg.updates) {
              // Route by model_type for independent training
              const modelType = msg.model_type ?? s.currentModel;
              
              setState(s => {
                const newAgents = (msg.updates as any[])?.map((u: any) => u) || [];
                
                // Helper to update agents
                const updateAgents = (agents: Agent[]) => 
                  agents.map(agent => {
                    const update = newAgents.find((u: any) => u.id === agent.id);
                    if (update) {
                      return { ...agent, typed_text: update.output ?? agent.typed_text };
                    }
                    return agent;
                  });
                
                const result: Partial<WebSocketState> = { currentTurn: msg.turn ?? s.currentTurn };
                
                if (modelType === 'char_nn') {
                  result.charNN = { ...s.charNN, agents: updateAgents(s.charNN.agents) };
                  if (s.currentModel === 'char_nn') {
                    result.agents = updateAgents(s.agents);
                  }
                } else if (modelType === 'word_nn') {
                  result.wordNN = { ...s.wordNN, agents: updateAgents(s.wordNN.agents) };
                  if (s.currentModel === 'word_nn') {
                    result.agents = updateAgents(s.agents);
                  }
                } else if (modelType === 'genome') {
                  result.genome = { ...s.genome, agents: updateAgents(s.genome.agents) };
                  if (s.currentModel === 'genome') {
                    result.agents = updateAgents(s.agents);
                  }
                } else {
                  // Fallback: update main agents
                  result.agents = updateAgents(s.agents);
                }
                
                return { ...s, ...result };
              });
            }
            break;
          
          case 'turn_end':
            // Update conversations with turn results
            if (msg.updates) {
              setState(s => {
                const newConversations = { ...s.conversations };
                for (const update of msg.updates!) {
                  if (!newConversations[update.agent_id]) {
                    newConversations[update.agent_id] = [];
                  }
                  newConversations[update.agent_id].push({
                    agent_text: update.agent_text,
                    openai_response: update.openai_response
                  });
                }
                return { ...s, conversations: newConversations };
              });
            }
            break;
            
          case 'generation_end':
            if (msg.mode === 'concurrent') {
              setState(s => ({
                ...s,
                globalVocabulary: msg.global_vocabulary ?? s.globalVocabulary,
                charNN: {
                  ...s.charNN,
                  generation: msg.char_nn?.generation ?? s.charNN.generation,
                  agents: msg.char_nn?.agents ?? s.charNN.agents,
                  history: msg.char_nn?.stats 
                    ? [...s.charNN.history, msg.char_nn.stats] 
                    : (msg.char_nn?.history ?? s.charNN.history)
                },
                wordNN: {
                  ...s.wordNN,
                  generation: msg.word_nn?.generation ?? s.wordNN.generation,
                  agents: msg.word_nn?.agents ?? s.wordNN.agents,
                  history: msg.word_nn?.stats 
                    ? [...s.wordNN.history, msg.word_nn.stats] 
                    : (msg.word_nn?.history ?? s.wordNN.history)
                }
              }));
            } else {
              // Route by model_type for independent training
              const modelType = msg.model_type ?? s.currentModel;
              
              setState(s => {
                const result: Partial<WebSocketState> = {
                  globalVocabulary: msg.global_vocabulary ?? s.globalVocabulary,
                  // Update pattern discoveries
                  discoveredBigrams: msg.discovered_bigrams ?? s.discoveredBigrams,
                  discoveredTrigrams: msg.discovered_trigrams ?? s.discoveredTrigrams,
                  discoveredSentences: msg.discovered_sentences ?? s.discoveredSentences,
                };
                
                const newModelState = {
                  generation: msg.generation ?? 0,
                  agents: msg.agents ?? [],
                  history: msg.stats ? [...(modelType === 'char_nn' ? s.charNN.history : modelType === 'word_nn' ? s.wordNN.history : s.genome.history), msg.stats] : [],
                  bestOutputs: msg.best_outputs ?? [],
                  milestones: msg.milestones ?? defaultMilestones,
                  isRunning: true, // Still running for next generation
                  conversations: msg.conversations ?? {}
                };
                
                if (modelType === 'char_nn') {
                  result.charNN = { ...s.charNN, ...newModelState, history: msg.stats ? [...s.charNN.history, msg.stats] : s.charNN.history };
                  if (s.currentModel === 'char_nn') {
                    result.generation = msg.generation ?? s.generation;
                    result.agents = msg.agents ?? s.agents;
                    result.history = msg.stats ? [...s.history, msg.stats] : s.history;
                    result.bestOutputs = msg.best_outputs ?? s.bestOutputs;
                    result.milestones = msg.milestones ?? s.milestones;
                    result.conversations = msg.conversations ?? s.conversations;
                  }
                } else if (modelType === 'word_nn') {
                  result.wordNN = { ...s.wordNN, ...newModelState, history: msg.stats ? [...s.wordNN.history, msg.stats] : s.wordNN.history };
                  if (s.currentModel === 'word_nn') {
                    result.generation = msg.generation ?? s.generation;
                    result.agents = msg.agents ?? s.agents;
                    result.history = msg.stats ? [...s.history, msg.stats] : s.history;
                    result.bestOutputs = msg.best_outputs ?? s.bestOutputs;
                    result.milestones = msg.milestones ?? s.milestones;
                    result.conversations = msg.conversations ?? s.conversations;
                  }
                } else if (modelType === 'genome') {
                  result.genome = { ...s.genome, ...newModelState, history: msg.stats ? [...s.genome.history, msg.stats] : s.genome.history };
                  if (s.currentModel === 'genome') {
                    result.generation = msg.generation ?? s.generation;
                    result.agents = msg.agents ?? s.agents;
                    result.history = msg.stats ? [...s.history, msg.stats] : s.history;
                    result.bestOutputs = msg.best_outputs ?? s.bestOutputs;
                    result.milestones = msg.milestones ?? s.milestones;
                    result.conversations = msg.conversations ?? s.conversations;
                  }
                } else {
                  // Fallback: update main state
                  result.generation = msg.generation ?? s.generation;
                  result.agents = msg.agents ?? s.agents;
                  result.history = msg.stats ? [...s.history, msg.stats] : s.history;
                  result.bestOutputs = msg.best_outputs ?? s.bestOutputs;
                  result.milestones = msg.milestones ?? s.milestones;
                  result.conversations = msg.conversations ?? s.conversations;
                }
                
                return { ...s, ...result };
              });
            }
            break;
          
          case 'milestone':
            // Show milestone celebration
            if (msg.milestones && (msg.milestones as unknown as string[]).length > 0) {
              setState(s => ({
                ...s,
                latestMilestone: (msg.milestones as unknown as string[])[0]
              }));
              // Clear after 3 seconds
              setTimeout(() => {
                setState(s => ({ ...s, latestMilestone: null }));
              }, 3000);
            }
            break;
            
          case 'reset':
            setState(s => ({
              ...s,
              agents: msg.agents ?? [],
              generation: 0,
              history: [],
              bestOutputs: [],
              currentTurn: 0,
              conversations: {},
              milestones: msg.milestones ?? {
                first_word: false,
                first_sentence: false,
                first_coherent: false,
                sustained_conversation: false,
                achieved: []
              },
              latestMilestone: null,
              currentModel: msg.current_model ?? s.currentModel,
            }));
            break;
          
          case 'model_switch':
            if (msg.mode === 'concurrent') {
              setState(s => ({
                ...s,
                currentModel: 'concurrent',
                globalVocabulary: msg.global_vocabulary ?? s.globalVocabulary,
                runningStatus: msg.running_status ?? s.runningStatus,
                isRunning: (msg.char_nn?.is_running || msg.word_nn?.is_running) ?? false,
                currentTurn: 0,
                conversations: {},
                charNN: {
                  ...s.charNN,
                  generation: msg.char_nn?.generation ?? 0,
                  agents: msg.char_nn?.agents ?? [],
                  history: msg.char_nn?.history ?? [],
                  bestOutputs: msg.char_nn?.best_outputs ?? [],
                  milestones: msg.char_nn?.milestones ?? s.charNN.milestones,
                  isRunning: msg.char_nn?.is_running ?? s.charNN.isRunning,
                  conversations: {}
                },
                wordNN: {
                  ...s.wordNN,
                  generation: msg.word_nn?.generation ?? 0,
                  agents: msg.word_nn?.agents ?? [],
                  history: msg.word_nn?.history ?? [],
                  bestOutputs: msg.word_nn?.best_outputs ?? [],
                  milestones: msg.word_nn?.milestones ?? s.wordNN.milestones,
                  isRunning: msg.word_nn?.is_running ?? s.wordNN.isRunning,
                  conversations: {}
                }
              }));
            } else {
              const newModel = msg.current_model ?? s.currentModel;
              setState(s => ({
                ...s,
                currentModel: newModel,
                generation: msg.generation ?? s.generation,
                agents: msg.agents ?? s.agents,
                history: msg.history ?? s.history,
                bestOutputs: msg.best_outputs ?? s.bestOutputs,
                milestones: msg.milestones ?? s.milestones,
                globalVocabulary: msg.global_vocabulary ?? s.globalVocabulary,
                runningStatus: msg.running_status ?? s.runningStatus,
                isRunning: msg.is_running ?? s.runningStatus[newModel as keyof RunningStatus] ?? false,
                currentTurn: 0,
                conversations: {},
              }));
            }
            break;
          
          case 'phase_transition':
            // Word Genome phase transition
            setState(s => ({
              ...s,
              wordGenomePhase: (msg.new_phase as WordGenomePhase) ?? s.wordGenomePhase,
              wordGenomePhaseProgress: 0,
            }));
            console.log(`[PHASE] Transitioned: ${msg.old_phase} -> ${msg.new_phase}`);
            break;
          
          case 'gpt_first':
            // GPT speaks first in Phase 4
            setState(s => ({
              ...s,
              gptFirstMessage: msg.message ?? '',
            }));
            console.log(`[GPT FIRST] "${msg.message}"`);
            break;
            
          case 'ping':
            ws.send(JSON.stringify({ type: 'pong' }));
            break;
        }
      } catch (e) {
        console.error('Error parsing WebSocket message:', e);
      }
    };
  }, []);

  useEffect(() => {
    connect();
    
    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [connect]);

  const sendMessage = useCallback((msg: object) => {
    console.log('[WS] sendMessage called:', msg, 'readyState:', wsRef.current?.readyState);
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      console.log('[WS] Sending message:', JSON.stringify(msg));
      wsRef.current.send(JSON.stringify(msg));
    } else {
      console.warn('[WS] Cannot send - WebSocket not open. State:', wsRef.current?.readyState);
    }
  }, []);

  return { ...state, sendMessage };
}
