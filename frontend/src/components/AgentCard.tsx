import { motion } from 'framer-motion';
import { NeuralViz } from './NeuralViz';
import type { Agent, ConversationTurn } from '../types';

interface AgentCardProps {
  agent: Agent;
  rank: number;
  isBest: boolean;
  isTyping: boolean;
  conversation?: ConversationTurn[];
  onClick?: () => void;
  showNeuralViz?: boolean;
  compact?: boolean;  // For concurrent mode - smaller cards
}

export function AgentCard({ agent, rank, isBest, isTyping, conversation, onClick, showNeuralViz = false, compact = false }: AgentCardProps) {
  const scores = agent.scores_breakdown;
  
  return (
    <motion.div
      className={`agent-card ${isBest ? 'best' : ''} ${compact ? 'compact' : ''}`}
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.2 }}
      layout
      onClick={onClick}
    >
      <div className="agent-header">
        <span className="agent-id">{agent.id}</span>
        <div className="agent-header-right">
          {agent.origin && (
            <span className={`origin-badge ${agent.origin}`}>
              {agent.origin === 'elite' ? 'Elite' : 
               agent.origin === 'crossover' ? 'Child' :
               agent.origin === 'mutation' ? 'Mutant' : 'New'}
            </span>
          )}
          <span className="agent-rank">#{rank}</span>
        </div>
      </div>
      
      {/* Show agent's conversation or typed text */}
      {conversation && conversation.length > 0 ? (
        <div className="agent-conversation">
          {conversation.map((turn, idx) => (
            <div key={idx} className="conversation-turn">
              <div className="turn-agent">
                <span className="turn-label">Agent:</span>
                <span className="turn-text">{turn.agent_text || '...'}</span>
              </div>
              {turn.openai_response && (
                <div className="turn-openai">
                  <span className="turn-label">GPT:</span>
                  <span className="turn-text">{turn.openai_response.slice(0, 80)}{turn.openai_response.length > 80 ? '...' : ''}</span>
                </div>
              )}
            </div>
          ))}
          {isTyping && <span className="cursor" />}
        </div>
      ) : (
        <div className="agent-text">
          {agent.typed_text || <span style={{ opacity: 0.3 }}>...</span>}
          {isTyping && <span className="cursor" />}
        </div>
      )}
      
      {/* Vocabulary display */}
      {agent.vocabulary && agent.vocabulary.length > 0 && (
        <div className="agent-vocabulary" title={agent.vocabulary.join(', ')}>
          <span className="vocab-words">
            {agent.vocabulary.slice(0, 5).join(', ')}
            {agent.vocabulary.length > 5 && ` +${agent.vocabulary.length - 5}`}
          </span>
        </div>
      )}
      
      {/* Top words preview (Word Genome only) */}
      {agent.top_words && agent.top_words.length > 0 && (
        <div className="top-words-preview">
          {agent.top_words.slice(0, 5).map((w, i) => (
            <span key={i} className="top-word">
              {w.word}: {(w.prob * 100).toFixed(0)}%
            </span>
          ))}
        </div>
      )}
      
      {/* Score breakdown */}
      <div className="agent-scores">
        <div className="score-row">
          <span className="score-label">Total</span>
          <span className="score-value primary">
            {agent.score > 0 ? agent.score.toFixed(2) : '—'}
            {scores?.vocab_bonus ? ` (+${(scores.vocab_bonus * 100).toFixed(0)}%)` : ''}
          </span>
        </div>
        {scores && (
          <div className="score-breakdown">
            <div className="score-item">
              <span className="score-item-label">Vocab</span>
              <div className="score-mini-bar">
                <div 
                  className="score-mini-fill vocab" 
                  style={{ width: `${(scores.vocabulary_score ?? 0) * 100}%` }}
                />
              </div>
              <span className="score-item-value">
                {scores.vocabulary_score !== undefined ? (scores.vocabulary_score * 100).toFixed(0) + '%' : '—'}
              </span>
            </div>
            <div className="score-item">
              <span className="score-item-label">Grammar</span>
              <div className="score-mini-bar">
                <div 
                  className="score-mini-fill grammar" 
                  style={{ width: `${(scores.grammar_score ?? 0) * 100}%` }}
                />
              </div>
              <span className="score-item-value">
                {scores.grammar_score !== undefined ? (scores.grammar_score * 100).toFixed(0) + '%' : '—'}
              </span>
            </div>
            <div className="score-item">
              <span className="score-item-label">Coherence</span>
              <div className="score-mini-bar">
                <div 
                  className="score-mini-fill coherence" 
                  style={{ width: `${(scores.coherence_score ?? 0) * 100}%` }}
                />
              </div>
              <span className="score-item-value">
                {scores.coherence_score !== undefined ? (scores.coherence_score * 100).toFixed(0) + '%' : '—'}
              </span>
            </div>
          </div>
        )}
      </div>
      
      {/* Show scoring reason for Word Genome, or Neural Viz for NN models */}
      {(isBest || showNeuralViz) && (
        scores?.reason ? (
          <div className="agent-reason">
            <span className="reason-text">{scores.reason}</span>
          </div>
        ) : (
          <NeuralViz agentId={agent.id} isVisible={true} compact={true} />
        )
      )}
    </motion.div>
  );
}
