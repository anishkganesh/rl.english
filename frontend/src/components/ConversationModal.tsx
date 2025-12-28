import { useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { NeuralViz } from './NeuralViz';
import type { Agent, ConversationTurn, BestOutput } from '../types';

interface ConversationModalProps {
  agent?: Agent | null;
  bestOutput?: BestOutput | null;
  conversation: ConversationTurn[];
  onClose: () => void;
}

export function ConversationModal({ agent, bestOutput, conversation, onClose }: ConversationModalProps) {
  // Handle escape key
  const handleKeyDown = useCallback((e: KeyboardEvent) => {
    if (e.key === 'Escape') {
      onClose();
    }
  }, [onClose]);

  useEffect(() => {
    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [handleKeyDown]);

  // Handle click outside
  const handleOverlayClick = (e: React.MouseEvent) => {
    if (e.target === e.currentTarget) {
      onClose();
    }
  };

  // Support both agent and bestOutput modes
  const isOpen = !!(agent || bestOutput);
  if (!isOpen) return null;

  // Get scores and conversation from either source
  const scores = agent?.scores_breakdown || bestOutput?.scores_breakdown;
  const displayConversation = conversation.length > 0 ? conversation : (bestOutput?.conversation || []);
  const title = agent ? agent.id : `Generation ${bestOutput?.generation}`;
  const totalScore = agent?.score ?? bestOutput?.score ?? 0;

  return (
    <AnimatePresence>
      <motion.div
        className="modal-overlay"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        transition={{ duration: 0.15 }}
        onClick={handleOverlayClick}
      >
        <motion.div
          className="modal-content"
          initial={{ opacity: 0, scale: 0.95, y: 10 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          exit={{ opacity: 0, scale: 0.95, y: 10 }}
          transition={{ duration: 0.2, ease: 'easeOut' }}
        >
          {/* Header */}
          <div className="modal-header">
            <div>
              <span className="modal-title">Conversation</span>
              <span className="modal-subtitle">{title}</span>
            </div>
            <button className="modal-close" onClick={onClose}>
              ×
            </button>
          </div>

          {/* Body - Conversation */}
          <div className="modal-body">
            {displayConversation.length > 0 ? (
              <div className="modal-conversation">
                {displayConversation.map((turn, i) => (
                  <div key={i} className="modal-turn">
                    <span className="modal-turn-label">Turn {i + 1}</span>
                    <div className="modal-turn-agent">
                      {turn.agent_text || '(no output)'}
                    </div>
                    {turn.openai_response && (
                      <div className="modal-turn-gpt">
                        {turn.openai_response}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            ) : (
              <div className="modal-conversation">
                <div className="modal-turn">
                  <span className="modal-turn-label">Current Output</span>
                  <div className="modal-turn-agent">
                    {agent?.typed_text || '(typing...)'}
                  </div>
                </div>
              </div>
            )}

            {/* Word Genome Details OR Neural Network Visualization */}
            {agent && (
              agent.model_type === 'word_genome' || agent.top_words ? (
                <div className="modal-genome-section">
                  {/* Word Probability Distribution */}
                  {agent.top_words && agent.top_words.length > 0 && (
                    <div className="modal-word-probs">
                      <h3 className="modal-section-title">Word Probabilities</h3>
                      <div className="word-prob-list">
                        {agent.top_words.map((w, i) => (
                          <div key={i} className="word-prob-row">
                            <span className="word-prob-word">{w.word}</span>
                            <div className="word-prob-bar-container">
                              <div 
                                className="word-prob-bar" 
                                style={{ width: `${Math.min(w.prob * 100 * 10, 100)}%` }}
                              />
                            </div>
                            <span className="word-prob-value">{(w.prob * 100).toFixed(1)}%</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                  
                  {/* Lineage Tree */}
                  {agent.lineage && agent.lineage.length > 0 && (
                    <div className="modal-lineage">
                      <h3 className="modal-section-title">Lineage</h3>
                      <div className="lineage-tree">
                        {/* Current agent */}
                        <div className="lineage-node current">
                          <span className={`lineage-origin ${agent.origin || 'random'}`}>
                            {agent.origin === 'elite' ? 'Elite' : 
                             agent.origin === 'crossover' ? 'Child' :
                             agent.origin === 'mutation' ? 'Mutant' : 'New'}
                          </span>
                          <span className="lineage-id">{agent.id}</span>
                          <span className="lineage-gen">Gen {agent.generation_born ?? '?'}</span>
                        </div>
                        
                        {/* Parent connections */}
                        {agent.parent_ids && agent.parent_ids.length > 0 && (
                          <div className="lineage-parents">
                            <span className="lineage-connector">↑ from </span>
                            <span className="lineage-parent-ids">
                              {agent.parent_ids.map((pid, i) => (
                                <span key={i} className="lineage-parent-id">
                                  {pid}{i < agent.parent_ids!.length - 1 ? ' + ' : ''}
                                </span>
                              ))}
                            </span>
                          </div>
                        )}
                        
                        {/* Ancestry */}
                        {agent.lineage.slice(0, 5).map((entry, i) => (
                          <div key={i} className="lineage-ancestor">
                            <span className="lineage-indent">{'  '.repeat(i + 1)}└─</span>
                            <span className={`lineage-origin ${entry.origin}`}>
                              {entry.origin === 'elite' ? 'Elite' : 
                               entry.origin === 'crossover' ? 'Child' :
                               entry.origin === 'mutation' ? 'Mutant' : 'New'}
                            </span>
                            <span className="lineage-id">{entry.id}</span>
                            <span className="lineage-gen">Gen {entry.gen}</span>
                          </div>
                        ))}
                        {agent.lineage.length > 5 && (
                          <div className="lineage-more">
                            <span className="lineage-indent">{'  '.repeat(6)}...</span>
                            <span className="lineage-count">+{agent.lineage.length - 5} more ancestors</span>
                          </div>
                        )}
                      </div>
                    </div>
                  )}
                </div>
              ) : (
                <div className="modal-viz-section">
                  <h3 className="modal-viz-title">Neural Network</h3>
                  <NeuralViz agentId={agent.id} isVisible={true} compact={false} />
                </div>
              )
            )}
          </div>

          {/* Footer - Scores */}
          <div className="modal-footer">
            <div className="modal-scores">
              <div className="modal-score-item">
                <span className="modal-score-label">Vocabulary</span>
                <span className="modal-score-value">
                  {scores?.vocabulary_score !== undefined 
                    ? `${(scores.vocabulary_score * 100).toFixed(0)}%` 
                    : '—'}
                </span>
              </div>
              <div className="modal-score-item">
                <span className="modal-score-label">Grammar</span>
                <span className="modal-score-value">
                  {scores?.grammar_score !== undefined 
                    ? `${(scores.grammar_score * 100).toFixed(0)}%` 
                    : '—'}
                </span>
              </div>
              <div className="modal-score-item">
                <span className="modal-score-label">Coherence</span>
                <span className="modal-score-value">
                  {scores?.coherence_score !== undefined 
                    ? `${(scores.coherence_score * 100).toFixed(0)}%` 
                    : '—'}
                </span>
              </div>
            </div>
            <div className="modal-score-item">
              <span className="modal-score-label">Total Score</span>
              <span className="modal-score-value total">
                {totalScore > 0 ? totalScore.toFixed(3) : '—'}
              </span>
            </div>
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
}

