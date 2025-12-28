import { motion, AnimatePresence } from 'framer-motion';
import type { BestOutput } from '../types';

interface BestOutputsProps {
  outputs: BestOutput[];
  onOutputClick?: (output: BestOutput) => void;
}

export function BestOutputs({ outputs, onOutputClick }: BestOutputsProps) {
  if (outputs.length === 0) {
    return (
      <div className="best-outputs">
        <div className="empty-state" style={{ height: '100%' }}>
          <div style={{ fontSize: '0.8rem' }}>Hall of fame</div>
          <div style={{ fontSize: '0.7rem', opacity: 0.6 }}>Best conversations appear here</div>
        </div>
      </div>
    );
  }

  return (
    <div className="best-outputs">
      <AnimatePresence mode="popLayout">
        {outputs.slice(0, 10).map((output, index) => (
          <motion.div
            key={`${output.generation}-${index}`}
            className="best-output-item clickable"
            initial={{ opacity: 0, x: -10 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: 10 }}
            transition={{ duration: 0.2, delay: index * 0.05 }}
            onClick={() => onOutputClick?.(output)}
          >
            <div className="best-output-header">
              <span className="best-output-gen">G{output.generation}</span>
              <span className="best-output-score">{output.score.toFixed(2)}</span>
            </div>
            
            {output.conversation && output.conversation.length > 0 ? (
              <div className="best-output-conversation">
                {output.conversation.slice(0, 2).map((turn, i) => (
                  <div key={i} className="mini-turn">
                    <span className="mini-turn-agent">→ {turn.agent_text.slice(0, 50)}{turn.agent_text.length > 50 ? '...' : ''}</span>
                    <span className="mini-turn-gpt">← {turn.openai_response.slice(0, 40)}...</span>
                  </div>
                ))}
              </div>
            ) : (
              <span className="best-output-text">
                {output.text || '(empty)'}
              </span>
            )}
            
            {output.scores_breakdown && (
              <div className="best-output-breakdown">
                <span className="breakdown-item">V:{(output.scores_breakdown.vocabulary_score * 100).toFixed(0)}%</span>
                <span className="breakdown-item">G:{(output.scores_breakdown.grammar_score * 100).toFixed(0)}%</span>
                <span className="breakdown-item">C:{(output.scores_breakdown.coherence_score * 100).toFixed(0)}%</span>
              </div>
            )}
          </motion.div>
        ))}
      </AnimatePresence>
    </div>
  );
}
