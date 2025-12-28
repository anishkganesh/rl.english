import { useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

interface ScoreResult {
  vocabulary: number;
  grammar: number;
  coherence: number;
  total: number;
  reason: string;
  valid_words: string[];
}

interface ScoreTextProps {
  isOpen: boolean;
  onClose: () => void;
}

export function ScoreText({ isOpen, onClose }: ScoreTextProps) {
  const [text, setText] = useState('');
  const [context, setContext] = useState('');
  const [result, setResult] = useState<ScoreResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleScore = useCallback(async () => {
    if (!text.trim()) return;
    
    setIsLoading(true);
    setError(null);
    setResult(null);
    
    try {
      const res = await fetch('/api/score-text', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: text.trim(), context: context.trim() })
      });
      
      const data = await res.json();
      
      if (data.error) {
        setError(data.error);
      } else {
        setResult(data.scores);
      }
    } catch (e) {
      setError('Failed to score text');
    } finally {
      setIsLoading(false);
    }
  }, [text, context]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && e.metaKey) {
      handleScore();
    }
    if (e.key === 'Escape') {
      onClose();
    }
  };

  const formatPercent = (value: number) => `${Math.round(value * 100)}%`;

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          className="modal-overlay"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          onClick={onClose}
        >
          <motion.div
            className="modal-content"
            initial={{ y: -20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            exit={{ y: -20, opacity: 0 }}
            onClick={(e) => e.stopPropagation()}
            style={{ maxWidth: '600px' }}
          >
            <div className="modal-header">
              <div>
                <span className="modal-title">Score Text</span>
                <span className="modal-subtitle">Test the OpenAI scoring system</span>
              </div>
              <button className="modal-close" onClick={onClose}>×</button>
            </div>

            <div className="modal-body">
              <div className="score-text-input-group">
                <label className="score-text-label">Your text to score:</label>
                <textarea
                  className="score-text-input"
                  placeholder="Type a sentence to see how the AI would score it..."
                  value={text}
                  onChange={(e) => setText(e.target.value)}
                  onKeyDown={handleKeyDown}
                  rows={3}
                />
              </div>

              <div className="score-text-input-group">
                <label className="score-text-label">Context (optional):</label>
                <input
                  type="text"
                  className="score-text-context"
                  placeholder="Previous conversation context..."
                  value={context}
                  onChange={(e) => setContext(e.target.value)}
                  onKeyDown={handleKeyDown}
                />
              </div>

              <button 
                className="btn primary" 
                onClick={handleScore}
                disabled={isLoading || !text.trim()}
                style={{ width: '100%', marginTop: 'var(--spacing-sm)' }}
              >
                {isLoading ? 'Scoring...' : 'Score Text (⌘+Enter)'}
              </button>

              {error && (
                <div className="score-text-error">
                  {error}
                </div>
              )}

              {result && (
                <motion.div 
                  className="score-text-result"
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                >
                  <div className="score-text-total">
                    <span className="score-text-total-label">Total Score</span>
                    <span className="score-text-total-value">{formatPercent(result.total)}</span>
                  </div>

                  <div className="score-text-breakdown">
                    <div className="score-text-item">
                      <span className="score-text-item-label">Vocabulary</span>
                      <div className="score-text-bar-container">
                        <div 
                          className="score-text-bar vocab"
                          style={{ width: formatPercent(result.vocabulary) }}
                        />
                      </div>
                      <span className="score-text-item-value">{formatPercent(result.vocabulary)}</span>
                    </div>

                    <div className="score-text-item">
                      <span className="score-text-item-label">Grammar</span>
                      <div className="score-text-bar-container">
                        <div 
                          className="score-text-bar grammar"
                          style={{ width: formatPercent(result.grammar) }}
                        />
                      </div>
                      <span className="score-text-item-value">{formatPercent(result.grammar)}</span>
                    </div>

                    <div className="score-text-item">
                      <span className="score-text-item-label">Coherence</span>
                      <div className="score-text-bar-container">
                        <div 
                          className="score-text-bar coherence"
                          style={{ width: formatPercent(result.coherence) }}
                        />
                      </div>
                      <span className="score-text-item-value">{formatPercent(result.coherence)}</span>
                    </div>
                  </div>

                  {result.reason && (
                    <div className="score-text-reason">
                      <span className="score-text-reason-label">Reason:</span>
                      <span className="score-text-reason-text">{result.reason}</span>
                    </div>
                  )}

                  {result.valid_words && result.valid_words.length > 0 && (
                    <div className="score-text-words">
                      <span className="score-text-words-label">Valid words found:</span>
                      <span className="score-text-words-list">
                        {result.valid_words.join(', ')}
                      </span>
                    </div>
                  )}
                </motion.div>
              )}
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}

