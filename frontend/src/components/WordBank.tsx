import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import type { DiscoveredSentence } from '../types';

interface WordBankProps {
  words: string[];
}

interface PatternBankProps {
  type: 'bigrams' | 'trigrams' | 'sentences';
  items: string[] | DiscoveredSentence[];
}

export function PatternBank({ type, items }: PatternBankProps) {
  const [isModalOpen, setIsModalOpen] = useState(false);
  
  const label = type === 'bigrams' ? 'Bigrams Found' 
              : type === 'trigrams' ? 'Trigrams Found' 
              : 'Sentences Found';
  
  const isSentences = type === 'sentences';
  const patterns = isSentences 
    ? (items as DiscoveredSentence[]).map(s => s.text)
    : (items as string[]);
  
  return (
    <>
      <div 
        className="word-bank-inline clickable"
        onClick={() => setIsModalOpen(true)}
        style={{ cursor: items.length > 0 ? 'pointer' : 'default' }}
      >
        <span className="word-bank-label">{label}</span>
        <span className="word-bank-count">{items.length}</span>
        {items.length > 0 && (
          <div className="word-bank-words-inline">
            <AnimatePresence mode="popLayout">
              {patterns.slice(0, 10).map((pattern, i) => (
                <motion.span
                  key={`${pattern}-${i}`}
                  className={`word-bank-word-inline ${isSentences ? 'sentence-item' : ''}`}
                  initial={{ opacity: 0, scale: 0.8 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.8 }}
                  transition={{ duration: 0.15 }}
                  layout
                >
                  {isSentences ? `"${pattern.slice(0, 25)}${pattern.length > 25 ? '...' : ''}"` : pattern}
                </motion.span>
              ))}
              {patterns.length > 10 && (
                <span className="word-bank-more">+{patterns.length - 10}</span>
              )}
            </AnimatePresence>
          </div>
        )}
      </div>

      {/* Pattern Bank Modal */}
      <AnimatePresence>
        {isModalOpen && (
          <motion.div
            className="modal-overlay"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={() => setIsModalOpen(false)}
          >
            <motion.div
              className="modal-content word-bank-modal"
              initial={{ opacity: 0, scale: 0.95, y: 20 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.95, y: 20 }}
              onClick={(e) => e.stopPropagation()}
            >
              <div className="modal-header">
                <h2 className="modal-title">{label}</h2>
                <span className="modal-subtitle">{items.length} {type} discovered</span>
                <button className="modal-close" onClick={() => setIsModalOpen(false)}>
                  &times;
                </button>
              </div>
              
              <div className="word-bank-modal-content">
                {items.length === 0 ? (
                  <div className="word-bank-empty">
                    <p>No {type} discovered yet</p>
                    <p className="hint">{type === 'sentences' ? 'Valid sentences appear here when Word NN generates coherent output' : `${type.charAt(0).toUpperCase() + type.slice(1)} appear here as agents find common patterns`}</p>
                  </div>
                ) : isSentences ? (
                  <div className="sentences-modal-list">
                    {(items as DiscoveredSentence[]).map((sentence, i) => (
                      <div key={i} className="sentence-modal-item">
                        <span className="sentence-text">"{sentence.text}"</span>
                        <div className="sentence-meta">
                          <span className="sentence-score">Score: {(sentence.score * 100).toFixed(0)}%</span>
                          <span className="sentence-gen">Gen {sentence.generation}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="pattern-modal-list">
                    {(items as string[]).map((pattern, i) => (
                      <span key={i} className="pattern-modal-item">{pattern}</span>
                    ))}
                  </div>
                )}
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}

export function WordBank({ words }: WordBankProps) {
  const [isModalOpen, setIsModalOpen] = useState(false);
  
  // Sort words alphabetically for display
  const sortedWords = [...words].sort();
  
  // Group words by length for the modal
  const wordsByLength: Record<number, string[]> = {};
  sortedWords.forEach(word => {
    const len = word.length;
    if (!wordsByLength[len]) wordsByLength[len] = [];
    wordsByLength[len].push(word);
  });
  const lengths = Object.keys(wordsByLength).map(Number).sort((a, b) => b - a); // longest first
  
  return (
    <>
      <div 
        className="word-bank-inline clickable"
        onClick={() => setIsModalOpen(true)}
        style={{ cursor: words.length > 0 ? 'pointer' : 'default' }}
      >
        <span className="word-bank-label">Words Found</span>
        <span className="word-bank-count">{words.length}</span>
        {words.length > 0 && (
          <div className="word-bank-words-inline">
            <AnimatePresence mode="popLayout">
              {sortedWords.slice(0, 20).map((word) => (
                <motion.span
                  key={word}
                  className="word-bank-word-inline"
                  initial={{ opacity: 0, scale: 0.8 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.8 }}
                  transition={{ duration: 0.15 }}
                  layout
                >
                  {word}
                </motion.span>
              ))}
              {sortedWords.length > 20 && (
                <span className="word-bank-more">+{sortedWords.length - 20}</span>
              )}
            </AnimatePresence>
          </div>
        )}
      </div>

      {/* Word Bank Modal */}
      <AnimatePresence>
        {isModalOpen && (
          <motion.div
            className="modal-overlay"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={() => setIsModalOpen(false)}
          >
            <motion.div
              className="modal-content word-bank-modal"
              initial={{ opacity: 0, scale: 0.95, y: 20 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.95, y: 20 }}
              onClick={(e) => e.stopPropagation()}
            >
              <div className="modal-header">
                <h2 className="modal-title">Word Bank</h2>
                <span className="modal-subtitle">{words.length} words discovered</span>
                <button className="modal-close" onClick={() => setIsModalOpen(false)}>
                  &times;
                </button>
              </div>
              
              <div className="word-bank-modal-content">
                {words.length === 0 ? (
                  <div className="word-bank-empty">
                    <p>No words discovered yet</p>
                    <p className="hint">Words appear here as agents find them</p>
                  </div>
                ) : (
                  <div className="word-bank-groups">
                    {lengths.map(len => (
                      <div key={len} className="word-bank-group">
                        <div className="word-bank-group-header">
                          {len} letter{len !== 1 ? 's' : ''} ({wordsByLength[len].length})
                        </div>
                        <div className="word-bank-group-words">
                          {wordsByLength[len].sort().map(word => (
                            <span key={word} className="word-bank-modal-word">
                              {word}
                            </span>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}
