import { useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

interface ChatResponse {
  response: string;
  agent_id: string | null;
  agent_score: number;
  generation: number;
}

interface ChatWithAgentProps {
  isOpen: boolean;
  onClose: () => void;
}

export function ChatWithAgent({ isOpen, onClose }: ChatWithAgentProps) {
  const [message, setMessage] = useState('');
  const [response, setResponse] = useState<ChatResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSend = useCallback(async () => {
    if (!message.trim()) return;
    
    setIsLoading(true);
    setError(null);
    
    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: message.trim() })
      });
      
      if (!res.ok) {
        throw new Error('Failed to get response');
      }
      
      const data = await res.json();
      setResponse(data);
    } catch (e) {
      setError('Failed to chat with agent. Make sure training has run at least once.');
    } finally {
      setIsLoading(false);
    }
  }, [message]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
    if (e.key === 'Escape') {
      onClose();
    }
  };

  if (!isOpen) return null;

  return (
    <AnimatePresence>
      <motion.div
        className="modal-overlay"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        onClick={(e) => e.target === e.currentTarget && onClose()}
      >
        <motion.div
          className="modal-content"
          initial={{ opacity: 0, scale: 0.95, y: 10 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          exit={{ opacity: 0, scale: 0.95, y: 10 }}
          style={{ maxWidth: '500px' }}
        >
          <div className="modal-header">
            <div>
              <span className="modal-title">Chat with Best Agent</span>
              {response && (
                <span className="modal-subtitle">
                  Gen {response.generation} • Score {response.agent_score?.toFixed(2)}
                </span>
              )}
            </div>
            <button className="modal-close" onClick={onClose}>×</button>
          </div>

          <div className="modal-body">
            <div className="chat-input-area">
              <input
                type="text"
                className="chat-input"
                placeholder="Type a message to see how the agent responds..."
                value={message}
                onChange={(e) => setMessage(e.target.value)}
                onKeyDown={handleKeyDown}
                disabled={isLoading}
              />
              <button
                className="btn primary"
                onClick={handleSend}
                disabled={isLoading || !message.trim()}
              >
                {isLoading ? '...' : 'Send'}
              </button>
            </div>

            {error && (
              <div className="chat-error">{error}</div>
            )}

            {response && (
              <div className="chat-response">
                <div className="chat-response-label">Agent's Response:</div>
                <div className="chat-response-text">
                  {response.response || '(no response)'}
                </div>
              </div>
            )}

            <div className="chat-note">
              Note: The agent responds using its learned character probability distribution.
              It's still learning, so responses may be random at first!
            </div>
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
}

