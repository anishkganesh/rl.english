import { useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useWebSocket, ModelType } from './hooks/useWebSocket';
import { AgentGrid } from './components/AgentGrid';
import { Dashboard } from './components/Dashboard';
import { ConversationModal } from './components/ConversationModal';
import { ChatWithAgent } from './components/ChatWithAgent';
import { ScoreText } from './components/ScoreText';
import { WordBank, PatternBank } from './components/WordBank';
import type { Agent, BestOutput } from './types';

// Use environment variable for backend URL, fallback to localhost for development
const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || 'localhost:8000';
const API_PROTOCOL = BACKEND_URL.includes('localhost') ? 'http' : 'https';
const API_BASE = `${API_PROTOCOL}://${BACKEND_URL}`;

const MODEL_LABELS: Record<ModelType, string> = {
  genome: 'Genome',
  char_nn: 'Char NN',
  word_nn: 'Word NN',
  word_genome: 'Word Genome',
  concurrent: 'Concurrent'
};

export default function App() {
  const {
    agents,
    generation,
    isRunning,
    runningStatus,
    history,
    bestOutputs,
    isConnected,
    currentTurn,
    totalTurns,
    milestones,
    conversations,
    latestMilestone,
    globalVocabulary,
    currentModel,
    charNN: _charNN,
    wordNN: _wordNN,
    genome: _genome,
    discoveredBigrams,
    discoveredTrigrams,
    discoveredSentences,
    sendMessage,
    // Word Genome phase tracking
    wordGenomePhase,
    wordGenomePhaseProgress,
    gptFirstMessage,
  } = useWebSocket();
  
  // Toggle for using 10k words vs discovered vocabulary only
  const [use10kWords, setUse10kWords] = useState(true);
  
  const handleToggle10k = useCallback(() => {
    const newValue = !use10kWords;
    setUse10kWords(newValue);
    sendMessage({ type: 'set_use_10k', use_10k: newValue });
  }, [use10kWords, sendMessage]);
  
  const handleResetVocab = useCallback(() => {
    // No confirmation - just reset directly
    sendMessage({ type: 'reset_vocab' });
  }, [sendMessage]);

  const [isSwitchingModel, setIsSwitchingModel] = useState(false);
  
  // Get running state for current model
  const currentModelRunning = runningStatus[currentModel as keyof typeof runningStatus] ?? isRunning;
    
  // Check if any model is running (for some UI decisions)
  const anyModelRunning = runningStatus.genome || runningStatus.char_nn || runningStatus.word_nn || runningStatus.word_genome;

  const [isLoading, setIsLoading] = useState(false);
  const [selectedAgent, setSelectedAgent] = useState<Agent | null>(null);
  const [selectedBestOutput, setSelectedBestOutput] = useState<BestOutput | null>(null);
  const [isChatOpen, setIsChatOpen] = useState(false);
  const [isScoreTextOpen, setIsScoreTextOpen] = useState(false);

  const handleAgentClick = useCallback((agent: Agent) => {
    setSelectedAgent(agent);
    setSelectedBestOutput(null);
  }, []);

  const handleBestOutputClick = useCallback((output: BestOutput) => {
    setSelectedBestOutput(output);
    setSelectedAgent(null);
  }, []);

  const handleCloseModal = useCallback(() => {
    setSelectedAgent(null);
    setSelectedBestOutput(null);
  }, []);

  const handleStart = useCallback(async () => {
    setIsLoading(true);
    try {
      // Use model-specific endpoint to ensure correct model is started
      await fetch(`${API_BASE}/start/${currentModel}`, { method: 'POST' });
    } catch (e) {
      console.error('Failed to start:', e);
    }
    setIsLoading(false);
  }, [currentModel]);

  const handleStop = useCallback(async () => {
    setIsLoading(true);
    try {
      // Use model-specific endpoint to ensure correct model is stopped
      await fetch(`${API_BASE}/stop/${currentModel}`, { method: 'POST' });
    } catch (e) {
      console.error('Failed to stop:', e);
    }
    setIsLoading(false);
  }, [currentModel]);

  const handleReset = useCallback(async () => {
    setIsLoading(true);
    try {
      await fetch(`${API_BASE}/reset`, { method: 'POST' });
    } catch (e) {
      console.error('Failed to reset:', e);
    }
    setIsLoading(false);
  }, []);

  const handleSwitchModel = useCallback(async (model: ModelType) => {
    if (model === currentModel || isSwitchingModel) return;
    
    setIsSwitchingModel(true);
    try {
      await fetch(`${API_BASE}/switch-model`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model })
      });
    } catch (e) {
      console.error('Failed to switch model:', e);
    }
    setIsSwitchingModel(false);
  }, [currentModel, isSwitchingModel]);

  // Calculate current best score
  const bestScore = agents.length > 0 ? Math.max(...agents.map(a => a.score)) : 0;

  // Calculate average exploration
  const avgExploration = agents.length > 0
    ? agents.reduce((sum, a) => sum + a.exploration_rate, 0) / agents.length
    : 0;

  // Agent count for display
  const agentCount = agents.length;

  // Get milestone display name
  const getMilestoneName = (type: string) => {
    switch (type) {
      case 'first_word': return '🎉 First English Word!';
      case 'first_sentence': return '✨ Grammatically Correct!';
      case 'first_coherent': return '🧠 Coherent Response!';
      case 'sustained_conversation': return '💬 Sustained Conversation!';
      default: return '🏆 Achievement Unlocked!';
    }
  };

  return (
    <div className="app">
      {/* Milestone Toast */}
      <AnimatePresence>
        {latestMilestone && (
          <motion.div
            className="milestone-toast"
            initial={{ opacity: 0, y: -50, scale: 0.9 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -20, scale: 0.9 }}
            transition={{ type: 'spring', stiffness: 300, damping: 25 }}
          >
            {getMilestoneName(latestMilestone)}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Header */}
      <header className="header">
        <div className="header-left">
          <div className="logo">rl.english</div>
          
          {/* Model Selector - View only, doesn't stop training */}
          <div className="model-selector">
            {(['genome', 'char_nn', 'word_nn', 'word_genome'] as ModelType[]).map((model) => {
              // Show running indicator for each model
              const modelRunning = runningStatus[model as keyof typeof runningStatus];
              return (
                <button
                  key={model}
                  className={`model-btn ${currentModel === model ? 'active' : ''} ${modelRunning ? 'running' : ''}`}
                  onClick={() => handleSwitchModel(model)}
                  disabled={isSwitchingModel}
                  title={modelRunning ? `${MODEL_LABELS[model]} is running` : MODEL_LABELS[model]}
                >
                  {MODEL_LABELS[model]}
                  {modelRunning && <span className="running-dot" />}
                </button>
              );
            })}
          </div>
          
          {/* Toggle for 10k words vs discovered vocab */}
          {(currentModel === 'word_nn' || currentModel === 'word_genome') && (
            <button
              className={`vocab-toggle ${use10kWords ? 'active' : ''}`}
              onClick={handleToggle10k}
              title={use10kWords ? 'Using 10k common words' : 'Using discovered vocabulary only'}
            >
              {use10kWords ? '10k' : 'Vocab'}
            </button>
          )}
          
          {/* Word Genome Phase Indicator */}
          {currentModel === 'word_genome' && (
            <div className="phase-indicator">
              <span className="phase-label">Phase {wordGenomePhase === 'STRUCTURE' ? '1' : wordGenomePhase === 'PATTERNS' ? '2' : wordGenomePhase === 'SENTENCES' ? '3' : '4'}</span>
              <span className="phase-name">{wordGenomePhase}</span>
              <div className="phase-progress">
                <div 
                  className="phase-progress-bar" 
                  style={{ width: `${wordGenomePhaseProgress * 100}%` }}
                />
              </div>
            </div>
          )}
          
          {/* GPT First Message (Phase 4) */}
          {currentModel === 'word_genome' && wordGenomePhase === 'CONVERSATION' && gptFirstMessage && (
            <div className="gpt-first-message">
              <span className="gpt-label">GPT:</span>
              <span className="gpt-text">{gptFirstMessage}</span>
            </div>
          )}
          
          <div className="stats-bar">
            <div className="stat">
              <span className="stat-label">Generation</span>
              <span className="stat-value">{generation}</span>
            </div>
            <div className="stat">
              <span className="stat-label">Turn</span>
              <span className="stat-value">{currentTurn}/{totalTurns}</span>
            </div>
            <div className="stat">
              <span className="stat-label">Best Score</span>
              <span className="stat-value highlight">
                {bestScore > 0 ? bestScore.toFixed(3) : '—'}
              </span>
            </div>
            <div className="stat">
              <span className="stat-label">Exploration</span>
              <span className="stat-value">
                {avgExploration > 0 ? `${(avgExploration * 100).toFixed(0)}%` : '—'}
              </span>
            </div>
            <div className="stat">
              <span className="stat-label">Agents</span>
              <span className="stat-value">{agentCount}</span>
            </div>
          </div>
        </div>

        {/* Global vocabulary and pattern banks - 2x2 grid */}
        <div className="discovery-banks-container">
          <div className="discovery-banks-grid">
            <WordBank words={globalVocabulary} />
            <PatternBank type="bigrams" items={discoveredBigrams} />
            <PatternBank type="trigrams" items={discoveredTrigrams} />
            <PatternBank type="sentences" items={discoveredSentences} />
          </div>
          <button
            className="reset-vocab-btn"
            onClick={handleResetVocab}
            title="Reset discovered words and patterns"
          >
            ×
          </button>
        </div>

        <div className="header-controls">
          <button
            className="btn"
            onClick={() => setIsScoreTextOpen(true)}
            disabled={!isConnected}
          >
            Baseline
          </button>
          <button
            className="btn"
            onClick={() => setIsChatOpen(true)}
            disabled={!isConnected}
          >
            Chat
          </button>
          {!currentModelRunning ? (
            <button
              className="btn primary"
              onClick={handleStart}
              disabled={isLoading || !isConnected}
            >
              {isLoading ? '...' : 'Start'}
            </button>
          ) : (
            <button
              className="btn"
              onClick={handleStop}
              disabled={isLoading}
            >
              {isLoading ? '...' : 'Pause'}
            </button>
          )}
          <button
            className="btn danger"
            onClick={handleReset}
            disabled={isLoading || currentModelRunning}
          >
            Reset
          </button>
        </div>
      </header>

      {/* Turn progress bar */}
      {currentModelRunning && (
        <div className="generation-progress">
          <div
            className="generation-progress-bar"
            style={{ width: `${(currentTurn / totalTurns) * 100}%` }}
          />
        </div>
      )}

      {/* Main content */}
      <main className="main-content">
        {/* Single model mode: normal view */}
        <AgentGrid 
          agents={agents} 
          isTyping={currentModelRunning} 
          conversations={conversations}
          onAgentClick={handleAgentClick}
        />
        <Dashboard 
          history={history} 
          bestOutputs={bestOutputs} 
          milestones={milestones}
          currentTurn={currentTurn}
          totalTurns={totalTurns}
          onBestOutputClick={handleBestOutputClick}
        />
      </main>

      {/* Conversation Modal */}
      {(selectedAgent || selectedBestOutput) && (
        <ConversationModal
          agent={selectedAgent}
          bestOutput={selectedBestOutput}
          conversation={selectedAgent ? (conversations[selectedAgent.id] || []) : []}
          onClose={handleCloseModal}
        />
      )}

      {/* Chat with Agent Modal */}
      <ChatWithAgent
        isOpen={isChatOpen}
        onClose={() => setIsChatOpen(false)}
      />

      {/* Score Text Modal */}
      <ScoreText
        isOpen={isScoreTextOpen}
        onClose={() => setIsScoreTextOpen(false)}
      />

      {/* Footer */}
      <footer className="footer">
        <div className="status-indicator">
          <div className={`status-dot ${currentModelRunning ? 'running' : ''}`} />
          <span>
            {!isConnected
              ? 'Connecting...'
              : currentModelRunning
              ? `Running • Turn ${currentTurn}/${totalTurns}`
              : anyModelRunning
              ? 'Paused (others running)'
              : 'Ready'}
          </span>
        </div>
        <div className="memory-info">
          Milestones: {milestones.achieved.length}/4
        </div>
      </footer>
    </div>
  );
}
