import { ScoreGraph } from './ScoreGraph';
import { BestOutputs } from './BestOutputs';
import type { GenerationStats, BestOutput, Milestones } from '../types';

interface DashboardProps {
  history: GenerationStats[];
  bestOutputs: BestOutput[];
  milestones: Milestones;
  currentTurn: number;
  totalTurns: number;
  onBestOutputClick?: (output: BestOutput) => void;
}

export function Dashboard({ history, bestOutputs, milestones, currentTurn, totalTurns, onBestOutputClick }: DashboardProps) {
  const latestStats = history[history.length - 1];

  return (
    <div className="sidebar">
      {/* Milestones Panel */}
      <div className="panel milestones-panel">
        <div className="panel-header">
          <span className="panel-title">Milestones</span>
        </div>
        <div className="milestones-grid">
          <div className={`milestone-item ${milestones.first_word ? 'achieved' : ''}`}>
            <span className="milestone-icon">{milestones.first_word ? '✓' : '○'}</span>
            <span className="milestone-label">First Word</span>
          </div>
          <div className={`milestone-item ${milestones.first_sentence ? 'achieved' : ''}`}>
            <span className="milestone-icon">{milestones.first_sentence ? '✓' : '○'}</span>
            <span className="milestone-label">Grammar OK</span>
          </div>
          <div className={`milestone-item ${milestones.first_coherent ? 'achieved' : ''}`}>
            <span className="milestone-icon">{milestones.first_coherent ? '✓' : '○'}</span>
            <span className="milestone-label">Coherent</span>
          </div>
          <div className={`milestone-item ${milestones.sustained_conversation ? 'achieved' : ''}`}>
            <span className="milestone-icon">{milestones.sustained_conversation ? '✓' : '○'}</span>
            <span className="milestone-label">Sustained</span>
          </div>
        </div>
      </div>

      {/* Turn Progress */}
      {currentTurn > 0 && (
        <div className="panel turn-panel">
          <div className="panel-header">
            <span className="panel-title">Conversation</span>
            <span className="turn-indicator">Turn {currentTurn}/{totalTurns}</span>
          </div>
          <div className="turn-progress">
            {Array.from({ length: totalTurns }).map((_, i) => (
              <div 
                key={i} 
                className={`turn-dot ${i < currentTurn ? 'completed' : ''} ${i === currentTurn - 1 ? 'current' : ''}`}
              />
            ))}
          </div>
        </div>
      )}

      {/* Score History */}
      <div className="panel">
        <div className="panel-header">
          <span className="panel-title">Score History</span>
          {latestStats && (
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
              Last: {(latestStats.best_score ?? latestStats.max ?? 0).toFixed(2)}
            </span>
          )}
        </div>
        <ScoreGraph history={history} />
      </div>

      {/* Best Outputs */}
      <div className="panel flex-1">
        <div className="panel-header">
          <span className="panel-title">Best Conversations</span>
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
            Top {bestOutputs.length}
          </span>
        </div>
        <BestOutputs outputs={bestOutputs} onOutputClick={onBestOutputClick} />
      </div>
    </div>
  );
}
