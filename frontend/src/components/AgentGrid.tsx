import { useMemo } from 'react';
import { AnimatePresence } from 'framer-motion';
import { AgentCard } from './AgentCard';
import type { Agent, ConversationTurn } from '../types';

interface AgentGridProps {
  agents: Agent[];
  isTyping: boolean;
  conversations: Record<string, ConversationTurn[]>;
  onAgentClick?: (agent: Agent) => void;
  compact?: boolean;  // For concurrent mode - smaller cards, 3 per row
}

export function AgentGrid({ agents, isTyping, conversations, onAgentClick, compact = false }: AgentGridProps) {
  // Sort agents by score for ranking
  const rankedAgents = useMemo(() => {
    const sorted = [...agents].sort((a, b) => b.score - a.score);
    return agents.map(agent => ({
      agent,
      rank: sorted.findIndex(a => a.id === agent.id) + 1,
      isBest: sorted[0]?.id === agent.id && sorted[0]?.score > 0,
    }));
  }, [agents]);

  if (agents.length === 0) {
    return (
      <div className="agent-grid-container">
        <div className="empty-state">
          <div className="empty-state-icon">⌨️</div>
          <div>No agents initialized</div>
          <div style={{ fontSize: '0.8rem' }}>Press Start to begin training</div>
        </div>
      </div>
    );
  }

  return (
    <div className={`agent-grid-container ${compact ? 'compact' : ''}`}>
      <div className={`agent-grid ${compact ? 'compact' : ''}`}>
        <AnimatePresence mode="popLayout">
          {rankedAgents.map(({ agent, rank, isBest }) => (
            <AgentCard
              key={agent.id}
              agent={agent}
              rank={rank}
              isBest={isBest}
              isTyping={isTyping}
              conversation={conversations[agent.id]}
              onClick={() => onAgentClick?.(agent)}
              compact={compact}
            />
          ))}
        </AnimatePresence>
      </div>
    </div>
  );
}
