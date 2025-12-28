import { useEffect, useState } from 'react';
import { motion } from 'framer-motion';

interface CharProb {
  char: string;
  prob: number;
}

interface VizData {
  char_probs?: CharProb[];
  word_probs?: { word: string; prob: number }[];
  attention: number[];
  layer_activations: number[][];
  context_chars?: string[];
  context_words?: string[];
  num_layers: number;
  num_heads: number;
  is_word_nn?: boolean;
}

interface NeuralVizProps {
  agentId: string;
  isVisible: boolean;
  compact?: boolean;
}

export function NeuralViz({ agentId, isVisible, compact = false }: NeuralVizProps) {
  const [vizData, setVizData] = useState<VizData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!isVisible || !agentId) return;

    const fetchVizData = async () => {
      setLoading(true);
      setError(null);
      try {
        const response = await fetch(`/api/agent/${agentId}/viz`);
        const data = await response.json();
        if (data.viz) {
          setVizData(data.viz);
        } else if (data.error) {
          setError(data.error);
        }
      } catch (err) {
        setError('Failed to load visualization');
      }
      setLoading(false);
    };

    fetchVizData();
    // Refresh every 2 seconds while visible
    const interval = setInterval(fetchVizData, 2000);
    return () => clearInterval(interval);
  }, [agentId, isVisible]);

  if (!isVisible) return null;

  if (loading && !vizData) {
    return (
      <div className="neural-viz-loading">
        <span>Loading neural network...</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="neural-viz-error">
        <span>{error}</span>
      </div>
    );
  }

  if (!vizData) return null;

  if (compact) {
    return <NeuralVizCompact data={vizData} />;
  }

  return <NeuralVizFull data={vizData} />;
}

function NeuralVizCompact({ data }: { data: VizData }) {
  // Handle both char_probs (Char NN) and word_probs (Word NN)
  const topItems = data?.char_probs?.slice(0, 5) ?? [];
  const isWordNN = data?.is_word_nn ?? false;
  
  if (topItems.length === 0) {
    return <div className="neural-viz-compact"><span className="viz-empty">No data</span></div>;
  }
  
  return (
    <div className="neural-viz-compact">
      <div className="viz-mini-probs">
        {topItems.map((cp, i) => {
          const label = isWordNN ? (cp.char?.slice(0, 6) ?? '?') : (cp.char === ' ' ? '␣' : cp.char);
          return (
            <div key={i} className="mini-prob-bar">
              <span className="mini-char" title={cp.char}>{label}</span>
              <div className="mini-bar-bg">
                <motion.div 
                  className="mini-bar-fill"
                  initial={{ width: 0 }}
                  animate={{ width: `${(cp.prob || 0) * 100}%` }}
                  transition={{ duration: 0.3 }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function NeuralVizFull({ data }: { data: VizData }) {
  const isWordNN = data?.is_word_nn ?? false;
  const contextItems = data?.context_chars ?? data?.context_words ?? [];
  const probItems = data?.char_probs ?? [];
  
  return (
    <div className="neural-viz-full">
      {/* Network Diagram */}
      <div className="viz-section">
        <h4 className="viz-section-title">Network</h4>
        <NetworkDiagram 
          numLayers={data.num_layers || 2} 
          layerActivations={data.layer_activations || []}
        />
      </div>

      {/* Attention Heatmap */}
      <div className="viz-section">
        <h4 className="viz-section-title">Attention</h4>
        <AttentionHeatmap 
          attention={data.attention || []} 
          contextChars={contextItems}
          isWordNN={isWordNN}
        />
      </div>

      {/* Character/Word Probabilities */}
      <div className="viz-section">
        <h4 className="viz-section-title">
          {isWordNN ? 'Next Word Probabilities' : 'Next Character Probabilities'}
        </h4>
        <CharProbsChart probs={probItems} isWordNN={isWordNN} />
      </div>
    </div>
  );
}

function NetworkDiagram({ numLayers, layerActivations }: { numLayers: number; layerActivations: number[][] }) {
  const nodesPerLayer = 6; // Show 6 nodes per layer for visual clarity
  const layerXPositions = [30, 100, 170, 240]; // Input, L1, L2, Output
  
  // Get activation for connection opacity
  const getActivation = (layerIdx: number, nodeIdx: number): number => {
    if (layerIdx < 0 || layerIdx >= layerActivations.length) return 0.3;
    const activations = layerActivations[layerIdx] || [];
    const actIdx = Math.floor((nodeIdx / nodesPerLayer) * Math.max(activations.length, 1));
    return activations[actIdx] ?? 0.3;
  };
  
  return (
    <div className="network-diagram">
      <svg viewBox="0 0 270 100" className="network-svg">
        {/* Draw connections first (behind nodes) */}
        <g className="connections">
          {/* Input to L1 connections */}
          {Array.from({ length: nodesPerLayer }).map((_, fromIdx) => (
            Array.from({ length: nodesPerLayer }).map((_, toIdx) => {
              const activation = getActivation(0, toIdx);
              return (
                <motion.line
                  key={`conn-0-${fromIdx}-${toIdx}`}
                  x1={layerXPositions[0] + 5}
                  y1={12 + fromIdx * 14}
                  x2={layerXPositions[1] - 5}
                  y2={12 + toIdx * 14}
                  className="network-connection"
                  initial={{ opacity: 0.05 }}
                  animate={{ opacity: 0.08 + activation * 0.15 }}
                  transition={{ duration: 0.5 }}
                />
              );
            })
          )).flat()}
          
          {/* L1 to L2 connections */}
          {Array.from({ length: nodesPerLayer }).map((_, fromIdx) => (
            Array.from({ length: nodesPerLayer }).map((_, toIdx) => {
              const activation = getActivation(1, toIdx);
              return (
                <motion.line
                  key={`conn-1-${fromIdx}-${toIdx}`}
                  x1={layerXPositions[1] + 5}
                  y1={12 + fromIdx * 14}
                  x2={layerXPositions[2] - 5}
                  y2={12 + toIdx * 14}
                  className="network-connection"
                  initial={{ opacity: 0.05 }}
                  animate={{ opacity: 0.08 + activation * 0.15 }}
                  transition={{ duration: 0.5, delay: 0.1 }}
                />
              );
            })
          )).flat()}
          
          {/* L2 to Output connections */}
          {Array.from({ length: nodesPerLayer }).map((_, fromIdx) => (
            Array.from({ length: nodesPerLayer }).map((_, toIdx) => {
              const activation = getActivation(1, fromIdx);
              return (
                <motion.line
                  key={`conn-2-${fromIdx}-${toIdx}`}
                  x1={layerXPositions[2] + 5}
                  y1={12 + fromIdx * 14}
                  x2={layerXPositions[3] - 5}
                  y2={12 + toIdx * 14}
                  className="network-connection output-connection"
                  initial={{ opacity: 0.05 }}
                  animate={{ opacity: 0.08 + activation * 0.2 }}
                  transition={{ duration: 0.5, delay: 0.2 }}
                />
              );
            })
          )).flat()}
        </g>

        {/* Input layer */}
        <g className="layer input-layer">
          {Array.from({ length: nodesPerLayer }).map((_, i) => (
            <circle
              key={`input-${i}`}
              cx={layerXPositions[0]}
              cy={12 + i * 14}
              r={5}
              className="network-node"
              style={{ opacity: 0.4 + Math.random() * 0.4 }}
            />
          ))}
          <text x={layerXPositions[0]} y={98} className="layer-label">In</text>
        </g>

        {/* Hidden layers */}
        {Array.from({ length: numLayers }).map((_, layerIdx) => {
          const x = layerXPositions[1 + layerIdx];
          const activations = layerActivations[layerIdx] || [];
          
          return (
            <g key={`layer-${layerIdx}`} className="layer hidden-layer">
              {Array.from({ length: nodesPerLayer }).map((_, nodeIdx) => {
                const actIdx = Math.floor((nodeIdx / nodesPerLayer) * Math.max(activations.length, 1));
                const activation = activations[actIdx] ?? 0.5;
                
                return (
                  <motion.circle
                    key={`h${layerIdx}-${nodeIdx}`}
                    cx={x}
                    cy={12 + nodeIdx * 14}
                    r={5}
                    className="network-node active"
                    initial={{ opacity: 0.3 }}
                    animate={{ opacity: 0.3 + activation * 0.7 }}
                    transition={{ duration: 0.3 }}
                  />
                );
              })}
              <text x={x} y={98} className="layer-label">L{layerIdx + 1}</text>
            </g>
          );
        })}

        {/* Output layer */}
        <g className="layer output-layer">
          {Array.from({ length: nodesPerLayer }).map((_, i) => (
            <motion.circle
              key={`output-${i}`}
              cx={layerXPositions[3]}
              cy={12 + i * 14}
              r={5}
              className="network-node output"
              initial={{ opacity: 0.3 }}
              animate={{ opacity: 0.4 + Math.random() * 0.5 }}
              transition={{ duration: 0.5 }}
            />
          ))}
          <text x={layerXPositions[3]} y={98} className="layer-label">Out</text>
        </g>
      </svg>
    </div>
  );
}

function AttentionHeatmap({ attention, contextChars, isWordNN = false }: { attention: number[]; contextChars: string[]; isWordNN?: boolean }) {
  // Show last 20 items max (characters or words)
  const displayCount = Math.min(attention.length, isWordNN ? 10 : 20);
  const startIdx = attention.length - displayCount;
  const displayAttention = attention.slice(startIdx);
  const displayChars = contextChars.slice(startIdx);

  if (displayAttention.length === 0) {
    return <div className="attention-empty">No context yet</div>;
  }

  return (
    <div className="attention-heatmap">
      <div className="attention-bars">
        {displayAttention.map((weight, i) => {
          const label = displayChars[i] || '?';
          const displayLabel = isWordNN 
            ? (label.slice(0, 4) + (label.length > 4 ? '..' : ''))
            : (label === ' ' ? '␣' : label);
          
          return (
            <div key={i} className="attention-item" title={label}>
              <motion.div 
                className="attention-bar"
                initial={{ height: 0 }}
                animate={{ height: `${Math.min((weight || 0) * 200, 100)}%` }}
                transition={{ duration: 0.3 }}
                style={{
                  backgroundColor: `rgba(23, 23, 23, ${0.2 + (weight || 0) * 0.8})`
                }}
              />
              <span className="attention-char" style={isWordNN ? { fontSize: '8px' } : {}}>
                {displayLabel}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function CharProbsChart({ probs, isWordNN = false }: { probs: CharProb[]; isWordNN?: boolean }) {
  if (!probs || probs.length === 0) {
    return <div className="char-probs-empty">No predictions yet</div>;
  }
  
  return (
    <div className="char-probs-chart">
      {probs.map((cp, i) => {
        const label = cp?.char ?? '?';
        const displayLabel = isWordNN 
          ? (label.slice(0, 8) + (label.length > 8 ? '..' : ''))
          : (label === ' ' ? '␣' : label);
        const prob = cp?.prob ?? 0;
        
        return (
          <div key={i} className="prob-row" title={label}>
            <span className="prob-char" style={isWordNN ? { width: '60px', fontSize: '11px' } : {}}>
              {displayLabel}
            </span>
            <div className="prob-bar-container">
              <motion.div 
                className="prob-bar"
                initial={{ width: 0 }}
                animate={{ width: `${prob * 100}%` }}
                transition={{ duration: 0.3, delay: i * 0.02 }}
              />
            </div>
            <span className="prob-value">{(prob * 100).toFixed(1)}%</span>
          </div>
        );
      })}
    </div>
  );
}

