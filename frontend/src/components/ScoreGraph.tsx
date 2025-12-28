import {
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Area,
  ComposedChart,
} from 'recharts';
import type { GenerationStats } from '../types';

interface ScoreGraphProps {
  history: GenerationStats[];
}

export function ScoreGraph({ history }: ScoreGraphProps) {
  // Take last 50 generations for display
  const data = history.slice(-50).map(h => ({
    gen: h.generation,
    best: h.best_score ?? h.max ?? 0,
    avg: h.avg_score ?? h.mean ?? 0,
  }));

  if (data.length === 0) {
    return (
      <div className="score-graph">
        <div className="empty-state" style={{ height: '100%' }}>
          <div style={{ fontSize: '0.8rem' }}>No data yet</div>
        </div>
      </div>
    );
  }

  return (
    <div className="score-graph">
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={data} margin={{ top: 5, right: 5, left: -20, bottom: 5 }}>
          <defs>
            <linearGradient id="bestGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#171717" stopOpacity={0.1} />
              <stop offset="100%" stopColor="#171717" stopOpacity={0} />
            </linearGradient>
          </defs>
          <XAxis
            dataKey="gen"
            stroke="#606070"
            fontSize={10}
            tickLine={false}
            axisLine={false}
          />
          <YAxis
            domain={[0, 1]}
            stroke="#606070"
            fontSize={10}
            tickLine={false}
            axisLine={false}
            ticks={[0, 0.25, 0.5, 0.75, 1]}
          />
          <Tooltip
            contentStyle={{
              background: '#ffffff',
              border: '1px solid #e5e5e5',
              borderRadius: '8px',
              fontSize: '12px',
              boxShadow: '0 4px 6px -1px rgba(0,0,0,0.1)',
            }}
            labelStyle={{ color: '#525252' }}
            formatter={(value: number, name: string) => [
              value.toFixed(3),
              name === 'best' ? 'Best' : 'Avg',
            ]}
            labelFormatter={(label) => `Gen ${label}`}
          />
          <Area
            type="monotone"
            dataKey="best"
            stroke="transparent"
            fill="url(#bestGradient)"
          />
          <Line
            type="monotone"
            dataKey="best"
            stroke="#171717"
            strokeWidth={2}
            dot={false}
            activeDot={{ r: 4, fill: '#171717' }}
          />
          <Line
            type="monotone"
            dataKey="avg"
            stroke="#a3a3a3"
            strokeWidth={1}
            strokeDasharray="3 3"
            dot={false}
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}

