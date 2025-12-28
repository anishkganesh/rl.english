"""Evolution engine with selection, mutation, and crossover logic."""

import random
from typing import List, Dict, Tuple
from agents import Agent, AgentGenome
from config import config
from memory import pattern_memory


def select_parents(agents: List[Agent], num_parents: int) -> List[Agent]:
    """Select top-performing agents as parents."""
    sorted_agents = sorted(agents, key=lambda a: a.score, reverse=True)
    return sorted_agents[:num_parents]


def mutate_genome(genome: AgentGenome, mutation_rate: float = None, strength: float = None) -> AgentGenome:
    """Create a mutated copy of a genome."""
    rate = mutation_rate if mutation_rate is not None else config.mutation_rate
    strength = strength if strength is not None else config.mutation_strength
    
    new_genome = genome.clone()
    
    # Mutate default probabilities
    for char in new_genome.default_probs:
        if random.random() < rate:
            # Add gaussian noise
            delta = random.gauss(0, strength)
            new_genome.default_probs[char] = max(0.01, new_genome.default_probs[char] + delta)
    
    # Normalize
    total = sum(new_genome.default_probs.values())
    new_genome.default_probs = {k: v / total for k, v in new_genome.default_probs.items()}
    
    # Mutate context-specific probabilities
    for context in new_genome.context_probs:
        for char in new_genome.context_probs[context]:
            if random.random() < rate:
                delta = random.gauss(0, strength)
                new_genome.context_probs[context][char] = max(
                    0.01, 
                    new_genome.context_probs[context][char] + delta
                )
        
        # Normalize
        total = sum(new_genome.context_probs[context].values())
        new_genome.context_probs[context] = {
            k: v / total for k, v in new_genome.context_probs[context].items()
        }
    
    return new_genome


def crossover_genomes(genome1: AgentGenome, genome2: AgentGenome) -> AgentGenome:
    """Create a child genome by combining two parent genomes."""
    child = AgentGenome()
    
    # Crossover default probabilities (average)
    for char in config.action_space:
        p1 = genome1.default_probs.get(char, 1.0 / len(config.action_space))
        p2 = genome2.default_probs.get(char, 1.0 / len(config.action_space))
        child.default_probs[char] = (p1 + p2) / 2
    
    # Normalize
    total = sum(child.default_probs.values())
    child.default_probs = {k: v / total for k, v in child.default_probs.items()}
    
    # Crossover context-specific probabilities
    all_contexts = set(genome1.context_probs.keys()) | set(genome2.context_probs.keys())
    
    for context in all_contexts:
        child.context_probs[context] = {}
        
        for char in config.action_space:
            p1 = genome1.context_probs.get(context, {}).get(char, genome1.default_probs.get(char, 0))
            p2 = genome2.context_probs.get(context, {}).get(char, genome2.default_probs.get(char, 0))
            
            # Random blend
            blend = random.random()
            child.context_probs[context][char] = blend * p1 + (1 - blend) * p2
        
        # Normalize
        total = sum(child.context_probs[context].values())
        if total > 0:
            child.context_probs[context] = {
                k: v / total for k, v in child.context_probs[context].items()
            }
    
    return child


def learn_from_success(agent: Agent):
    """Update an agent's genome based on its successful output."""
    if agent.score < 0.2:
        return
    
    text = agent.typed_text
    if len(text) < 2:
        return
    
    # Reinforce the patterns that led to this output
    for i in range(len(text)):
        context = text[max(0, i - config.context_length):i]
        char = text[i]
        
        if context not in agent.genome.context_probs:
            agent.genome.context_probs[context] = agent.genome.default_probs.copy()
        
        # Increase probability of this character given context
        boost = agent.score * 0.1  # Proportional to score
        agent.genome.context_probs[context][char] = min(
            0.9,
            agent.genome.context_probs[context].get(char, 0) + boost
        )
        
        # Normalize
        total = sum(agent.genome.context_probs[context].values())
        agent.genome.context_probs[context] = {
            k: v / total for k, v in agent.genome.context_probs[context].items()
        }


def evolve_generation(agents: List[Agent]) -> List[Agent]:
    """Create the next generation of agents through evolution."""
    num_agents = len(agents)
    
    # Sort by score
    sorted_agents = sorted(agents, key=lambda a: a.score, reverse=True)
    
    # Store successful patterns in memory
    for agent in sorted_agents[:3]:  # Top 3
        pattern_memory.add_pattern(agent.typed_text, agent.score)
        learn_from_success(agent)
    
    new_agents: List[Agent] = []
    
    # Elitism: Keep top performers unchanged (including vocabulary)
    num_elite = max(1, int(num_agents * config.elitism_ratio))
    for i in range(num_elite):
        elite = Agent(
            genome=sorted_agents[i].genome.clone(),
            exploration_rate=sorted_agents[i].exploration_rate,
            vocabulary=sorted_agents[i].vocabulary.copy()  # Inherit vocabulary
        )
        elite.decay_exploration()
        new_agents.append(elite)
    
    # Crossover: Create some through breeding
    num_crossover = int(num_agents * config.crossover_ratio)
    parents = sorted_agents[:max(2, num_elite)]
    
    for _ in range(num_crossover):
        p1, p2 = random.sample(parents, 2)
        child_genome = crossover_genomes(p1.genome, p2.genome)
        child_genome = mutate_genome(child_genome, mutation_rate=config.mutation_rate / 2)
        
        # Merge vocabularies from both parents
        merged_vocab = p1.vocabulary.union(p2.vocabulary)
        
        child = Agent(
            genome=child_genome,
            exploration_rate=(p1.exploration_rate + p2.exploration_rate) / 2,
            vocabulary=merged_vocab  # Inherit merged vocabulary
        )
        child.decay_exploration()
        new_agents.append(child)
    
    # Mutation: Fill the rest with mutated copies of top performers
    while len(new_agents) < num_agents:
        parent = random.choice(parents)
        mutated_genome = mutate_genome(parent.genome)
        
        child = Agent(
            genome=mutated_genome,
            exploration_rate=parent.exploration_rate,
            vocabulary=parent.vocabulary.copy()  # Inherit vocabulary
        )
        child.decay_exploration()
        new_agents.append(child)
    
    return new_agents


def get_generation_stats(agents: List[Agent]) -> Dict:
    """Get statistics for the current generation."""
    if not agents:
        return {}
    
    scores = [a.score for a in agents]
    sorted_agents = sorted(agents, key=lambda a: a.score, reverse=True)
    
    return {
        "best_score": max(scores),
        "avg_score": sum(scores) / len(scores),
        "worst_score": min(scores),
        "best_text": sorted_agents[0].typed_text if sorted_agents else "",
        "best_agent_id": sorted_agents[0].id if sorted_agents else None,
        "avg_exploration": sum(a.exploration_rate for a in agents) / len(agents)
    }

