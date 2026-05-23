from llm_client import call_llm
import config
from datetime import datetime
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

logger = logging.getLogger(__name__)

# Thread-safe lock for transcript writing
transcript_lock = threading.Lock()

class DebateAgent:
    def __init__(self, model_spec, role, agent_id):
        self.model_spec = model_spec
        self.role = role  # 'pro' or 'con'
        self.agent_id = agent_id
    
    def generate_argument(self, report, factor, evidence, debate_history):
        """Generate an argument for the current debate round."""
        
        logger.debug(f"→ {self.agent_id} generating {self.role} argument...")
        
        system_prompt = f"""You are a {self.role.upper()} debater. Your role is to argue {self.role} the factor.

HARD RULES:
1. You MUST address opponent claims directly
2. Do NOT repeat previous arguments
3. Evidence augments reasoning - don't quote-dump
4. Ignoring rebuttals will be penalized
5. Max {config.MAX_ARGUMENT_LENGTH} words

Be sharp, logical, and responsive."""

        # Format evidence
        pro_evidence_text = "\n".join([f"- {e['text'][:200]}... (source: {e['source']})" 
                                        for e in evidence['pro'][:3]])
        con_evidence_text = "\n".join([f"- {e['text'][:200]}... (source: {e['source']})" 
                                        for e in evidence['con'][:3]])
        
        # Format debate history
        history_text = ""
        if debate_history:
            history_text = "\n\nDEBATE SO FAR:\n" + "\n".join(debate_history)
        
        prompt = f"""ORIGINAL REPORT:
{report[:500]}...

FACTOR TO DEBATE:
{factor}

PRO EVIDENCE:
{pro_evidence_text}

CON EVIDENCE:
{con_evidence_text}
{history_text}

YOUR TASK:
Make your {'supporting' if self.role == 'pro' else 'opposing'} argument for this factor.
Address the opponent's points if they exist.
Be concise and sharp. Max {config.MAX_ARGUMENT_LENGTH} words."""

        response = call_llm(self.model_spec, prompt, system_prompt)
        return response.strip()

def run_debate(report, factor, evidence, agent_count=5):
    """
    Run a multi-round debate for a single factor.
    Returns the debate transcript.
    agent_count: 3 or 5
    """
    logger.info(f"⚔️  Starting debate for factor: {factor} (Agents: {agent_count})")
    
    # Initialize agents based on count
    # 5 agents = 2 Pro + 2 Con (plus Judge external)
    # 3 agents = 1 Pro + 1 Con (plus Judge external)
    
    pro_agent_1 = DebateAgent(config.PRO_MODEL_1, 'pro', 'Pro-A')
    con_agent_1 = DebateAgent(config.CON_MODEL_1, 'con', 'Con-A')
    
    agents = [pro_agent_1, con_agent_1]
    
    if agent_count >= 5:
        pro_agent_2 = DebateAgent(config.PRO_MODEL_2, 'pro', 'Pro-B')
        con_agent_2 = DebateAgent(config.CON_MODEL_2, 'con', 'Con-B')
        agents.extend([pro_agent_2, con_agent_2])
        logger.info(f"✓ Initialized 4 debating agents (2 Pro, 2 Con)")
    else:
        logger.info(f"✓ Initialized 2 debating agents (1 Pro, 1 Con)")
    
    transcript = []
    transcript.append(f"DEBATE: {factor}")
    transcript.append(f"Started: {datetime.now().isoformat()}")
    transcript.append("=" * 80)
    
    debate_history = []
    
    for round_num in range(config.DEBATE_ROUNDS):
        logger.info(f"🎯 Round {round_num + 1}/{config.DEBATE_ROUNDS}")
        transcript.append(f"\n--- ROUND {round_num + 1} ---\n")
        
        # Generate arguments in parallel for this round
        round_results = []
        with ThreadPoolExecutor(max_workers=len(agents)) as executor:
            future_to_agent = {
                executor.submit(agent.generate_argument, report, factor, evidence, debate_history): agent
                for agent in agents
            }
            
            for future in as_completed(future_to_agent):
                agent = future_to_agent[future]
                try:
                    argument = future.result()
                    round_results.append((agent, argument))
                    logger.info(f"✓ {agent.agent_id} completed argument ({len(argument)} chars)")
                except Exception as e:
                    logger.error(f"❌ {agent.agent_id} failed: {e}")
                    round_results.append((agent, f"ERROR: {str(e)}"))
        
        # Sort results by agent order (Pro-A, Pro-B, Con-A, Con-B) and add to transcript
        # Define order priority
        order_priority = ['Pro-A', 'Pro-B', 'Con-A', 'Con-B']
        round_results.sort(key=lambda x: order_priority.index(x[0].agent_id) if x[0].agent_id in order_priority else 99)
        
        for agent, argument in round_results:
            turn_text = f"[{agent.agent_id}] ({agent.role.upper()}):\n{argument}\n"
            transcript.append(turn_text)
            debate_history.append(turn_text)
    
    transcript.append("\n" + "=" * 80)
    transcript.append(f"Ended: {datetime.now().isoformat()}")
    
    logger.info(f"✓ Debate complete: {len(debate_history)} total arguments")
    return "\n".join(transcript)
