"""
crew.py — Multi-agent orchestration using a lightweight Crew framework.

Defines 4 AI-aware agents (profiler, transformer, quality_checker, documenter)
with roles, goals, and backstories. A Crew coordinates them in sequence,
passing each agent's output to the next.

This is a custom implementation inspired by CrewAI's design pattern.
CrewAI itself requires Python <3.14, so we built a compatible version
that demonstrates the same multi-agent orchestration concepts:

    - Agent: has a role, goal, backstory, and a callable task function
    - Task: wraps a function call with a description and assigned agent
    - Crew: executes tasks sequentially, passing context between them

The key difference from direct agent calls (Phase 3) is that the Crew
adds AI-enhanced logging: each agent uses Ollama to generate a natural
language summary of what it did, which gets included in the final report.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

import pandas as pd

from core.llm import query_ollama, is_ollama_available
from core.utils import get_logger, save_json
from core.config import REPORTS_DIR
from core.database import DuckDBManager
from core.report import ReportGenerator
from agents.profiler import ProfilerAgent
from agents.transformer import TransformerAgent
from agents.quality_checker import QualityCheckerAgent
from agents.documenter import DocumenterAgent

logger = get_logger(__name__)


@dataclass
class Agent:
    """
    Represents an AI agent with a defined role and purpose.

    Each agent has a persona (role, goal, backstory) that describes
    what it does in the pipeline. The execute_fn is the actual function
    that does the work.

    Attributes:
        name:       Short identifier (e.g., "profiler").
        role:       Job title (e.g., "Senior Data Analyst").
        goal:       What this agent is trying to achieve.
        backstory:  Background that shapes the agent's approach.
        execute_fn: The callable that performs the actual work.
                    Set to None initially, assigned when tasks are created.
    """
    name: str
    role: str
    goal: str
    backstory: str
    execute_fn: Callable | None = None


@dataclass
class Task:
    """
    A unit of work assigned to an agent.

    Attributes:
        description: What this task does (human-readable).
        agent:       The Agent responsible for executing it.
        execute_fn:  The actual function to call.
        result:      Stores the output after execution.
    """
    description: str
    agent: Agent
    execute_fn: Callable
    result: Any = None


@dataclass
class CrewResult:
    """
    Final output from a Crew run, containing all agent results.

    Attributes:
        dataset_name:    Name of the dataset processed.
        started_at:      When the crew started.
        completed_at:    When the crew finished.
        agent_summaries: List of per-agent summaries with AI commentary.
        task_results:    Dict mapping task description to its output.
    """
    dataset_name: str
    started_at: str
    completed_at: str = ""
    agent_summaries: list[dict] = field(default_factory=list)
    task_results: dict = field(default_factory=dict)


class Crew:
    """
    Orchestrates multiple agents executing tasks in sequence.

    Similar to CrewAI's Crew class: takes a list of agents and tasks,
    runs them in order, and collects results. Adds AI-generated summaries
    if Ollama is available.

    Usage:
        crew = Crew(agents=[...], tasks=[...])
        result = crew.kickoff()
    """

    def __init__(
        self,
        agents: list[Agent],
        tasks: list[Task],
        dataset_name: str = "unknown",
    ) -> None:
        """
        Initialize the crew with agents and tasks.

        Args:
            agents:       List of Agent instances.
            tasks:        List of Task instances (executed in order).
            dataset_name: Name of the dataset being processed.
        """
        self.agents = agents
        self.tasks = tasks
        self.dataset_name = dataset_name
        self._ollama_available = is_ollama_available()

    def kickoff(self) -> CrewResult:
        """
        Execute all tasks in sequence and return the combined result.

        Each task's execute_fn is called in order. After each task,
        the agent optionally generates an AI summary of what it did.

        Returns:
            CrewResult with all task outputs and agent summaries.
        """
        result = CrewResult(
            dataset_name=self.dataset_name,
            started_at=datetime.now().isoformat(),
        )

        logger.info("=" * 60)
        logger.info(f"Crew started: processing '{self.dataset_name}'")
        logger.info(f"Agents: {[a.name for a in self.agents]}")
        logger.info(f"Tasks: {len(self.tasks)}")
        logger.info("=" * 60)

        for i, task in enumerate(self.tasks, 1):
            agent = task.agent
            logger.info(
                f"\n[Task {i}/{len(self.tasks)}] "
                f"{agent.role}: {task.description}"
            )

            # Execute the task
            task.result = task.execute_fn()

            # Store the result
            result.task_results[task.description] = task.result

            # Generate AI summary if Ollama is available
            summary = self._generate_agent_summary(agent, task)
            result.agent_summaries.append(summary)

            logger.info(
                f"  Agent '{agent.name}' completed: {summary['summary']}"
            )

        result.completed_at = datetime.now().isoformat()

        logger.info("\n" + "=" * 60)
        logger.info("Crew completed all tasks")
        logger.info("=" * 60)

        return result

    def _generate_agent_summary(self, agent: Agent, task: Task) -> dict:
        """
        Generate a summary for what an agent did, optionally using AI.

        If Ollama is available, asks the LLM to summarize the agent's work.
        Otherwise, uses a simple rule-based summary.

        Args:
            agent: The agent that executed the task.
            task:  The completed task with its result.

        Returns:
            Dict with agent info and summary text.
        """
        # Build a basic summary from the task result
        basic_summary = self._basic_summary(agent, task)

        ai_summary = None
        if self._ollama_available:
            ai_summary = self._ai_summary(agent, task, basic_summary)

        return {
            "agent_name": agent.name,
            "agent_role": agent.role,
            "task": task.description,
            "summary": ai_summary or basic_summary,
            "summary_source": "ai" if ai_summary else "rule-based",
            "timestamp": datetime.now().isoformat(),
        }

    def _basic_summary(self, agent: Agent, task: Task) -> str:
        """
        Create a simple rule-based summary of a completed task.

        Args:
            agent: The agent that ran.
            task:  The completed task.

        Returns:
            A short summary string.
        """
        result = task.result

        if agent.name == "profiler" and isinstance(result, dict):
            overview = result.get("overview", {})
            return (
                f"Profiled {overview.get('row_count', '?')} rows, "
                f"{overview.get('completeness_score', '?')}% complete, "
                f"{len(result.get('quality_issues', []))} issues found"
            )
        elif agent.name == "transformer" and isinstance(result, pd.DataFrame):
            return f"Transformed data to {len(result)} rows, {len(result.columns)} columns"
        elif agent.name == "quality_checker" and isinstance(result, dict):
            return (
                f"Quality score: {result.get('score', '?')}% "
                f"(Grade {result.get('grade', '?')}), "
                f"{result.get('checks_passed', '?')}/{result.get('total_checks', '?')} passed"
            )
        elif agent.name == "documenter" and isinstance(result, dict):
            return (
                f"Generated docs: {len(result.get('data_dictionary', []))} columns, "
                f"{len(result.get('usage_notes', []))} usage notes"
            )
        else:
            return f"Task completed by {agent.role}"

    def _ai_summary(
        self, agent: Agent, task: Task, basic_summary: str
    ) -> str | None:
        """
        Ask the LLM to write a natural language summary of the agent's work.

        Args:
            agent:         The agent that ran.
            task:          The completed task.
            basic_summary: The rule-based summary (used as context for the LLM).

        Returns:
            AI-generated summary string, or None if it failed.
        """
        prompt = (
            f"You are a {agent.role}. {agent.backstory} "
            f"You just completed this task: '{task.description}'. "
            f"Here are the results: {basic_summary}. "
            f"Write a one-sentence professional summary of what you found. "
            f"Reply with only the summary."
        )

        result = query_ollama(prompt)
        if result:
            # Take just the first sentence
            first_sentence = result.split(".")[0].strip()
            if len(first_sentence) > 10:
                return first_sentence + "."
        return None


# ---------------------------------------------------------------------------
# Pre-built agent definitions
# ---------------------------------------------------------------------------

def _create_agents() -> dict[str, Agent]:
    """
    Create the 4 pipeline agents with their personas.

    Returns:
        Dict mapping agent name to Agent instance.
    """
    return {
        "profiler": Agent(
            name="profiler",
            role="Senior Data Analyst",
            goal="Thoroughly profile datasets to uncover data quality issues, "
                 "statistical patterns, and anomalies before transformation",
            backstory="You have 10 years of experience in healthcare data analytics. "
                      "You've worked with WHO, FDA, and CDC datasets extensively. "
                      "You know that data quality issues caught early save weeks of "
                      "debugging downstream.",
        ),
        "transformer": Agent(
            name="transformer",
            role="Data Engineer",
            goal="Clean, standardize, and transform raw healthcare data into "
                 "analysis-ready format while maintaining a complete audit trail",
            backstory="You specialize in ETL pipelines for healthcare data. "
                      "You've standardized data from dozens of international health "
                      "organizations. You believe every transformation should be "
                      "logged and reversible.",
        ),
        "quality_checker": Agent(
            name="quality_checker",
            role="Quality Assurance Specialist",
            goal="Rigorously verify data quality against defined thresholds "
                 "and produce a graded scorecard with actionable findings",
            backstory="You've built data quality frameworks for hospital systems "
                      "and pharmaceutical companies. You know that a single bad "
                      "data point in healthcare can have real consequences. You "
                      "grade datasets honestly — no inflated scores.",
        ),
        "documenter": Agent(
            name="documenter",
            role="Technical Documentation Lead",
            goal="Generate comprehensive, accurate documentation that makes "
                 "datasets self-describing and accessible to any data consumer",
            backstory="You create data documentation that analysts actually read. "
                      "You combine automated profiling with domain expertise to "
                      "write data dictionaries, lineage docs, and usage notes "
                      "that save teams hours of guesswork.",
        ),
    }


def run_crew(
    df: pd.DataFrame,
    dataset_name: str,
    source_metadata: dict | None = None,
) -> dict:
    """
    Run the full pipeline using multi-agent Crew orchestration.

    This is the main entry point for Crew mode. It creates agents,
    defines tasks, and executes them in sequence via the Crew.

    Args:
        df:              Raw DataFrame to process.
        dataset_name:    Name for labeling outputs.
        source_metadata: Metadata from the ingestion source.

    Returns:
        Dict with all pipeline results (profile, clean_df, scorecard,
        docs, crew_result).
    """
    agents = _create_agents()

    # Create the agent instances that do the actual work
    profiler = ProfilerAgent()
    transformer = TransformerAgent()
    checker = QualityCheckerAgent()
    documenter = DocumenterAgent()

    # These variables will be populated as tasks execute.
    # Each task's execute_fn is a closure that captures the variables
    # it needs and updates the shared state.
    pipeline_state: dict[str, Any] = {
        "raw_df": df,
        "profile": None,
        "clean_df": None,
        "transform_log": None,
        "scorecard": None,
        "docs": None,
    }

    # --- Define tasks as closures that read/write pipeline_state ---

    def profile_task() -> dict:
        """Profile the raw data."""
        result = profiler.run(pipeline_state["raw_df"], dataset_name)
        pipeline_state["profile"] = result
        return result

    def transform_task() -> pd.DataFrame:
        """Transform the raw data."""
        result = transformer.run(pipeline_state["raw_df"], dataset_name)
        pipeline_state["clean_df"] = result
        pipeline_state["transform_log"] = transformer.get_transform_summary()
        return result

    def quality_task() -> dict:
        """Run quality checks on transformed data."""
        result = checker.run(pipeline_state["clean_df"], dataset_name)
        pipeline_state["scorecard"] = result
        return result

    def document_task() -> dict:
        """Generate documentation."""
        result = documenter.run(
            pipeline_state["clean_df"],
            dataset_name,
            source_metadata=source_metadata,
            profile_data=pipeline_state["profile"],
            transform_log=pipeline_state["transform_log"],
            quality_scorecard=pipeline_state["scorecard"],
        )
        pipeline_state["docs"] = result
        return result

    # --- Build tasks linked to agents ---

    tasks = [
        Task(
            description="Profile raw dataset for statistics, outliers, and quality issues",
            agent=agents["profiler"],
            execute_fn=profile_task,
        ),
        Task(
            description="Clean and standardize data with full audit trail",
            agent=agents["transformer"],
            execute_fn=transform_task,
        ),
        Task(
            description="Run quality checks and produce graded scorecard",
            agent=agents["quality_checker"],
            execute_fn=quality_task,
        ),
        Task(
            description="Generate comprehensive data documentation",
            agent=agents["documenter"],
            execute_fn=document_task,
        ),
    ]

    # --- Execute the crew ---

    crew = Crew(
        agents=list(agents.values()),
        tasks=tasks,
        dataset_name=dataset_name,
    )
    crew_result = crew.kickoff()

    # Save the crew execution report
    crew_report = {
        "dataset_name": crew_result.dataset_name,
        "started_at": crew_result.started_at,
        "completed_at": crew_result.completed_at,
        "agent_summaries": crew_result.agent_summaries,
    }
    report_path = REPORTS_DIR / f"crew_report_{dataset_name}.json"
    save_json(crew_report, report_path)
    logger.info(f"Crew report saved to: {report_path}")

    # Generate HTML report from all pipeline outputs
    report_gen = ReportGenerator()
    profile_data = pipeline_state["profile"] or {}
    docs_data = pipeline_state["docs"] or {}
    scorecard_data = pipeline_state["scorecard"] or {}
    report_path = report_gen.generate(
        dataset_name=dataset_name,
        overview=profile_data.get("overview", {}),
        scorecard=scorecard_data,
        data_dictionary=docs_data.get("data_dictionary", []),
        column_profiles=profile_data.get("columns", {}),
        transform_log=pipeline_state["transform_log"] or [],
        quality_issues=profile_data.get("quality_issues", []),
        correlations=profile_data.get("correlations", []),
        usage_notes=docs_data.get("usage_notes", []),
    )
    logger.info(f"HTML report: {report_path}")

    # Load into DuckDB
    import re
    table_name = re.sub(r"[^a-zA-Z0-9]", "_", dataset_name).strip("_").lower()
    table_name = re.sub(r"_+", "_", table_name)

    with DuckDBManager() as db:
        db.load_dataframe(pipeline_state["clean_df"], table_name)
        tables = db.list_tables()
        logger.info(f"DuckDB tables: {tables}")

    return {
        "profile": pipeline_state["profile"],
        "clean_df": pipeline_state["clean_df"],
        "scorecard": pipeline_state["scorecard"],
        "docs": pipeline_state["docs"],
        "crew_result": crew_result,
    }
