"""
Sequential Multi-Agent Task Workflow
====================================
Orchestrates a sequential workflow between:
1. Technical Analyst Agent: Retrieves specific technical data and specifications from the database.
2. Reviewer Agent: Takes the raw technical data from Agent 1 and writes a detailed, engaging product review or comparison.

Workflow Architecture:
----------------------
[User Query]
     │
     ▼ STAGE 1: Technical Analyst Agent
[Retrieves Specs from Relational DB + Vector Knowledge]
     │
     ▼ (Passes raw data & technical report)
[STAGE 2: Reviewer Agent]
[Generates engaging consumer review, pros/cons, & buying verdict]
     │
     ▼
[Final Response: Technical Specs + Engaging Consumer Review]
"""

import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional

# Import Agent 1 & Agent 2
from technical_analyst_agent import TechnicalAnalystAgent, get_technical_analyst_agent
from reviewer_agent import ReviewerAgent, get_reviewer_agent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("SequentialWorkflow")


class SequentialPhoneWorkflow:
    """
    Sequential multi-agent task orchestrator for Samsung Phone queries and reviews.
    """

    def __init__(self):
        self.analyst_agent = get_technical_analyst_agent()
        self.reviewer_agent = get_reviewer_agent()
        logger.info("Sequential Multi-Agent Workflow initialized: [Technical Analyst] -> [Reviewer].")

    def run(self, user_query: str) -> Dict[str, Any]:
        """
        Executes the sequential multi-agent workflow:
        Stage 1: Technical Analyst Agent retrieves technical metrics.
        Stage 2: Reviewer Agent crafts engaging review using the technical output.
        """
        start_time = time.time()
        logger.info(">>> STEP 1: Invoking Technical Analyst Agent for query: '%s' ...", user_query)

        # STAGE 1: Technical Analyst Agent
        tech_output = self.analyst_agent.analyze(user_query)
        logger.info(">>> STEP 1 COMPLETE: Technical Analyst retrieved %d sources.", len(tech_output.get("sources", [])))

        # STAGE 2: Reviewer Agent (receives Stage 1 output directly)
        logger.info(">>> STEP 2: Passing technical data to Reviewer Agent ...")
        review_output = self.reviewer_agent.review(user_query, tech_output)
        logger.info(">>> STEP 2 COMPLETE: Reviewer Agent crafted '%s'.", review_output.get("review_type"))

        elapsed = round(time.time() - start_time, 3)

        # Combine both stages into a unified, publication-ready final response
        final_response = (
            f"{review_output['review']}\n\n"
            f"---\n\n"
            f"### 📑 Appendix: Verified Technical Specifications (Stage 1 Data)\n"
            f"{tech_output['report']}\n"
        )

        return {
            "workflow": "Sequential Multi-Agent Pipeline",
            "query": user_query,
            "status": "success",
            "execution_time_seconds": elapsed,
            "sources": tech_output.get("sources", []),
            "stage_1_technical_data": {
                "agent": tech_output.get("agent"),
                "report": tech_output.get("report"),
                "structured_data": tech_output.get("structured_data")
            },
            "stage_2_review": {
                "agent": review_output.get("agent"),
                "review_type": review_output.get("review_type"),
                "review_text": review_output.get("review")
            },
            "final_response": final_response
        }


# Global workflow instance
_global_workflow: Optional[SequentialPhoneWorkflow] = None


def run_sequential_workflow(user_query: str) -> Dict[str, Any]:
    """Public helper function to execute the sequential workflow."""
    global _global_workflow
    if _global_workflow is None:
        _global_workflow = SequentialPhoneWorkflow()
    return _global_workflow.run(user_query)


# ==============================================================================
# CLI Demonstration
# ==============================================================================

def main():
    print("=" * 85)
    print(" SEQUENTIAL MULTI-AGENT WORKFLOW DEMO")
    print(" [Stage 1: Technical Analyst Agent] ──► [Stage 2: Reviewer Agent]")
    print("=" * 85)

    workflow = SequentialPhoneWorkflow()

    test_queries = [
        "What are the camera specs of the Samsung Galaxy S23?",
        "Compare the technical specifications and user experience between Samsung Galaxy S23 and S24 Ultra",
        "Which Samsung Galaxy phones have 5000 mAh battery?"
    ]

    for idx, q in enumerate(test_queries, 1):
        print(f"\n{'='*35} WORKFLOW RUN {idx} {'='*35}")
        print(f"User Query: {q}")
        print("-" * 80)

        result = workflow.run(q)

        print(f"Elapsed Time: {result['execution_time_seconds']}s | Sources: {', '.join(result['sources'])}")
        print("\n" + result["final_response"])
        print("-" * 80)


if __name__ == "__main__":
    main()
