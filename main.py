"""
Samsung Phone Query and Review System - FastAPI Backend
========================================================
Wraps the RAG chatbot and CrewAI multi-agent system into a production-grade FastAPI service.

Routing Logic (/query):
- Direct Specifications: Routed to the Simple RAG Pipeline (FAISS + Hugging Face embeddings).
- Reviews / Comparisons: Routed to the CrewAI Multi-Agent System (Technical Analyst + Reviewer Agents).
"""

import logging
import os
import re
import sys
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Literal, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Core modules from previous steps
from db_manager import Base, CameraLens, Phone, PhoneSpecification, get_database_engine, log_chat_query, ChatQueryLog
from rag_pipeline import query_phone_specs
from sequential_workflow import SequentialPhoneWorkflow, run_sequential_workflow
from technical_analyst_agent import get_technical_analyst_agent
from reviewer_agent import get_reviewer_agent

# CrewAI imports
try:
    from crewai import Agent, Crew, Process, Task
    from crewai.tools import tool as crewai_tool
    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("SamsungFastAPI")


# ==============================================================================
# 1. Pydantic Request & Response Schemas
# ==============================================================================

class QueryRequest(BaseModel):
    """Payload for the /query endpoint."""
    question: str = Field(
        ...,
        description="The natural language question about Samsung Galaxy phones",
        examples=["What are the camera specs of the Samsung Galaxy S23?", "Compare Samsung Galaxy S23 and S24 Ultra in detail"]
    )
    force_route: Optional[Literal["auto", "rag", "crewai"]] = Field(
        default="auto",
        description="Optional routing override: 'auto' (smart intent detection), 'rag' (force RAG specs), 'crewai' (force multi-agent review)"
    )


class QueryResponse(BaseModel):
    """Clean JSON response format returned by the /query endpoint."""
    question: str
    route: str
    routed_reason: str
    engine: str
    response: str
    source_phones: List[str]
    execution_time_seconds: float
    metadata: Dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    database: str
    total_phones_in_db: int
    faiss_vector_index: str
    crewai_supported: bool


# ==============================================================================
# 2. Smart Query Intent Router
# ==============================================================================

class QueryIntentRouter:
    """
    Analyzes user question semantics to intelligently route between:
    1. Simple RAG Pipeline (for direct hardware and spec lookups)
    2. CrewAI Multi-Agent System (for reviews, comparisons, and purchasing advice)
    """

    REVIEW_KEYWORDS = [
        r"\breview\b",
        r"\breviews\b",
        r"\bcompare\b",
        r"\bcomparison\b",
        r"\bversus\b",
        r"\bvs\b",
        r"\bdifference\b",
        r"\bdifferences\b",
        r"\bbetter\b",
        r"\bwhich (?:one|phone|is better)\b",
        r"\brecommend\b",
        r"\brecommendation\b",
        r"\bpros\b",
        r"\bcons\b",
        r"\bworth\b",
        r"\bbuying guide\b",
        r"\bshould i buy\b",
        r"\bopinion\b",
        r"\bevaluation\b",
        r"\bverdict\b",
        r"\bexperience\b",
        r"\bshowdown\b"
    ]

    DIRECT_SPEC_KEYWORDS = [
        r"\bcamera\b",
        r"\bspecs?\b",
        r"\bspecifications?\b",
        r"\bbattery\b",
        r"\bcharging\b",
        r"\bprocessor\b",
        r"\bchipset\b",
        r"\bcpu\b",
        r"\bgpu\b",
        r"\bdisplay\b",
        r"\bscreen\b",
        r"\bresolution\b",
        r"\bweight\b",
        r"\bdimensions?\b",
        r"\bpixels?\b",
        r"\bmah\b",
        r"\bmp\b",
        r"\bram\b",
        r"\bstorage\b"
    ]

    @classmethod
    def route(cls, question: str, force_route: str = "auto") -> tuple[str, str]:
        """
        Determines routing destination:
        Returns: (route_name, reason)
        """
        if force_route == "rag":
            return "simple_rag", "Forced by client request parameter (force_route='rag')"
        if force_route == "crewai":
            return "crewai_multi_agent", "Forced by client request parameter (force_route='crewai')"

        q_lower = question.lower()

        # Check for review/comparison intent
        for pat in cls.REVIEW_KEYWORDS:
            if re.search(pat, q_lower):
                return (
                    "crewai_multi_agent",
                    f"Matched review/comparison intent keyword pattern: '{pat}'"
                )

        # Check for multiple devices in single query (e.g. 's23 and s24')
        models_found = len(re.findall(r"\bs\d{2}\b|\ba\d{2}\b", q_lower))
        if models_found >= 2:
            return (
                "crewai_multi_agent",
                f"Multiple target phones detected ({models_found} models) - routed to comparative agent workflow"
            )

        # Direct spec lookup
        for pat in cls.DIRECT_SPEC_KEYWORDS:
            if re.search(pat, q_lower):
                return (
                    "simple_rag",
                    f"Matched direct specification inquiry pattern: '{pat}'"
                )

        # Default fallback: RAG lookup
        return "simple_rag", "Default semantic specification lookup"


# ==============================================================================
# 3. CrewAI Multi-Agent System Wrapper
# ==============================================================================

class CrewAIMultiAgentService:
    """
    CrewAI Multi-Agent System orchestration:
    - Technical Analyst Agent: Extracts verified metrics from the database.
    - Reviewer Agent: Produces engaging, structured product reviews and comparisons.
    """

    @classmethod
    def execute_crew(cls, user_question: str) -> Dict[str, Any]:
        """
        Executes the Technical Analyst ➔ Reviewer Agent pipeline.
        Uses CrewAI architecture with seamless local execution guarantee.
        """
        logger.info("[CrewAI Multi-Agent Service] Initiating workflow for: '%s'", user_question)

        # We execute the sequential workflow which encapsulates the two specialized agents:
        # Agent 1: Senior Technical Analyst
        # Agent 2: Consumer Reviewer & Tech Journalist
        workflow_result = run_sequential_workflow(user_question)

        return {
            "engine": "CrewAI Multi-Agent System (Technical Analyst + Reviewer)",
            "response": workflow_result.get("final_response", ""),
            "source_phones": workflow_result.get("sources", []),
            "stage_1_analyst": workflow_result.get("stage_1_technical_data", {}),
            "stage_2_reviewer": workflow_result.get("stage_2_review", {})
        }


# ==============================================================================
# 4. FastAPI Application Setup & Lifespan
# ==============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup & Shutdown lifecycle event handler."""
    logger.info("Initializing Samsung Phone Database & FAISS Vector Store on startup...")
    engine, db_type = get_database_engine()
    app.state.db_engine = engine
    app.state.db_type = db_type

    # Verify RAG pipeline vector store is primed
    try:
        from rag_pipeline import VectorStoreManager
        v_mgr = VectorStoreManager()
        v_mgr.initialize_vector_store()
        app.state.vector_ready = True
        logger.info("FAISS Vector Index initialized and ready.")
    except Exception as e:
        logger.warning("Vector index warming warning: %s", e)
        app.state.vector_ready = False

    logger.info("Samsung Phone Query & Review FastAPI Backend is LIVE!")
    yield
    logger.info("Shutting down FastAPI service...")


app = FastAPI(
    title="Samsung Phone Query & Review System API",
    description=(
        "Backend REST API for Samsung Galaxy specifications, RAG chatbot, "
        "and CrewAI multi-agent review system. Routes queries automatically "
        "between direct RAG specs lookup and multi-agent review workflows."
    ),
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for seamless integration with any frontend / port
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================================================================
# 5. Core API Endpoints
# ==============================================================================

@app.get("/", tags=["System"])
async def root():
    """Root endpoint welcoming clients and describing API capabilities."""
    return {
        "service": "Samsung Phone Query and Review System API",
        "version": "1.0.0",
        "endpoints": {
            "POST /query": "Main question routing endpoint (RAG vs CrewAI Multi-Agent)",
            "GET /query": "Convenience browser query endpoint (?question=...)",
            "GET /phones": "List all 15 Samsung Galaxy models in database",
            "GET /health": "System health and database connectivity status",
            "GET /docs": "Interactive Swagger UI documentation"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Health check endpoint providing status of Database and FAISS Vector Store."""
    from sqlalchemy.orm import Session
    engine, db_type = get_database_engine()
    total_phones = 0
    with Session(engine) as session:
        total_phones = session.query(Phone).count()

    faiss_exists = os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "faiss_index", "index.faiss"))

    return HealthResponse(
        status="healthy",
        database=db_type,
        total_phones_in_db=total_phones,
        faiss_vector_index="loaded" if faiss_exists else "not_indexed",
        crewai_supported=CREWAI_AVAILABLE
    )


@app.post("/query", response_model=QueryResponse, tags=["Query & Review"])
async def process_query(payload: QueryRequest):
    """
    Main Query Endpoint:
    Accepts a user question and dynamically routes it:
    - Direct Specs ──► Simple RAG Pipeline (FAISS + Hugging Face)
    - Reviews / Comparisons ──► CrewAI Multi-Agent System (Technical Analyst + Reviewer)
    """
    question = payload.question.strip()
    if not question:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The 'question' field cannot be empty."
        )

    start_time = time.time()
    route, reason = QueryIntentRouter.route(question, force_route=payload.force_route or "auto")
    logger.info("Incoming query: '%s' | Routed to: '%s' | Reason: %s", question, route, reason)

    response_text = ""
    source_phones: List[str] = []
    engine_name = ""
    metadata: Dict[str, Any] = {"route_decision_reason": reason}

    try:
        # Route 1: Simple RAG Pipeline (Direct specs lookup)
        if route == "simple_rag":
            rag_output = query_phone_specs(question)
            response_text = rag_output.get("answer", "No specifications found.")
            source_phones = rag_output.get("source_phones", [])
            engine_name = "LangChain + FAISS + Hugging Face Embeddings (Simple RAG)"
            metadata["rag_matches"] = rag_output.get("retrieved_matches", [])

        # Route 2: CrewAI Multi-Agent System (Reviews & Comparisons)
        else:
            crew_output = CrewAIMultiAgentService.execute_crew(question)
            response_text = crew_output.get("response", "")
            source_phones = crew_output.get("source_phones", [])
            engine_name = crew_output.get("engine", "CrewAI Multi-Agent System")
            metadata["stage_1_analyst"] = crew_output.get("stage_1_analyst", {})
            metadata["stage_2_reviewer"] = crew_output.get("stage_2_reviewer", {})

    except Exception as exc:
        logger.error("Error executing query through %s: %s", route, exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error executing query through {route}: {str(exc)}"
        )

    elapsed = round(time.time() - start_time, 4)

    # Persist question and response in PostgreSQL database
    log_chat_query(
        question=question,
        route=route,
        response=response_text,
        source_phones=source_phones,
        execution_time=elapsed
    )

    return QueryResponse(
        question=question,
        route=route,
        routed_reason=reason,
        engine=engine_name,
        response=response_text,
        source_phones=source_phones,
        execution_time_seconds=elapsed,
        metadata=metadata
    )


@app.get("/query", response_model=QueryResponse, tags=["Query & Review"])
async def query_get_endpoint(
    question: str = Query(..., description="The user question about Samsung phones"),
    force_route: Optional[Literal["auto", "rag", "crewai"]] = Query("auto", description="Routing override")
):
    """Convenience GET endpoint to test /query directly from a browser bar."""
    return await process_query(QueryRequest(question=question, force_route=force_route))


@app.get("/phones", tags=["Database Catalog"])
async def list_phones():
    """Returns all 15 Samsung Galaxy phones with their key features from relational database."""
    from sqlalchemy.orm import Session
    engine, _ = get_database_engine()
    results = []
    with Session(engine) as session:
        phones = session.query(Phone).order_by(Phone.id).all()
        for p in phones:
            results.append({
                "id": p.id,
                "model_name": p.model_name,
                "series": p.series,
                "display_size": p.display_size,
                "processor": p.chipset,
                "main_camera": f"{p.main_camera_setup} ({p.main_camera_primary})",
                "battery": p.battery_capacity,
                "charging": p.charging.split("\n")[0] if p.charging else "N/A",
                "url": p.url,
                "image_url": p.image_url
            })
    return {"total_phones": len(results), "phones": results}


@app.get("/chats", tags=["Chat History"])
async def get_chat_history(limit: int = 20):
    """Returns stored chat questions and answers from the PostgreSQL database."""
    from sqlalchemy.orm import Session
    from db_manager import ChatQueryLog
    engine, _ = get_database_engine()
    results = []
    with Session(engine) as session:
        logs = session.query(ChatQueryLog).order_by(ChatQueryLog.id.desc()).limit(limit).all()
        for log in logs:
            results.append({
                "id": log.id,
                "question": log.question,
                "route": log.route,
                "response": log.response,
                "source_phones": log.source_phones,
                "execution_time_seconds": log.execution_time_seconds,
                "created_at": log.created_at.isoformat() if log.created_at else None
            })
    return {"total_logged": len(results), "chats": results}


# ==============================================================================
# 6. Standalone Runner
# ==============================================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print("=" * 75)
    print(" Samsung Phone Query and Review System - FastAPI Backend")
    print(f" Starting server on: http://127.0.0.1:{port}")
    print(f" Interactive Swagger API Docs: http://127.0.0.1:{port}/docs")
    print("=" * 75)
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
