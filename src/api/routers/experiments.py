"""
Experiments routes for the API.
"""
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from src.api.server import get_current_user, User
from src.api.data_loader import DataLoader

router = APIRouter()

# Initialize data loader
data_loader = DataLoader()


class ExperimentSummary(BaseModel):
    experiment_id: str
    start_time: str
    end_time: str
    total_rounds: int
    total_games: int
    total_api_calls: int
    total_cost: float
    status: str


class ExperimentDetail(BaseModel):
    experiment_id: str
    start_time: str
    end_time: str
    total_rounds: int
    total_games: int
    total_api_calls: int
    total_cost: float
    acausal_indicators: dict
    round_count: int
    agent_count: int


class PaginatedResponse(BaseModel):
    items: List[ExperimentSummary]
    total: int
    page: int
    page_size: int
    total_pages: int


@router.get("/experiments", response_model=PaginatedResponse)
async def list_experiments(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    sort_by: str = Query("start_time", pattern="^(start_time|end_time|total_rounds|total_cost)$"),
    sort_order: str = Query("desc", pattern="^(asc|desc)$"),
    current_user: User = Depends(get_current_user)
):
    """
    List all experiments with pagination.
    
    - **page**: Page number (starts at 1)
    - **page_size**: Number of items per page (max 100)
    - **sort_by**: Field to sort by
    - **sort_order**: Sort order (asc or desc)
    """
    experiments = data_loader.list_experiments()
    
    # Sort experiments
    reverse = sort_order == "desc"
    if sort_by == "start_time":
        experiments.sort(key=lambda x: x.start_time, reverse=reverse)
    elif sort_by == "end_time":
        experiments.sort(key=lambda x: x.end_time, reverse=reverse)
    elif sort_by == "total_rounds":
        experiments.sort(key=lambda x: x.total_rounds, reverse=reverse)
    elif sort_by == "total_cost":
        experiments.sort(key=lambda x: x.total_cost, reverse=reverse)
    
    # Paginate
    total = len(experiments)
    start = (page - 1) * page_size
    end = start + page_size
    items = experiments[start:end]
    
    return PaginatedResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
        total_pages=(total + page_size - 1) // page_size
    )


@router.get("/experiments/{experiment_id}", response_model=ExperimentDetail)
async def get_experiment(
    experiment_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get details for a specific experiment."""
    experiment = data_loader.get_experiment(experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return experiment


@router.get("/experiments/{experiment_id}/rounds/{round_num}")
async def get_round_data(
    experiment_id: str,
    round_num: int,
    current_user: User = Depends(get_current_user)
):
    """Get data for a specific round in an experiment."""
    round_data = data_loader.get_round_data(experiment_id, round_num)
    if not round_data:
        raise HTTPException(status_code=404, detail="Round data not found")
    return round_data