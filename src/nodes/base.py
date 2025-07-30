"""Base node classes for async operations."""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, TypeVar, Generic
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


class AsyncNode(ABC):
    """Base class for async operations with retry logic."""
    
    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0):
        """Initialize AsyncNode with retry configuration.
        
        Args:
            max_retries: Maximum number of retry attempts
            retry_delay: Base delay between retries (exponential backoff)
        """
        self.max_retries = max_retries
        self.retry_delay = retry_delay
    
    @abstractmethod
    async def _execute_impl(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Implementation of the node's core logic.
        
        Args:
            context: Experiment context dictionary
            
        Returns:
            Updated context dictionary
        """
        pass
    
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute node with retry logic.
        
        Args:
            context: Experiment context dictionary
            
        Returns:
            Updated context dictionary
            
        Raises:
            Exception: If all retry attempts fail
        """
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                return await self._execute_impl(context)
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(
                        f"{self.__class__.__name__} attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {delay}s..."
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        f"{self.__class__.__name__} failed after {self.max_retries} attempts: {e}"
                    )
                    with open("experiment_errors.log", "a") as f:
                        f.write(f"{datetime.now().isoformat()} - {self.__class__.__name__} - "
                               f"Failed after {self.max_retries} attempts: {e}\n")
        
        raise last_error


class AsyncFlow:
    """Base class for orchestrating multiple nodes."""
    
    def __init__(self):
        """Initialize AsyncFlow with empty node list."""
        self.nodes: List[AsyncNode] = []
    
    def add_node(self, node: AsyncNode) -> 'AsyncFlow':
        """Add a node to the flow.
        
        Args:
            node: AsyncNode to add to the flow
            
        Returns:
            Self for method chaining
        """
        self.nodes.append(node)
        return self
    
    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Run all nodes in sequence.
        
        Args:
            context: Initial context dictionary
            
        Returns:
            Final context dictionary after all nodes execute
        """
        for node in self.nodes:
            context = await node.execute(context)
        return context


T = TypeVar('T')
R = TypeVar('R')


class AsyncParallelBatchNode(AsyncNode, Generic[T, R]):
    """Execute multiple async operations in parallel with error isolation."""
    
    @abstractmethod
    async def process_item(self, item: T) -> R:
        """Process a single item.
        
        Args:
            item: Item to process
            
        Returns:
            Processed result
        """
        pass
    
    async def execute_batch(self, items: List[T]) -> List[Optional[R]]:
        """Execute batch processing with partial failure handling.
        
        Args:
            items: List of items to process in parallel
            
        Returns:
            List of results (None for failed items)
        """
        tasks = [self._process_with_error_handling(item) for item in items]
        return await asyncio.gather(*tasks)
    
    async def _process_with_error_handling(self, item: T) -> Optional[R]:
        """Process item with error handling.
        
        Args:
            item: Item to process
            
        Returns:
            Processed result or None if failed
        """
        try:
            return await self.process_item(item)
        except Exception as e:
            logger.error(f"Failed to process item {item}: {e}")
            with open("experiment_errors.log", "a") as f:
                f.write(f"{datetime.now().isoformat()} - {self.__class__.__name__} - "
                       f"Failed to process item: {e}\n")
            return None
    
    async def _execute_impl(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Default implementation that should be overridden by subclasses."""
        raise NotImplementedError("Subclasses must implement _execute_impl")


@dataclass
class ContextKeys:
    """Standard keys for the context dictionary."""
    EXPERIMENT_ID = "experiment_id"
    ROUND = "round"
    AGENTS = "agents"
    STRATEGIES = "strategies"
    GAMES = "games"
    ROUND_SUMMARIES = "round_summaries"
    CONFIG = "config"
    DATA_MANAGER = "data_manager"


def validate_context(context: Dict[str, Any], required_keys: List[str]) -> None:
    """Validate that context contains all required keys.
    
    Args:
        context: Context dictionary to validate
        required_keys: List of required key names
        
    Raises:
        ValueError: If any required keys are missing
    """
    missing_keys = [key for key in required_keys if key not in context]
    if missing_keys:
        raise ValueError(f"Context missing required keys: {missing_keys}")