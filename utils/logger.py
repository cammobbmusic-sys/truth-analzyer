"""
Logger - Logging and auditability utilities.
Provides structured logging for the multi-AI system.
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict


@dataclass
class LogEntry:
    """Structured log entry."""
    timestamp: str
    level: str
    component: str
    operation: str
    query: Optional[str] = None
    models_used: Optional[list] = None
    duration: Optional[float] = None
    success: Optional[bool] = None
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class Logger:
    """Centralized logging system for the multi-AI system."""

    def __init__(self, log_level: str = "INFO", log_file: Optional[str] = None):
        self.log_level = getattr(logging, log_level.upper(), logging.INFO)
        self.log_file = log_file or "multi_ai_system.log"

        # Set up Python logging
        self._setup_logging()

        # Initialize audit log
        self.audit_entries = []

    def _setup_logging(self):
        """Set up Python logging configuration."""
        # Create logger
        self.logger = logging.getLogger("multi_ai_system")
        self.logger.setLevel(self.log_level)

        # Remove existing handlers
        self.logger.handlers.clear()

        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(component)s - %(message)s'
        )

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.log_level)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # File handler
        try:
            file_handler = logging.FileHandler(self.log_file)
            file_handler.setLevel(logging.DEBUG)  # Log everything to file
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        except Exception as e:
            print(f"Warning: Could not set up file logging: {e}")

        # Add custom fields to log records
        old_factory = logging.getLogRecordFactory()

        def record_factory(*args, **kwargs):
            record = old_factory(*args, **kwargs)
            record.component = getattr(record, 'component', 'unknown')
            return record

        logging.setLogRecordFactory(record_factory)

    def log_operation(
        self,
        component: str,
        operation: str,
        query: Optional[str] = None,
        models_used: Optional[list] = None,
        duration: Optional[float] = None,
        success: Optional[bool] = None,
        error_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        level: str = "INFO"
    ):
        """
        Log an operation with structured data.

        Args:
            component: Component that performed the operation (e.g., 'orchestrator', 'verification')
            operation: Operation performed (e.g., 'run_analysis', 'cross_validate')
            query: The query or topic being processed
            models_used: List of AI models used
            duration: Operation duration in seconds
            success: Whether the operation succeeded
            error_message: Error message if operation failed
            metadata: Additional metadata
            level: Log level (DEBUG, INFO, WARNING, ERROR)
        """
        # Create log entry
        entry = LogEntry(
            timestamp=datetime.now().isoformat(),
            level=level,
            component=component,
            operation=operation,
            query=query,
            models_used=models_used,
            duration=duration,
            success=success,
            error_message=error_message,
            metadata=metadata
        )

        # Add to audit log
        self.audit_entries.append(entry)

        # Convert to log message
        log_message = f"{operation}"
        if query:
            # Truncate long queries for logging
            truncated_query = query[:100] + "..." if len(query) > 100 else query
            log_message += f" - Query: {truncated_query}"
        if duration:
            log_message += ".2f"
        if success is not None:
            log_message += f" - Success: {success}"
        if error_message:
            log_message += f" - Error: {error_message}"

        # Log using Python logging
        log_method = getattr(self.logger, level.lower(), self.logger.info)
        log_method(log_message, extra={'component': component})

    def log_verification_result(
        self,
        query: str,
        consensus_achieved: bool,
        confidence_score: float,
        models_used: List[str],
        duration: float
    ):
        """Log verification operation results."""
        self.log_operation(
            component="verification",
            operation="cross_validation",
            query=query,
            models_used=models_used,
            duration=duration,
            success=consensus_achieved,
            metadata={
                "consensus_achieved": consensus_achieved,
                "confidence_score": confidence_score
            }
        )

    def log_brainstorming_result(
        self,
        query: str,
        num_ideas: int,
        num_clusters: int,
        diversity_score: float,
        models_used: List[str],
        duration: float
    ):
        """Log brainstorming operation results."""
        self.log_operation(
            component="brainstorming",
            operation="idea_generation",
            query=query,
            models_used=models_used,
            duration=duration,
            success=True,
            metadata={
                "num_ideas": num_ideas,
                "num_clusters": num_clusters,
                "diversity_score": diversity_score
            }
        )

    def log_model_call(
        self,
        model_name: str,
        prompt_length: int,
        response_length: int,
        duration: float,
        success: bool,
        error_message: Optional[str] = None
    ):
        """Log individual model API call."""
        self.log_operation(
            component="model_agent",
            operation="api_call",
            models_used=[model_name],
            duration=duration,
            success=success,
            error_message=error_message,
            metadata={
                "prompt_length": prompt_length,
                "response_length": response_length
            },
            level="DEBUG"
        )

    def log_error(
        self,
        component: str,
        operation: str,
        error: Exception,
        query: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Log an error with full context."""
        error_message = f"{type(error).__name__}: {str(error)}"

        self.log_operation(
            component=component,
            operation=operation,
            query=query,
            success=False,
            error_message=error_message,
            metadata=metadata,
            level="ERROR"
        )

    def get_audit_trail(self, component: Optional[str] = None, limit: int = 100) -> list:
        """
        Get audit trail entries.

        Args:
            component: Filter by component (optional)
            limit: Maximum number of entries to return

        Returns:
            List of audit entries
        """
        entries = self.audit_entries

        if component:
            entries = [e for e in entries if e.component == component]

        # Return most recent entries first
        return [asdict(entry) for entry in entries[-limit:]]

    def export_audit_log(self, filepath: str):
        """
        Export audit log to JSON file.

        Args:
            filepath: Path to export audit log
        """
        try:
            audit_data = {
                "export_timestamp": datetime.now().isoformat(),
                "total_entries": len(self.audit_entries),
                "entries": [asdict(entry) for entry in self.audit_entries]
            }

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(audit_data, f, indent=2, ensure_ascii=False)

            self.logger.info(f"Audit log exported to {filepath}")

        except Exception as e:
            self.logger.error(f"Failed to export audit log: {e}")

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics from audit log."""
        if not self.audit_entries:
            return {"total_operations": 0}

        stats = {
            "total_operations": len(self.audit_entries),
            "successful_operations": sum(1 for e in self.audit_entries if e.success),
            "failed_operations": sum(1 for e in self.audit_entries if e.success is False),
            "average_duration": 0.0,
            "component_breakdown": {},
            "operation_breakdown": {}
        }

        # Calculate averages and breakdowns
        durations = [e.duration for e in self.audit_entries if e.duration is not None]
        if durations:
            stats["average_duration"] = sum(durations) / len(durations)

        # Component breakdown
        for entry in self.audit_entries:
            stats["component_breakdown"][entry.component] = \
                stats["component_breakdown"].get(entry.component, 0) + 1

            stats["operation_breakdown"][entry.operation] = \
                stats["operation_breakdown"].get(entry.operation, 0) + 1

        return stats

    def cleanup_old_entries(self, max_entries: int = 10000):
        """
        Clean up old audit entries to prevent memory issues.

        Args:
            max_entries: Maximum number of entries to keep
        """
        if len(self.audit_entries) > max_entries:
            # Keep most recent entries
            self.audit_entries = self.audit_entries[-max_entries:]
            self.logger.info(f"Cleaned up audit log, keeping {max_entries} most recent entries")


class ComponentLogger:
    """Logger wrapper for specific components."""

    def __init__(self, main_logger: Logger, component: str):
        self.main_logger = main_logger
        self.component = component

    def log_operation(self, operation: str, **kwargs):
        """Log operation for this component."""
        self.main_logger.log_operation(component=self.component, operation=operation, **kwargs)

    def log_error(self, operation: str, error: Exception, **kwargs):
        """Log error for this component."""
        self.main_logger.log_error(component=self.component, operation=operation, error=error, **kwargs)
