"""Celery application configuration and initialization"""

import os
import logging
from celery import Celery
from kombu import Queue, Exchange

logger = logging.getLogger(__name__)


def make_celery(app_name='sales_strategist'):
    """
    Create and configure Celery application.
    
    Args:
        app_name: Name of the Celery application
        
    Returns:
        Configured Celery instance
    """
    # Get configuration from environment
    broker_url = os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0')
    result_backend = os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0')
    
    # Create Celery instance
    celery_app = Celery(
        app_name,
        broker=broker_url,
        backend=result_backend,
        include=[
            'src.api.tasks'
        ]
    )
    
    # Configure Celery
    celery_app.conf.update(
        # Task execution settings
        task_serializer='json',
        accept_content=['json'],
        result_serializer='json',
        timezone='UTC',
        enable_utc=True,
        
        # Task result settings
        result_expires=3600,  # Results expire after 1 hour
        result_extended=True,
        
        # Task routing
        task_routes={
            'src.api.tasks.analyze_company_async': {'queue': 'analysis'},
            'src.api.tasks.analyze_market_async': {'queue': 'analysis'},
            'src.api.tasks.generate_strategy_async': {'queue': 'strategy'},
            'src.api.tasks.monitor_performance_async': {'queue': 'monitoring'},
            'src.api.tasks.optimize_business_async': {'queue': 'optimization'},
            'src.api.tasks.explain_model_async': {'queue': 'explanation'},
        },
        
        # Task time limits
        task_time_limit=int(os.getenv('TASK_TIMEOUT', 300)),  # Hard limit
        task_soft_time_limit=int(os.getenv('TASK_TIMEOUT', 300)) - 30,  # Soft limit
        
        # Worker settings
        worker_prefetch_multiplier=1,
        worker_max_tasks_per_child=100,
        
        # Task acknowledgment
        task_acks_late=True,
        task_reject_on_worker_lost=True,
        
        # Monitoring
        worker_send_task_events=True,
        task_send_sent_event=True,
        
        # Error handling
        task_annotations={
            '*': {
                'rate_limit': '100/m',  # 100 tasks per minute
                'max_retries': 3,
                'default_retry_delay': 60,  # Retry after 60 seconds
            }
        }
    )
    
    # Define task queues
    celery_app.conf.task_queues = (
        Queue('default', Exchange('default'), routing_key='default'),
        Queue('analysis', Exchange('analysis'), routing_key='analysis'),
        Queue('strategy', Exchange('strategy'), routing_key='strategy'),
        Queue('monitoring', Exchange('monitoring'), routing_key='monitoring'),
        Queue('optimization', Exchange('optimization'), routing_key='optimization'),
        Queue('explanation', Exchange('explanation'), routing_key='explanation'),
    )
    
    logger.info(f"Celery app configured with broker: {broker_url}")
    
    return celery_app


# Create global Celery instance
celery = make_celery()


# Task base class with error handling
class BaseTask(celery.Task):
    """Base task class with error handling and logging"""
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """
        Error handler called when task fails.
        
        Args:
            exc: Exception raised
            task_id: Unique task ID
            args: Task positional arguments
            kwargs: Task keyword arguments
            einfo: Exception info
        """
        logger.error(f"Task {task_id} failed: {exc}")
        logger.error(f"Exception info: {einfo}")
    
    def on_retry(self, exc, task_id, args, kwargs, einfo):
        """
        Handler called when task is retried.
        
        Args:
            exc: Exception that caused retry
            task_id: Unique task ID
            args: Task positional arguments
            kwargs: Task keyword arguments
            einfo: Exception info
        """
        logger.warning(f"Task {task_id} retrying due to: {exc}")
    
    def on_success(self, retval, task_id, args, kwargs):
        """
        Handler called when task succeeds.
        
        Args:
            retval: Return value of task
            task_id: Unique task ID
            args: Task positional arguments
            kwargs: Task keyword arguments
        """
        logger.info(f"Task {task_id} completed successfully")


if __name__ == '__main__':
    # Start Celery worker
    celery.start()
