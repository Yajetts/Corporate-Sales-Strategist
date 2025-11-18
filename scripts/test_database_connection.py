"""Test script to verify database connections"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.api.database import (
    postgres_manager,
    mongodb_manager,
    AnalysisRepository,
    MarketAnalysisRepository,
    StrategyRepository,
    PerformanceRepository,
    BusinessOptimizationRepository,
    UserConfigurationRepository
)
from src.api.migrations import create_tables_if_not_exist
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_postgresql_connection():
    """Test PostgreSQL connection"""
    logger.info("Testing PostgreSQL connection...")
    try:
        postgres_manager.initialize()
        healthy = postgres_manager.health_check()
        if healthy:
            logger.info("✓ PostgreSQL connection successful")
            return True
        else:
            logger.error("✗ PostgreSQL health check failed")
            return False
    except Exception as e:
        logger.error(f"✗ PostgreSQL connection failed: {e}")
        return False


def test_mongodb_connection():
    """Test MongoDB connection"""
    logger.info("Testing MongoDB connection...")
    try:
        mongodb_manager.initialize()
        healthy = mongodb_manager.health_check()
        if healthy:
            logger.info("✓ MongoDB connection successful")
            return True
        else:
            logger.error("✗ MongoDB health check failed")
            return False
    except Exception as e:
        logger.error(f"✗ MongoDB connection failed: {e}")
        return False


def test_table_creation():
    """Test table creation"""
    logger.info("Testing table creation...")
    try:
        success = create_tables_if_not_exist()
        if success:
            logger.info("✓ Tables created/verified successfully")
            return True
        else:
            logger.error("✗ Table creation failed")
            return False
    except Exception as e:
        logger.error(f"✗ Table creation failed: {e}")
        return False


def test_repository_operations():
    """Test repository CRUD operations"""
    logger.info("Testing repository operations...")
    
    try:
        # Test AnalysisRepository
        test_result = {
            'task_id': 'test_task_001',
            'text_hash': 'test_hash_001',
            'source_type': 'test',
            'product_category': 'Test Category',
            'business_domain': 'Test Domain',
            'value_proposition': 'Test value proposition',
            'key_features': ['feature1', 'feature2'],
            'confidence_scores': {'category': 0.95, 'domain': 0.89},
            'processing_time_ms': 100
        }
        
        record_id = AnalysisRepository.save_analysis_result(test_result)
        logger.info(f"✓ Saved analysis result with ID: {record_id}")
        
        retrieved = AnalysisRepository.get_analysis_by_task_id('test_task_001')
        if retrieved:
            logger.info(f"✓ Retrieved analysis result: {retrieved['product_category']}")
        else:
            logger.error("✗ Failed to retrieve analysis result")
            return False
        
        # Test UserConfigurationRepository
        user_id = UserConfigurationRepository.save_user_config(
            'test_user_001',
            {'theme': 'dark', 'language': 'en'},
            api_quota=1000
        )
        logger.info(f"✓ Saved user configuration with ID: {user_id}")
        
        user_config = UserConfigurationRepository.get_user_config('test_user_001')
        if user_config:
            logger.info(f"✓ Retrieved user config: {user_config['preferences']}")
        else:
            logger.error("✗ Failed to retrieve user config")
            return False
        
        logger.info("✓ All repository operations successful")
        return True
        
    except Exception as e:
        logger.error(f"✗ Repository operations failed: {e}")
        return False


def test_mongodb_operations():
    """Test MongoDB operations"""
    logger.info("Testing MongoDB operations...")
    
    try:
        collection = mongodb_manager.get_collection('test_collection')
        
        # Insert test document
        test_doc = {
            'task_id': 'test_task_001',
            'data': {'key': 'value'},
            'test': True
        }
        result = collection.insert_one(test_doc)
        logger.info(f"✓ Inserted document with ID: {result.inserted_id}")
        
        # Retrieve document
        retrieved = collection.find_one({'task_id': 'test_task_001'})
        if retrieved:
            logger.info(f"✓ Retrieved document: {retrieved['data']}")
        else:
            logger.error("✗ Failed to retrieve document")
            return False
        
        # Clean up
        collection.delete_one({'task_id': 'test_task_001'})
        logger.info("✓ Cleaned up test document")
        
        logger.info("✓ All MongoDB operations successful")
        return True
        
    except Exception as e:
        logger.error(f"✗ MongoDB operations failed: {e}")
        return False


def main():
    """Run all database tests"""
    logger.info("=" * 60)
    logger.info("Database Connection Test Suite")
    logger.info("=" * 60)
    
    results = {
        'PostgreSQL Connection': test_postgresql_connection(),
        'MongoDB Connection': test_mongodb_connection(),
        'Table Creation': test_table_creation(),
        'Repository Operations': test_repository_operations(),
        'MongoDB Operations': test_mongodb_operations()
    }
    
    logger.info("=" * 60)
    logger.info("Test Results Summary")
    logger.info("=" * 60)
    
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        logger.info(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    
    logger.info("=" * 60)
    if all_passed:
        logger.info("✓ All tests passed!")
        return 0
    else:
        logger.error("✗ Some tests failed. Check the logs above.")
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
