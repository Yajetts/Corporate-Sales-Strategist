"""Database connection management and ORM setup"""

import os
import logging
from typing import Optional
from contextlib import contextmanager
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, JSON, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.pool import QueuePool
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from datetime import datetime

logger = logging.getLogger(__name__)

# SQLAlchemy Base
Base = declarative_base()


# Database Models

class AnalysisResult(Base):
    """Model for storing company analysis results"""
    __tablename__ = 'analysis_results'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(String(255), unique=True, index=True)
    text_hash = Column(String(64), index=True)
    source_type = Column(String(50))
    product_category = Column(String(255))
    business_domain = Column(String(255))
    value_proposition = Column(Text)
    key_features = Column(JSON)
    confidence_scores = Column(JSON)
    processing_time_ms = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)


class MarketAnalysisResult(Base):
    """Model for storing market analysis results"""
    __tablename__ = 'market_analysis_results'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(String(255), unique=True, index=True)
    num_entities = Column(Integer)
    num_clusters = Column(Integer)
    clustering_method = Column(String(50))
    latent_dimensions = Column(Integer)
    processing_time_seconds = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)


class StrategyResult(Base):
    """Model for storing strategy generation results"""
    __tablename__ = 'strategy_results'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(String(255), unique=True, index=True)
    market_state = Column(JSON)
    recommendations = Column(JSON)
    confidence_score = Column(Float)
    confidence_level = Column(String(50))
    has_explanation = Column(Integer)  # Boolean as integer
    created_at = Column(DateTime, default=datetime.utcnow)


class PerformanceResult(Base):
    """Model for storing performance monitoring results"""
    __tablename__ = 'performance_results'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(String(255), unique=True, index=True)
    forecast_horizon_days = Column(Integer)
    num_alerts = Column(Integer)
    critical_alerts = Column(Integer)
    has_feedback = Column(Integer)  # Boolean as integer
    processing_time_seconds = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)


class BusinessOptimizationResult(Base):
    """Model for storing business optimization results"""
    __tablename__ = 'business_optimization_results'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(String(255), unique=True, index=True)
    num_products = Column(Integer)
    total_revenue = Column(Float)
    total_cost = Column(Float)
    profit = Column(Float)
    roi = Column(Float)
    optimization_success = Column(Integer)  # Boolean as integer
    processing_time_seconds = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)


class UserConfiguration(Base):
    """Model for storing user configurations"""
    __tablename__ = 'user_configurations'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(255), unique=True, index=True)
    preferences = Column(JSON)
    api_quota = Column(Integer)
    api_usage = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# PostgreSQL Connection Manager

class PostgreSQLManager:
    """Manages PostgreSQL database connections using SQLAlchemy"""
    
    def __init__(self):
        """Initialize PostgreSQL connection manager"""
        self.engine = None
        self.session_factory = None
        self.Session = None
        self._initialized = False
    
    def initialize(self):
        """Initialize database connection and create tables"""
        if self._initialized:
            return
        
        try:
            # Build connection string
            host = os.getenv('POSTGRES_HOST', 'localhost')
            port = os.getenv('POSTGRES_PORT', '5432')
            database = os.getenv('POSTGRES_DB', 'sales_strategist')
            user = os.getenv('POSTGRES_USER', 'postgres')
            password = os.getenv('POSTGRES_PASSWORD', 'postgres')
            
            connection_string = f'postgresql://{user}:{password}@{host}:{port}/{database}'
            
            # Create engine with connection pooling
            self.engine = create_engine(
                connection_string,
                poolclass=QueuePool,
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True,  # Verify connections before using
                pool_recycle=3600,  # Recycle connections after 1 hour
                echo=False
            )
            
            # Create session factory
            self.session_factory = sessionmaker(bind=self.engine)
            self.Session = scoped_session(self.session_factory)
            
            # Create tables
            Base.metadata.create_all(self.engine)
            
            self._initialized = True
            logger.info(f"PostgreSQL connection initialized: {host}:{port}/{database}")
            
        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL connection: {e}")
            raise
    
    @contextmanager
    def get_session(self):
        """
        Context manager for database sessions.
        
        Usage:
            with db_manager.get_session() as session:
                result = session.query(Model).all()
        """
        if not self._initialized:
            self.initialize()
        
        session = self.Session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()
    
    def close(self):
        """Close database connections"""
        if self.Session:
            self.Session.remove()
        if self.engine:
            self.engine.dispose()
        self._initialized = False
        logger.info("PostgreSQL connections closed")
    
    def health_check(self) -> bool:
        """
        Check database connection health.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            if not self._initialized:
                self.initialize()
            
            with self.get_session() as session:
                session.execute('SELECT 1')
            return True
        except Exception as e:
            logger.error(f"PostgreSQL health check failed: {e}")
            return False


# MongoDB Connection Manager

class MongoDBManager:
    """Manages MongoDB database connections"""
    
    def __init__(self):
        """Initialize MongoDB connection manager"""
        self.client = None
        self.db = None
        self._initialized = False
    
    def initialize(self):
        """Initialize MongoDB connection"""
        if self._initialized:
            return
        
        try:
            # Build connection string
            host = os.getenv('MONGODB_HOST', 'localhost')
            port = int(os.getenv('MONGODB_PORT', '27017'))
            database = os.getenv('MONGODB_DB', 'sales_strategist')
            
            # Optional authentication
            username = os.getenv('MONGODB_USER')
            password = os.getenv('MONGODB_PASSWORD')
            
            if username and password:
                connection_string = f'mongodb://{username}:{password}@{host}:{port}/{database}'
            else:
                connection_string = f'mongodb://{host}:{port}/{database}'
            
            # Create client with connection pooling
            self.client = MongoClient(
                connection_string,
                maxPoolSize=50,
                minPoolSize=10,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=10000
            )
            
            # Get database
            self.db = self.client[database]
            
            # Test connection
            self.client.admin.command('ping')
            
            self._initialized = True
            logger.info(f"MongoDB connection initialized: {host}:{port}/{database}")
            
        except Exception as e:
            logger.error(f"Failed to initialize MongoDB connection: {e}")
            raise
    
    def get_collection(self, collection_name: str):
        """
        Get MongoDB collection.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            MongoDB collection object
        """
        if not self._initialized:
            self.initialize()
        
        return self.db[collection_name]
    
    def close(self):
        """Close MongoDB connections"""
        if self.client:
            self.client.close()
        self._initialized = False
        logger.info("MongoDB connections closed")
    
    def health_check(self) -> bool:
        """
        Check database connection health.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            if not self._initialized:
                self.initialize()
            
            self.client.admin.command('ping')
            return True
        except ConnectionFailure as e:
            logger.error(f"MongoDB health check failed: {e}")
            return False


# Global database manager instances
postgres_manager = PostgreSQLManager()
mongodb_manager = MongoDBManager()


# Repository Pattern Implementation

class AnalysisRepository:
    """Repository for analysis results"""
    
    @staticmethod
    def save_analysis_result(result: dict) -> int:
        """
        Save analysis result to PostgreSQL.
        
        Args:
            result: Analysis result dictionary
            
        Returns:
            ID of saved record
        """
        with postgres_manager.get_session() as session:
            analysis = AnalysisResult(
                task_id=result.get('task_id'),
                text_hash=result.get('text_hash'),
                source_type=result.get('source_type'),
                product_category=result.get('product_category'),
                business_domain=result.get('business_domain'),
                value_proposition=result.get('value_proposition'),
                key_features=result.get('key_features'),
                confidence_scores=result.get('confidence_scores'),
                processing_time_ms=result.get('processing_time_ms')
            )
            session.add(analysis)
            session.flush()
            return analysis.id
    
    @staticmethod
    def get_analysis_by_task_id(task_id: str) -> Optional[dict]:
        """
        Get analysis result by task ID.
        
        Args:
            task_id: Task identifier
            
        Returns:
            Analysis result dictionary or None
        """
        with postgres_manager.get_session() as session:
            analysis = session.query(AnalysisResult).filter_by(task_id=task_id).first()
            if analysis:
                return {
                    'id': analysis.id,
                    'task_id': analysis.task_id,
                    'product_category': analysis.product_category,
                    'business_domain': analysis.business_domain,
                    'value_proposition': analysis.value_proposition,
                    'key_features': analysis.key_features,
                    'confidence_scores': analysis.confidence_scores,
                    'processing_time_ms': analysis.processing_time_ms,
                    'created_at': analysis.created_at.isoformat()
                }
            return None


class MarketAnalysisRepository:
    """Repository for market analysis results"""
    
    @staticmethod
    def save_market_analysis(result: dict) -> int:
        """Save market analysis result"""
        with postgres_manager.get_session() as session:
            analysis = MarketAnalysisResult(
                task_id=result.get('task_id'),
                num_entities=result.get('num_entities'),
                num_clusters=result.get('clusters', {}).get('n_clusters'),
                clustering_method=result.get('clusters', {}).get('method'),
                latent_dimensions=result.get('latent_dimensions'),
                processing_time_seconds=result.get('processing_time_seconds')
            )
            session.add(analysis)
            session.flush()
            
            # Save detailed results to MongoDB
            collection = mongodb_manager.get_collection('market_analysis_details')
            collection.insert_one({
                'task_id': result.get('task_id'),
                'clusters': result.get('clusters'),
                'graph': result.get('graph'),
                'potential_clients': result.get('potential_clients'),
                'created_at': datetime.utcnow()
            })
            
            return analysis.id


class StrategyRepository:
    """Repository for strategy results"""
    
    @staticmethod
    def save_strategy(result: dict) -> int:
        """Save strategy result"""
        with postgres_manager.get_session() as session:
            strategy = StrategyResult(
                task_id=result.get('task_id'),
                market_state=result.get('market_state'),
                recommendations=result.get('recommendations'),
                confidence_score=result.get('confidence_score'),
                confidence_level=result.get('confidence_level'),
                has_explanation=1 if result.get('explanation') else 0
            )
            session.add(strategy)
            session.flush()
            
            # Save detailed explanation to MongoDB if present
            if result.get('explanation'):
                collection = mongodb_manager.get_collection('strategy_explanations')
                collection.insert_one({
                    'task_id': result.get('task_id'),
                    'explanation': result.get('explanation'),
                    'actionable_insights': result.get('actionable_insights'),
                    'created_at': datetime.utcnow()
                })
            
            return strategy.id
    
    @staticmethod
    def get_strategy_by_task_id(task_id: str) -> Optional[dict]:
        """
        Get strategy result by task ID.
        
        Args:
            task_id: Task identifier
            
        Returns:
            Strategy result dictionary or None
        """
        with postgres_manager.get_session() as session:
            strategy = session.query(StrategyResult).filter_by(task_id=task_id).first()
            if strategy:
                result = {
                    'id': strategy.id,
                    'task_id': strategy.task_id,
                    'market_state': strategy.market_state,
                    'recommendations': strategy.recommendations,
                    'confidence_score': strategy.confidence_score,
                    'confidence_level': strategy.confidence_level,
                    'created_at': strategy.created_at.isoformat()
                }
                
                # Get explanation from MongoDB if available
                if strategy.has_explanation:
                    collection = mongodb_manager.get_collection('strategy_explanations')
                    explanation_doc = collection.find_one({'task_id': task_id})
                    if explanation_doc:
                        result['explanation'] = explanation_doc.get('explanation')
                        result['actionable_insights'] = explanation_doc.get('actionable_insights')
                
                return result
            return None


class PerformanceRepository:
    """Repository for performance monitoring results"""
    
    @staticmethod
    def save_performance_result(result: dict) -> int:
        """Save performance monitoring result"""
        with postgres_manager.get_session() as session:
            performance = PerformanceResult(
                task_id=result.get('task_id'),
                forecast_horizon_days=result.get('forecast_horizon_days'),
                num_alerts=len(result.get('alerts', [])),
                critical_alerts=sum(1 for a in result.get('alerts', []) if a.get('severity') == 'critical'),
                has_feedback=1 if result.get('feedback_summary') else 0,
                processing_time_seconds=result.get('processing_time_seconds')
            )
            session.add(performance)
            session.flush()
            
            # Save detailed results to MongoDB
            collection = mongodb_manager.get_collection('performance_details')
            collection.insert_one({
                'task_id': result.get('task_id'),
                'forecast': result.get('forecast'),
                'confidence_intervals': result.get('confidence_intervals'),
                'alerts': result.get('alerts'),
                'feedback_summary': result.get('feedback_summary'),
                'created_at': datetime.utcnow()
            })
            
            return performance.id
    
    @staticmethod
    def get_performance_by_task_id(task_id: str) -> Optional[dict]:
        """
        Get performance result by task ID.
        
        Args:
            task_id: Task identifier
            
        Returns:
            Performance result dictionary or None
        """
        with postgres_manager.get_session() as session:
            performance = session.query(PerformanceResult).filter_by(task_id=task_id).first()
            if performance:
                result = {
                    'id': performance.id,
                    'task_id': performance.task_id,
                    'forecast_horizon_days': performance.forecast_horizon_days,
                    'num_alerts': performance.num_alerts,
                    'critical_alerts': performance.critical_alerts,
                    'processing_time_seconds': performance.processing_time_seconds,
                    'created_at': performance.created_at.isoformat()
                }
                
                # Get detailed results from MongoDB
                collection = mongodb_manager.get_collection('performance_details')
                details_doc = collection.find_one({'task_id': task_id})
                if details_doc:
                    result['forecast'] = details_doc.get('forecast')
                    result['confidence_intervals'] = details_doc.get('confidence_intervals')
                    result['alerts'] = details_doc.get('alerts')
                    result['feedback_summary'] = details_doc.get('feedback_summary')
                
                return result
            return None


class BusinessOptimizationRepository:
    """Repository for business optimization results"""
    
    @staticmethod
    def save_optimization_result(result: dict) -> int:
        """Save business optimization result"""
        with postgres_manager.get_session() as session:
            optimization = BusinessOptimizationResult(
                task_id=result.get('task_id'),
                num_products=len(result.get('production_priorities', [])),
                total_revenue=result.get('total_revenue'),
                total_cost=result.get('total_cost'),
                profit=result.get('profit'),
                roi=result.get('roi'),
                optimization_success=1 if result.get('optimization_success') else 0,
                processing_time_seconds=result.get('processing_time_seconds')
            )
            session.add(optimization)
            session.flush()
            
            # Save detailed results to MongoDB
            collection = mongodb_manager.get_collection('business_optimization_details')
            collection.insert_one({
                'task_id': result.get('task_id'),
                'production_priorities': result.get('production_priorities'),
                'resource_distribution': result.get('resource_distribution'),
                'constraints_applied': result.get('constraints_applied'),
                'optimization_details': result.get('optimization_details'),
                'created_at': datetime.utcnow()
            })
            
            return optimization.id
    
    @staticmethod
    def get_optimization_by_task_id(task_id: str) -> Optional[dict]:
        """
        Get optimization result by task ID.
        
        Args:
            task_id: Task identifier
            
        Returns:
            Optimization result dictionary or None
        """
        with postgres_manager.get_session() as session:
            optimization = session.query(BusinessOptimizationResult).filter_by(task_id=task_id).first()
            if optimization:
                result = {
                    'id': optimization.id,
                    'task_id': optimization.task_id,
                    'num_products': optimization.num_products,
                    'total_revenue': optimization.total_revenue,
                    'total_cost': optimization.total_cost,
                    'profit': optimization.profit,
                    'roi': optimization.roi,
                    'optimization_success': bool(optimization.optimization_success),
                    'processing_time_seconds': optimization.processing_time_seconds,
                    'created_at': optimization.created_at.isoformat()
                }
                
                # Get detailed results from MongoDB
                collection = mongodb_manager.get_collection('business_optimization_details')
                details_doc = collection.find_one({'task_id': task_id})
                if details_doc:
                    result['production_priorities'] = details_doc.get('production_priorities')
                    result['resource_distribution'] = details_doc.get('resource_distribution')
                    result['constraints_applied'] = details_doc.get('constraints_applied')
                    result['optimization_details'] = details_doc.get('optimization_details')
                
                return result
            return None


class UserConfigurationRepository:
    """Repository for user configurations"""
    
    @staticmethod
    def save_user_config(user_id: str, preferences: dict, api_quota: int = 1000) -> int:
        """
        Save or update user configuration.
        
        Args:
            user_id: User identifier
            preferences: User preferences dictionary
            api_quota: API usage quota
            
        Returns:
            ID of saved record
        """
        with postgres_manager.get_session() as session:
            # Check if user config exists
            config = session.query(UserConfiguration).filter_by(user_id=user_id).first()
            
            if config:
                # Update existing
                config.preferences = preferences
                config.api_quota = api_quota
                config.updated_at = datetime.utcnow()
            else:
                # Create new
                config = UserConfiguration(
                    user_id=user_id,
                    preferences=preferences,
                    api_quota=api_quota
                )
                session.add(config)
            
            session.flush()
            return config.id
    
    @staticmethod
    def get_user_config(user_id: str) -> Optional[dict]:
        """
        Get user configuration by user ID.
        
        Args:
            user_id: User identifier
            
        Returns:
            User configuration dictionary or None
        """
        with postgres_manager.get_session() as session:
            config = session.query(UserConfiguration).filter_by(user_id=user_id).first()
            if config:
                return {
                    'id': config.id,
                    'user_id': config.user_id,
                    'preferences': config.preferences,
                    'api_quota': config.api_quota,
                    'api_usage': config.api_usage,
                    'created_at': config.created_at.isoformat(),
                    'updated_at': config.updated_at.isoformat()
                }
            return None
    
    @staticmethod
    def increment_api_usage(user_id: str) -> bool:
        """
        Increment API usage counter for user.
        
        Args:
            user_id: User identifier
            
        Returns:
            True if successful, False if quota exceeded
        """
        with postgres_manager.get_session() as session:
            config = session.query(UserConfiguration).filter_by(user_id=user_id).first()
            if config:
                if config.api_usage < config.api_quota:
                    config.api_usage += 1
                    return True
                else:
                    return False
            return False


# Initialize databases on module import
def initialize_databases():
    """Initialize all database connections"""
    try:
        postgres_manager.initialize()
        mongodb_manager.initialize()
        logger.info("All database connections initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize databases: {e}")
        raise


# Cleanup function
def cleanup_databases():
    """Close all database connections"""
    postgres_manager.close()
    mongodb_manager.close()
    logger.info("All database connections closed")
