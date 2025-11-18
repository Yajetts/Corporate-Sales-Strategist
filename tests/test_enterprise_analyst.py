"""Tests for Enterprise Analyst module"""

import pytest
import torch
from src.models.enterprise_analyst import EnterpriseAnalyst
from src.services.enterprise_analyst_service import EnterpriseAnalystService


class TestEnterpriseAnalyst:
    """Test cases for EnterpriseAnalyst model"""
    
    def test_model_initialization(self):
        """Test that model can be initialized"""
        # Use base BERT model for testing
        model = EnterpriseAnalyst(
            model_path="bert-base-uncased",
            device="cpu",
            max_length=128
        )
        
        assert model is not None
        assert model.device == "cpu"
        assert model.max_length == 128
        assert model.tokenizer is not None
        assert model.model is not None
    
    def test_model_initialization_with_defaults(self):
        """Test model initialization with default parameters"""
        model = EnterpriseAnalyst(model_path="bert-base-uncased")
        
        assert model is not None
        assert model.device in ["cpu", "cuda"]
        assert model.max_length == 512
    
    # Tokenization and Preprocessing Tests
    
    def test_preprocess_text_basic(self):
        """Test basic text preprocessing and tokenization"""
        model = EnterpriseAnalyst(
            model_path="bert-base-uncased",
            device="cpu",
            max_length=128
        )
        
        text = "TechCorp provides cloud-based software solutions."
        inputs = model._preprocess_text(text)
        
        # Verify output structure
        assert 'input_ids' in inputs
        assert 'attention_mask' in inputs
        
        # Verify tensor properties
        assert isinstance(inputs['input_ids'], torch.Tensor)
        assert isinstance(inputs['attention_mask'], torch.Tensor)
        assert inputs['input_ids'].shape[1] == 128  # max_length
        assert inputs['attention_mask'].shape[1] == 128
    
    def test_preprocess_text_with_whitespace(self):
        """Test preprocessing with leading/trailing whitespace"""
        model = EnterpriseAnalyst(
            model_path="bert-base-uncased",
            device="cpu",
            max_length=128
        )
        
        text = "   TechCorp provides solutions.   "
        inputs = model._preprocess_text(text)
        
        # Should handle whitespace correctly
        assert inputs['input_ids'].shape[1] == 128
        assert torch.sum(inputs['attention_mask']) > 0  # Has actual tokens
    
    def test_preprocess_text_long_input(self):
        """Test preprocessing with text exceeding max_length"""
        model = EnterpriseAnalyst(
            model_path="bert-base-uncased",
            device="cpu",
            max_length=64
        )
        
        # Create long text
        text = " ".join(["word"] * 200)  # Much longer than max_length
        inputs = model._preprocess_text(text)
        
        # Should truncate to max_length
        assert inputs['input_ids'].shape[1] == 64
        assert inputs['attention_mask'].shape[1] == 64
    
    def test_preprocess_text_special_characters(self):
        """Test preprocessing with special characters"""
        model = EnterpriseAnalyst(
            model_path="bert-base-uncased",
            device="cpu",
            max_length=128
        )
        
        text = "TechCorp™ provides AI/ML solutions @ $99.99!"
        inputs = model._preprocess_text(text)
        
        # Should handle special characters
        assert inputs['input_ids'].shape[1] == 128
        assert torch.sum(inputs['attention_mask']) > 0
    
    def test_extract_features(self):
        """Test feature extraction from text"""
        model = EnterpriseAnalyst(
            model_path="bert-base-uncased",
            device="cpu",
            max_length=128
        )
        
        text = "TechCorp provides cloud-based software solutions."
        embeddings = model._extract_features(text)
        
        # Verify embeddings shape (batch_size=1, hidden_size=768 for BERT-base)
        assert embeddings.shape[0] == 1
        assert embeddings.shape[1] == 768  # BERT-base hidden size
        assert isinstance(embeddings, torch.Tensor)
    
    # Model Inference Tests
    
    def test_analyze_company_basic(self):
        """Test basic company analysis"""
        model = EnterpriseAnalyst(
            model_path="bert-base-uncased",
            device="cpu",
            max_length=128
        )
        
        text = """
        TechCorp is a leading software company that provides cloud-based solutions
        for enterprise resource planning. Our platform helps businesses streamline
        their operations and improve efficiency.
        """
        
        result = model.analyze_company(text)
        
        # Verify result structure
        assert 'product_category' in result
        assert 'business_domain' in result
        assert 'value_proposition' in result
        assert 'key_features' in result
        assert 'confidence_scores' in result
        assert 'processing_time_ms' in result
        
        # Verify types
        assert isinstance(result['product_category'], str)
        assert isinstance(result['business_domain'], str)
        assert isinstance(result['value_proposition'], str)
        assert isinstance(result['key_features'], list)
        assert isinstance(result['confidence_scores'], dict)
        assert isinstance(result['processing_time_ms'], int)
        
        # Verify confidence scores structure
        assert 'category' in result['confidence_scores']
        assert 'domain' in result['confidence_scores']
        assert 0 <= result['confidence_scores']['category'] <= 1
        assert 0 <= result['confidence_scores']['domain'] <= 1
    
    def test_analyze_company_short_text(self):
        """Test analysis with short text input"""
        model = EnterpriseAnalyst(
            model_path="bert-base-uncased",
            device="cpu",
            max_length=128
        )
        
        text = "Software company."
        result = model.analyze_company(text)
        
        # Should still return valid structure
        assert 'product_category' in result
        assert 'business_domain' in result
        assert result['processing_time_ms'] > 0
    
    def test_analyze_company_long_text(self):
        """Test analysis with long text input"""
        model = EnterpriseAnalyst(
            model_path="bert-base-uncased",
            device="cpu",
            max_length=512
        )
        
        # Create a long text (simulating annual report excerpt)
        text = """
        TechCorp International is a global leader in enterprise software solutions.
        Founded in 2010, the company has grown to serve over 5,000 clients worldwide.
        Our flagship product, CloudERP, revolutionizes how businesses manage their
        operations, finance, and human resources. The platform integrates seamlessly
        with existing systems and provides real-time analytics and reporting capabilities.
        
        Our mission is to empower businesses through innovative technology solutions
        that drive efficiency and growth. We invest heavily in research and development,
        with over 30% of our revenue dedicated to innovation. Our team of 500+ engineers
        works continuously to enhance our products and develop new features.
        
        Key features include automated workflows, customizable dashboards, mobile access,
        and advanced security protocols. We serve clients across various industries
        including manufacturing, healthcare, retail, and financial services.
        """ * 3  # Repeat to make it longer
        
        result = model.analyze_company(text)
        
        # Should handle long text and return results
        assert 'product_category' in result
        assert 'business_domain' in result
        assert result['processing_time_ms'] > 0
    
    def test_analyze_company_empty_text(self):
        """Test that empty text raises error"""
        model = EnterpriseAnalyst(
            model_path="bert-base-uncased",
            device="cpu"
        )
        
        with pytest.raises(ValueError, match="Input text cannot be empty"):
            model.analyze_company("")
    
    def test_analyze_company_whitespace_only(self):
        """Test that whitespace-only text raises error"""
        model = EnterpriseAnalyst(
            model_path="bert-base-uncased",
            device="cpu"
        )
        
        with pytest.raises(ValueError, match="Input text cannot be empty"):
            model.analyze_company("   \n\t   ")
    
    def test_analyze_company_with_source_type(self):
        """Test analysis with source type"""
        model = EnterpriseAnalyst(
            model_path="bert-base-uncased",
            device="cpu"
        )
        
        text = "Company provides software solutions for businesses."
        result = model.analyze_company(text, source_type="product_summary")
        
        assert result['source_type'] == "product_summary"
    
    def test_analyze_company_annual_report(self):
        """Test analysis with annual report source type"""
        model = EnterpriseAnalyst(
            model_path="bert-base-uncased",
            device="cpu"
        )
        
        text = """
        Annual Report 2023: TechCorp achieved record revenue of $500M,
        representing 25% year-over-year growth. Our enterprise software
        division continues to be our primary revenue driver.
        """
        
        result = model.analyze_company(text, source_type="annual_report")
        
        assert result['source_type'] == "annual_report"
        assert 'product_category' in result
    
    def test_analyze_company_whitepaper(self):
        """Test analysis with whitepaper source type"""
        model = EnterpriseAnalyst(
            model_path="bert-base-uncased",
            device="cpu"
        )
        
        text = """
        This whitepaper explores how AI-powered analytics can transform
        business intelligence. Our platform leverages machine learning
        to provide predictive insights and automated decision-making.
        """
        
        result = model.analyze_company(text, source_type="whitepaper")
        
        assert result['source_type'] == "whitepaper"
        assert 'value_proposition' in result
    
    def test_analyze_company_processing_time(self):
        """Test that processing time is reasonable"""
        model = EnterpriseAnalyst(
            model_path="bert-base-uncased",
            device="cpu",
            max_length=128
        )
        
        text = "TechCorp provides software solutions."
        result = model.analyze_company(text)
        
        # Processing time should be positive and reasonable (< 5 seconds = 5000ms)
        assert result['processing_time_ms'] > 0
        assert result['processing_time_ms'] < 5000
    
    def test_batch_analyze(self):
        """Test batch analysis of multiple texts"""
        model = EnterpriseAnalyst(
            model_path="bert-base-uncased",
            device="cpu",
            max_length=128
        )
        
        texts = [
            "TechCorp provides software solutions.",
            "HealthCare Inc offers medical devices.",
            "RetailCo operates e-commerce platforms."
        ]
        
        results = model.batch_analyze(texts)
        
        assert len(results) == 3
        for result in results:
            assert 'product_category' in result
            assert 'business_domain' in result
    
    def test_batch_analyze_with_source_types(self):
        """Test batch analysis with different source types"""
        model = EnterpriseAnalyst(
            model_path="bert-base-uncased",
            device="cpu",
            max_length=128
        )
        
        texts = [
            "TechCorp provides software solutions.",
            "Annual report shows strong growth."
        ]
        source_types = ["product_summary", "annual_report"]
        
        results = model.batch_analyze(texts, source_types)
        
        assert len(results) == 2
        assert results[0]['source_type'] == "product_summary"
        assert results[1]['source_type'] == "annual_report"
    
    def test_analyze_company_with_numbers(self):
        """Test analysis with numerical data"""
        model = EnterpriseAnalyst(
            model_path="bert-base-uncased",
            device="cpu"
        )
        
        text = """
        TechCorp achieved $100M revenue in 2023 with 50% profit margins.
        We serve 1,000+ enterprise clients across 25 countries.
        """
        
        result = model.analyze_company(text)
        
        assert 'product_category' in result
        assert result['processing_time_ms'] > 0
    
    def test_analyze_company_with_technical_terms(self):
        """Test analysis with technical terminology"""
        model = EnterpriseAnalyst(
            model_path="bert-base-uncased",
            device="cpu"
        )
        
        text = """
        Our SaaS platform leverages microservices architecture, Kubernetes
        orchestration, and GraphQL APIs to deliver scalable cloud-native
        solutions with 99.99% uptime SLA.
        """
        
        result = model.analyze_company(text)
        
        assert 'product_category' in result
        assert 'business_domain' in result


class TestEnterpriseAnalystService:
    """Test cases for EnterpriseAnalystService"""
    
    def test_service_initialization(self):
        """Test that service can be initialized"""
        service = EnterpriseAnalystService()
        assert service is not None
    
    def test_service_singleton(self):
        """Test that service follows singleton pattern"""
        service1 = EnterpriseAnalystService()
        service2 = EnterpriseAnalystService()
        assert service1 is service2
    
    def test_get_model_info(self):
        """Test getting model information"""
        service = EnterpriseAnalystService()
        info = service.get_model_info()
        
        assert 'status' in info
        assert info['status'] in ['ready', 'not_initialized']
    
    def test_health_check(self):
        """Test health check"""
        service = EnterpriseAnalystService()
        health = service.health_check()
        
        assert 'status' in health
        assert health['status'] in ['healthy', 'unhealthy']
