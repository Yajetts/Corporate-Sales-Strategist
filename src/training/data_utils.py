"""Data loading utilities for BERT fine-tuning

Utilities for loading and preprocessing annual reports, product summaries,
and whitepapers for enterprise analysis.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DataSample:
    """Data sample structure"""
    text: str
    source_type: str
    category: Optional[str] = None
    domain: Optional[str] = None
    category_id: Optional[int] = None
    domain_id: Optional[int] = None
    metadata: Optional[Dict] = None


class DataLoader:
    """
    Utility class for loading enterprise data from various sources.
    """
    
    # Category and domain mappings
    CATEGORIES = [
        "Software", "Hardware", "Services", "Manufacturing",
        "Healthcare", "Finance", "Retail", "Other"
    ]
    
    DOMAINS = [
        "B2B", "B2C", "Enterprise", "SMB", "Consumer",
        "Industrial", "Technology", "Healthcare", "Financial Services"
    ]
    
    def __init__(self):
        """Initialize data loader"""
        self.category_to_id = {cat: idx for idx, cat in enumerate(self.CATEGORIES)}
        self.domain_to_id = {dom: idx for idx, dom in enumerate(self.DOMAINS)}
        self.id_to_category = {idx: cat for cat, idx in self.category_to_id.items()}
        self.id_to_domain = {idx: dom for dom, idx in self.domain_to_id.items()}
    
    def load_annual_reports(self, directory: str) -> List[DataSample]:
        """
        Load annual reports from directory.
        
        Args:
            directory: Path to directory containing annual reports
            
        Returns:
            List of DataSample objects
        """
        samples = []
        dir_path = Path(directory)
        
        if not dir_path.exists():
            logger.warning(f"Directory not found: {directory}")
            return samples
        
        # Load text files
        for file_path in dir_path.glob('*.txt'):
            text = self._read_text_file(file_path)
            if text:
                samples.append(DataSample(
                    text=text,
                    source_type='annual_report',
                    metadata={'file_name': file_path.name}
                ))
        
        # Load JSON files with structured data
        for file_path in dir_path.glob('*.json'):
            json_samples = self._load_json_with_labels(file_path, 'annual_report')
            samples.extend(json_samples)
        
        logger.info(f"Loaded {len(samples)} annual reports from {directory}")
        return samples
    
    def load_product_summaries(self, directory: str) -> List[DataSample]:
        """
        Load product summaries from directory.
        
        Args:
            directory: Path to directory containing product summaries
            
        Returns:
            List of DataSample objects
        """
        samples = []
        dir_path = Path(directory)
        
        if not dir_path.exists():
            logger.warning(f"Directory not found: {directory}")
            return samples
        
        # Load text files
        for file_path in dir_path.glob('*.txt'):
            text = self._read_text_file(file_path)
            if text:
                samples.append(DataSample(
                    text=text,
                    source_type='product_summary',
                    metadata={'file_name': file_path.name}
                ))
        
        # Load JSON files
        for file_path in dir_path.glob('*.json'):
            json_samples = self._load_json_with_labels(file_path, 'product_summary')
            samples.extend(json_samples)
        
        logger.info(f"Loaded {len(samples)} product summaries from {directory}")
        return samples
    
    def load_whitepapers(self, directory: str) -> List[DataSample]:
        """
        Load whitepapers from directory.
        
        Args:
            directory: Path to directory containing whitepapers
            
        Returns:
            List of DataSample objects
        """
        samples = []
        dir_path = Path(directory)
        
        if not dir_path.exists():
            logger.warning(f"Directory not found: {directory}")
            return samples
        
        # Load text files
        for file_path in dir_path.glob('*.txt'):
            text = self._read_text_file(file_path)
            if text:
                samples.append(DataSample(
                    text=text,
                    source_type='whitepaper',
                    metadata={'file_name': file_path.name}
                ))
        
        # Load JSON files
        for file_path in dir_path.glob('*.json'):
            json_samples = self._load_json_with_labels(file_path, 'whitepaper')
            samples.extend(json_samples)
        
        logger.info(f"Loaded {len(samples)} whitepapers from {directory}")
        return samples
    
    def load_all_sources(self, base_directory: str) -> List[DataSample]:
        """
        Load data from all source types.
        
        Args:
            base_directory: Base directory containing subdirectories for each source type
            
        Returns:
            List of all DataSample objects
        """
        base_path = Path(base_directory)
        samples = []
        
        # Load from each source type subdirectory
        annual_reports_dir = base_path / 'annual_reports'
        if annual_reports_dir.exists():
            samples.extend(self.load_annual_reports(str(annual_reports_dir)))
        
        product_summaries_dir = base_path / 'product_summaries'
        if product_summaries_dir.exists():
            samples.extend(self.load_product_summaries(str(product_summaries_dir)))
        
        whitepapers_dir = base_path / 'whitepapers'
        if whitepapers_dir.exists():
            samples.extend(self.load_whitepapers(str(whitepapers_dir)))
        
        logger.info(f"Loaded total of {len(samples)} samples from all sources")
        return samples
    
    def _read_text_file(self, file_path: Path) -> Optional[str]:
        """Read text from a file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()
            return text if text else None
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return None
    
    def _load_json_with_labels(self, file_path: Path, source_type: str) -> List[DataSample]:
        """Load JSON file with labels"""
        samples = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle both single object and list of objects
            if isinstance(data, dict):
                data = [data]
            
            for item in data:
                if 'text' not in item:
                    continue
                
                # Extract labels
                category = item.get('category')
                domain = item.get('domain')
                
                sample = DataSample(
                    text=item['text'],
                    source_type=source_type,
                    category=category,
                    domain=domain,
                    category_id=self.category_to_id.get(category) if category else None,
                    domain_id=self.domain_to_id.get(domain) if domain else None,
                    metadata=item.get('metadata', {})
                )
                samples.append(sample)
        
        except Exception as e:
            logger.error(f"Error loading JSON file {file_path}: {e}")
        
        return samples
    
    def save_to_json(self, samples: List[DataSample], output_path: str):
        """
        Save samples to JSON file.
        
        Args:
            samples: List of DataSample objects
            output_path: Output file path
        """
        data = []
        for sample in samples:
            item = {
                'text': sample.text,
                'source_type': sample.source_type,
                'category': sample.category,
                'domain': sample.domain,
                'category_id': sample.category_id,
                'domain_id': sample.domain_id,
                'metadata': sample.metadata or {}
            }
            data.append(item)
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(samples)} samples to {output_path}")
    
    def split_train_val_test(
        self,
        samples: List[DataSample],
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        shuffle: bool = True
    ) -> Tuple[List[DataSample], List[DataSample], List[DataSample]]:
        """
        Split samples into train, validation, and test sets.
        
        Args:
            samples: List of DataSample objects
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            shuffle: Whether to shuffle before splitting
            
        Returns:
            Tuple of (train_samples, val_samples, test_samples)
        """
        import random
        
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
        
        if shuffle:
            samples = samples.copy()
            random.shuffle(samples)
        
        n = len(samples)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        train_samples = samples[:train_end]
        val_samples = samples[train_end:val_end]
        test_samples = samples[val_end:]
        
        logger.info(f"Split data: train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}")
        
        return train_samples, val_samples, test_samples
    
    def get_statistics(self, samples: List[DataSample]) -> Dict:
        """
        Get statistics about the dataset.
        
        Args:
            samples: List of DataSample objects
            
        Returns:
            Dictionary with statistics
        """
        stats = {
            'total_samples': len(samples),
            'source_types': {},
            'categories': {},
            'domains': {},
            'avg_text_length': 0,
            'samples_with_labels': 0
        }
        
        total_length = 0
        
        for sample in samples:
            # Count source types
            source_type = sample.source_type
            stats['source_types'][source_type] = stats['source_types'].get(source_type, 0) + 1
            
            # Count categories
            if sample.category:
                stats['categories'][sample.category] = stats['categories'].get(sample.category, 0) + 1
            
            # Count domains
            if sample.domain:
                stats['domains'][sample.domain] = stats['domains'].get(sample.domain, 0) + 1
            
            # Track samples with labels
            if sample.category and sample.domain:
                stats['samples_with_labels'] += 1
            
            # Calculate text length
            total_length += len(sample.text)
        
        stats['avg_text_length'] = total_length / len(samples) if samples else 0
        
        return stats


def prepare_training_data(
    input_directory: str,
    output_directory: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1
):
    """
    Prepare training data from raw files.
    
    Args:
        input_directory: Directory containing raw data
        output_directory: Directory to save processed data
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
    """
    loader = DataLoader()
    
    # Load all data
    logger.info(f"Loading data from {input_directory}")
    samples = loader.load_all_sources(input_directory)
    
    if not samples:
        logger.error("No samples loaded. Please check the input directory structure.")
        return
    
    # Print statistics
    stats = loader.get_statistics(samples)
    logger.info(f"Dataset statistics: {json.dumps(stats, indent=2)}")
    
    # Split data
    train_samples, val_samples, test_samples = loader.split_train_val_test(
        samples,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio
    )
    
    # Save splits
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    
    loader.save_to_json(train_samples, str(output_path / 'train.json'))
    loader.save_to_json(val_samples, str(output_path / 'val.json'))
    loader.save_to_json(test_samples, str(output_path / 'test.json'))
    
    logger.info(f"Data preparation completed. Files saved to {output_directory}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare training data for BERT fine-tuning")
    parser.add_argument('--input_dir', type=str, required=True, help='Input directory with raw data')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for processed data')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Training set ratio')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='Validation set ratio')
    parser.add_argument('--test_ratio', type=float, default=0.1, help='Test set ratio')
    
    args = parser.parse_args()
    
    prepare_training_data(
        input_directory=args.input_dir,
        output_directory=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio
    )
