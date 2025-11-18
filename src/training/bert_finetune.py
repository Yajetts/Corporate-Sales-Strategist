"""BERT Fine-tuning Script for Enterprise Analyst Module

This script implements a training pipeline for fine-tuning BERT on domain-specific
corpora including annual reports, product summaries, and whitepapers.
"""

import os
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoConfig,
    get_linear_schedule_with_warmup,
    AdamW
)
import mlflow
import mlflow.pytorch
from tqdm import tqdm
import numpy as np

from src.utils.config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnterpriseDataset(Dataset):
    """
    Dataset class for loading and processing enterprise text data.
    Supports annual reports, product summaries, and whitepapers.
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 512,
        source_types: Optional[List[str]] = None
    ):
        """
        Initialize the dataset.
        
        Args:
            data_path: Path to data directory or file
            tokenizer: Hugging Face tokenizer
            max_length: Maximum sequence length
            source_types: List of source types to include
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.source_types = source_types or ['annual_report', 'product_summary', 'whitepaper']
        
        # Load data
        self.samples = self._load_data(data_path)
        logger.info(f"Loaded {len(self.samples)} samples from {data_path}")
    
    def _load_data(self, data_path: str) -> List[Dict]:
        """
        Load data from various sources.
        
        Args:
            data_path: Path to data directory or file
            
        Returns:
            List of data samples
        """
        import json
        
        samples = []
        data_path = Path(data_path)
        
        if data_path.is_file():
            # Load from single JSON file
            samples = self._load_json_file(data_path)
        elif data_path.is_dir():
            # Load from directory structure
            for source_type in self.source_types:
                source_dir = data_path / source_type
                if source_dir.exists():
                    samples.extend(self._load_from_directory(source_dir, source_type))
        else:
            raise ValueError(f"Data path does not exist: {data_path}")
        
        return samples
    
    def _load_json_file(self, file_path: Path) -> List[Dict]:
        """Load samples from a JSON file"""
        import json
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Expected format: list of dicts with 'text', 'category', 'domain', etc.
        return data if isinstance(data, list) else [data]
    
    def _load_from_directory(self, directory: Path, source_type: str) -> List[Dict]:
        """Load samples from a directory of text files"""
        samples = []
        
        for file_path in directory.glob('*.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()
            
            if text:
                samples.append({
                    'text': text,
                    'source_type': source_type,
                    'file_name': file_path.name
                })
        
        # Also check for JSON files
        for file_path in directory.glob('*.json'):
            samples.extend(self._load_json_file(file_path))
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing tokenized inputs and labels
        """
        sample = self.samples[idx]
        text = sample['text']
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Squeeze batch dimension
        item = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0)
        }
        
        # Add labels if available
        if 'category' in sample:
            item['category_label'] = torch.tensor(sample.get('category_id', 0), dtype=torch.long)
        
        if 'domain' in sample:
            item['domain_label'] = torch.tensor(sample.get('domain_id', 0), dtype=torch.long)
        
        return item


class BERTMultiTaskModel(nn.Module):
    """
    BERT model with multi-task classification heads for category and domain prediction.
    """
    
    def __init__(
        self,
        model_name: str,
        num_categories: int,
        num_domains: int,
        dropout: float = 0.1
    ):
        """
        Initialize the multi-task BERT model.
        
        Args:
            model_name: Hugging Face model name or path
            num_categories: Number of product categories
            num_domains: Number of business domains
            dropout: Dropout probability
        """
        super().__init__()
        
        self.bert = AutoModel.from_pretrained(model_name)
        self.config = self.bert.config
        hidden_size = self.config.hidden_size
        
        # Classification heads
        self.dropout = nn.Dropout(dropout)
        self.category_classifier = nn.Linear(hidden_size, num_categories)
        self.domain_classifier = nn.Linear(hidden_size, num_domains)
        
        logger.info(f"Initialized BERTMultiTaskModel with {num_categories} categories and {num_domains} domains")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        category_label: Optional[torch.Tensor] = None,
        domain_label: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            category_label: Category labels (optional, for training)
            domain_label: Domain labels (optional, for training)
            
        Returns:
            Dictionary containing logits and loss
        """
        # Get BERT outputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        
        # Classification
        category_logits = self.category_classifier(pooled_output)
        domain_logits = self.domain_classifier(pooled_output)
        
        result = {
            'category_logits': category_logits,
            'domain_logits': domain_logits,
            'embeddings': pooled_output
        }
        
        # Calculate loss if labels provided
        if category_label is not None and domain_label is not None:
            loss_fct = nn.CrossEntropyLoss()
            category_loss = loss_fct(category_logits, category_label)
            domain_loss = loss_fct(domain_logits, domain_label)
            
            # Combined loss
            total_loss = category_loss + domain_loss
            result['loss'] = total_loss
            result['category_loss'] = category_loss
            result['domain_loss'] = domain_loss
        
        return result


class BERTFineTuner:
    """
    Fine-tuning trainer for BERT Enterprise Analyst model.
    """
    
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_categories: int = 8,
        num_domains: int = 9,
        learning_rate: float = 2e-5,
        batch_size: int = 16,
        max_length: int = 512,
        num_epochs: int = 3,
        warmup_steps: int = 500,
        device: Optional[str] = None,
        output_dir: str = "models/bert",
        use_mlflow: bool = True
    ):
        """
        Initialize the fine-tuner.
        
        Args:
            model_name: Base BERT model name
            num_categories: Number of product categories
            num_domains: Number of business domains
            learning_rate: Learning rate
            batch_size: Training batch size
            max_length: Maximum sequence length
            num_epochs: Number of training epochs
            warmup_steps: Number of warmup steps
            device: Device to train on
            output_dir: Directory to save models
            use_mlflow: Whether to use MLflow logging
        """
        self.model_name = model_name
        self.num_categories = num_categories
        self.num_domains = num_domains
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_epochs = num_epochs
        self.warmup_steps = warmup_steps
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.output_dir = Path(output_dir)
        self.use_mlflow = use_mlflow
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = BERTMultiTaskModel(
            model_name=model_name,
            num_categories=num_categories,
            num_domains=num_domains
        )
        self.model.to(self.device)
        
        logger.info(f"Initialized BERTFineTuner on device: {self.device}")
    
    def prepare_data(
        self,
        train_data_path: str,
        val_data_path: Optional[str] = None,
        val_split: float = 0.1
    ) -> Tuple[DataLoader, Optional[DataLoader]]:
        """
        Prepare training and validation data loaders.
        
        Args:
            train_data_path: Path to training data
            val_data_path: Path to validation data (optional)
            val_split: Validation split ratio if val_data_path not provided
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        # Load training dataset
        train_dataset = EnterpriseDataset(
            data_path=train_data_path,
            tokenizer=self.tokenizer,
            max_length=self.max_length
        )
        
        # Split or load validation dataset
        if val_data_path:
            val_dataset = EnterpriseDataset(
                data_path=val_data_path,
                tokenizer=self.tokenizer,
                max_length=self.max_length
            )
        elif val_split > 0:
            # Split training data
            val_size = int(len(train_dataset) * val_split)
            train_size = len(train_dataset) - val_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                train_dataset,
                [train_size, val_size]
            )
        else:
            val_dataset = None
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2
        )
        
        val_loader = None
        if val_dataset:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=2
            )
        
        logger.info(f"Prepared data loaders: train={len(train_loader)} batches, val={len(val_loader) if val_loader else 0} batches")
        return train_loader, val_loader
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        checkpoint_every: int = 1000
    ):
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            checkpoint_every: Save checkpoint every N steps
        """
        # Setup optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        
        total_steps = len(train_loader) * self.num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=total_steps
        )
        
        # Initialize MLflow
        if self.use_mlflow:
            mlflow.set_tracking_uri(Config.MLFLOW_TRACKING_URI)
            mlflow.set_experiment(Config.MLFLOW_EXPERIMENT_NAME)
            
            run_name = f"bert_finetune_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            mlflow.start_run(run_name=run_name)
            
            # Log parameters
            mlflow.log_params({
                'model_name': self.model_name,
                'num_categories': self.num_categories,
                'num_domains': self.num_domains,
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size,
                'max_length': self.max_length,
                'num_epochs': self.num_epochs,
                'warmup_steps': self.warmup_steps
            })
        
        # Training loop
        global_step = 0
        best_val_loss = float('inf')
        
        for epoch in range(self.num_epochs):
            logger.info(f"Starting epoch {epoch + 1}/{self.num_epochs}")
            
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_category_loss = 0.0
            train_domain_loss = 0.0
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
            for batch_idx, batch in enumerate(progress_bar):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch)
                loss = outputs['loss']
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                # Track metrics
                train_loss += loss.item()
                if 'category_loss' in outputs:
                    train_category_loss += outputs['category_loss'].item()
                if 'domain_loss' in outputs:
                    train_domain_loss += outputs['domain_loss'].item()
                
                global_step += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': loss.item(),
                    'lr': scheduler.get_last_lr()[0]
                })
                
                # Log to MLflow
                if self.use_mlflow and global_step % 10 == 0:
                    mlflow.log_metrics({
                        'train_loss': loss.item(),
                        'learning_rate': scheduler.get_last_lr()[0]
                    }, step=global_step)
                
                # Save checkpoint
                if global_step % checkpoint_every == 0:
                    self._save_checkpoint(epoch, global_step)
            
            # Calculate average training loss
            avg_train_loss = train_loss / len(train_loader)
            avg_category_loss = train_category_loss / len(train_loader)
            avg_domain_loss = train_domain_loss / len(train_loader)
            
            logger.info(f"Epoch {epoch + 1} - Train Loss: {avg_train_loss:.4f}")
            
            # Validation phase
            if val_loader:
                val_loss = self._validate(val_loader)
                logger.info(f"Epoch {epoch + 1} - Val Loss: {val_loss:.4f}")
                
                # Log epoch metrics
                if self.use_mlflow:
                    mlflow.log_metrics({
                        'epoch_train_loss': avg_train_loss,
                        'epoch_train_category_loss': avg_category_loss,
                        'epoch_train_domain_loss': avg_domain_loss,
                        'epoch_val_loss': val_loss
                    }, step=epoch)
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self._save_model('best_model')
                    logger.info(f"Saved best model with val_loss: {val_loss:.4f}")
            else:
                if self.use_mlflow:
                    mlflow.log_metrics({
                        'epoch_train_loss': avg_train_loss,
                        'epoch_train_category_loss': avg_category_loss,
                        'epoch_train_domain_loss': avg_domain_loss
                    }, step=epoch)
        
        # Save final model
        self._save_model('final_model')
        logger.info("Training completed!")
        
        # End MLflow run
        if self.use_mlflow:
            # Log model to MLflow
            mlflow.pytorch.log_model(self.model, "model")
            mlflow.end_run()
    
    def _validate(self, val_loader: DataLoader) -> float:
        """
        Validate the model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                total_loss += outputs['loss'].item()
        
        return total_loss / len(val_loader)
    
    def _save_checkpoint(self, epoch: int, step: int):
        """Save training checkpoint"""
        checkpoint_dir = self.output_dir / f"checkpoint-epoch{epoch}-step{step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model and tokenizer
        self.model.bert.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)
        
        # Save classification heads
        torch.save({
            'category_classifier': self.model.category_classifier.state_dict(),
            'domain_classifier': self.model.domain_classifier.state_dict(),
            'epoch': epoch,
            'step': step
        }, checkpoint_dir / 'classifier_heads.pt')
        
        logger.info(f"Saved checkpoint to {checkpoint_dir}")
    
    def _save_model(self, name: str):
        """Save the trained model"""
        save_dir = self.output_dir / name
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model and tokenizer
        self.model.bert.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        
        # Save classification heads
        torch.save({
            'category_classifier': self.model.category_classifier.state_dict(),
            'domain_classifier': self.model.domain_classifier.state_dict(),
            'num_categories': self.num_categories,
            'num_domains': self.num_domains
        }, save_dir / 'classifier_heads.pt')
        
        logger.info(f"Saved model to {save_dir}")


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Fine-tune BERT for Enterprise Analyst")
    
    # Data arguments
    parser.add_argument('--train_data', type=str, required=True, help='Path to training data')
    parser.add_argument('--val_data', type=str, default=None, help='Path to validation data')
    parser.add_argument('--val_split', type=float, default=0.1, help='Validation split ratio')
    
    # Model arguments
    parser.add_argument('--model_name', type=str, default='bert-base-uncased', help='Base BERT model')
    parser.add_argument('--num_categories', type=int, default=8, help='Number of categories')
    parser.add_argument('--num_domains', type=int, default=9, help='Number of domains')
    
    # Training arguments
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--max_length', type=int, default=512, help='Max sequence length')
    parser.add_argument('--num_epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--warmup_steps', type=int, default=500, help='Warmup steps')
    parser.add_argument('--checkpoint_every', type=int, default=1000, help='Checkpoint frequency')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='models/bert', help='Output directory')
    parser.add_argument('--no_mlflow', action='store_true', help='Disable MLflow logging')
    
    args = parser.parse_args()
    
    # Initialize fine-tuner
    fine_tuner = BERTFineTuner(
        model_name=args.model_name,
        num_categories=args.num_categories,
        num_domains=args.num_domains,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        max_length=args.max_length,
        num_epochs=args.num_epochs,
        warmup_steps=args.warmup_steps,
        output_dir=args.output_dir,
        use_mlflow=not args.no_mlflow
    )
    
    # Prepare data
    train_loader, val_loader = fine_tuner.prepare_data(
        train_data_path=args.train_data,
        val_data_path=args.val_data,
        val_split=args.val_split
    )
    
    # Train
    fine_tuner.train(
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint_every=args.checkpoint_every
    )


if __name__ == '__main__':
    main()
