"""Generate sample training data for BERT fine-tuning

This script generates synthetic sample data to help users get started with training.
In production, replace this with real enterprise data.
"""

import json
import random
from pathlib import Path
from typing import List, Dict
import argparse


# Sample data templates
CATEGORIES = [
    "Software", "Hardware", "Services", "Manufacturing",
    "Healthcare", "Finance", "Retail", "Other"
]

DOMAINS = [
    "B2B", "B2C", "Enterprise", "SMB", "Consumer",
    "Industrial", "Technology", "Healthcare", "Financial Services"
]

SAMPLE_TEXTS = {
    "annual_report": [
        "Our company has demonstrated strong growth in the {domain} sector, with revenue increasing by 25% year-over-year. "
        "We specialize in {category} solutions that help businesses streamline operations and improve efficiency. "
        "Our flagship products include advanced analytics platforms and automation tools.",
        
        "This year marked significant expansion in our {category} division. We serve primarily {domain} customers "
        "with innovative solutions that address critical business challenges. Our market share has grown to 15% "
        "in the competitive landscape.",
        
        "The company's focus on {category} innovation has positioned us as a leader in the {domain} market. "
        "We invested heavily in R&D, resulting in three new product launches and partnerships with major industry players."
    ],
    
    "product_summary": [
        "Our {category} solution is designed specifically for {domain} organizations. It provides comprehensive "
        "features including real-time analytics, automated workflows, and seamless integration with existing systems. "
        "Key benefits include 40% reduction in operational costs and improved decision-making capabilities.",
        
        "This {category} platform revolutionizes how {domain} companies manage their operations. With intuitive "
        "interfaces and powerful automation, users can accomplish tasks 3x faster. The solution scales from "
        "small teams to enterprise deployments.",
        
        "Introducing our next-generation {category} product for the {domain} sector. Built with cutting-edge "
        "technology, it offers unparalleled performance, security, and reliability. Trusted by over 500 companies worldwide."
    ],
    
    "whitepaper": [
        "The {domain} industry faces unprecedented challenges in adopting {category} technologies. This whitepaper "
        "explores best practices, implementation strategies, and ROI considerations. We analyze case studies from "
        "leading organizations and provide actionable recommendations.",
        
        "Digital transformation in {category} is reshaping the {domain} landscape. Organizations must adapt to "
        "remain competitive. This paper examines emerging trends, technology adoption patterns, and success factors "
        "based on research across 200+ companies.",
        
        "Understanding {category} solutions for {domain} enterprises requires a comprehensive approach. We discuss "
        "architecture patterns, integration strategies, and change management. Learn how industry leaders are "
        "achieving measurable business outcomes."
    ]
}


def generate_sample(source_type: str, category: str, domain: str) -> Dict:
    """Generate a single sample"""
    templates = SAMPLE_TEXTS[source_type]
    text = random.choice(templates).format(category=category, domain=domain)
    
    return {
        "text": text,
        "source_type": source_type,
        "category": category,
        "domain": domain,
        "category_id": CATEGORIES.index(category),
        "domain_id": DOMAINS.index(domain),
        "metadata": {
            "generated": True,
            "template_based": True
        }
    }


def generate_dataset(
    num_samples_per_type: int = 50,
    output_dir: str = "data/sample"
) -> List[Dict]:
    """Generate a complete sample dataset"""
    samples = []
    
    for source_type in ["annual_report", "product_summary", "whitepaper"]:
        for _ in range(num_samples_per_type):
            category = random.choice(CATEGORIES)
            domain = random.choice(DOMAINS)
            sample = generate_sample(source_type, category, domain)
            samples.append(sample)
    
    return samples


def save_samples_by_type(samples: List[Dict], output_dir: str):
    """Save samples organized by source type"""
    output_path = Path(output_dir)
    
    # Group by source type
    by_type = {}
    for sample in samples:
        source_type = sample["source_type"]
        if source_type not in by_type:
            by_type[source_type] = []
        by_type[source_type].append(sample)
    
    # Save each type
    for source_type, type_samples in by_type.items():
        type_dir = output_path / source_type
        type_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as JSON
        output_file = type_dir / f"{source_type}_samples.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(type_samples, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(type_samples)} samples to {output_file}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Generate sample training data")
    parser.add_argument(
        '--num_samples',
        type=int,
        default=50,
        help='Number of samples per source type'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/sample',
        help='Output directory'
    )
    
    args = parser.parse_args()
    
    print(f"Generating {args.num_samples * 3} sample documents...")
    samples = generate_dataset(
        num_samples_per_type=args.num_samples,
        output_dir=args.output_dir
    )
    
    print(f"Saving samples to {args.output_dir}...")
    save_samples_by_type(samples, args.output_dir)
    
    print(f"\nGeneration complete!")
    print(f"Total samples: {len(samples)}")
    print(f"\nTo prepare training data, run:")
    print(f"python -m src.training.data_utils --input_dir {args.output_dir} --output_dir data/processed")


if __name__ == '__main__':
    main()