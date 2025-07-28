import argparse
import json
from src.models.jina_model import JinaEmbeddingModel
from transformers import AutoTokenizer

def main():
    parser = argparse.ArgumentParser(description='Generate embeddings with Jina Embeddings v4')
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-VL-3B-Instruct', 
                        help='Model path or name')
    parser.add_argument('--task', type=str, default='retrieval',
                        choices=['retrieval', 'text-matching', 'code'],
                        help='Task type')
    parser.add_argument('--input', type=str, required=True,
                        help='Input file (JSONL format)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output file for embeddings')
    parser.add_argument('--truncate_dim', type=int, default=2048,
                        help='Truncate dimension')
    parser.add_argument('--multivector', action='store_true',
                        help='Return multivector embeddings')
    
    args = parser.parse_args()
    
    # Load model
    model = JinaEmbeddingModel.from_pretrained(
        args.model, 
        trust_remote_code=True,
        pooling_strategy='mean'
    )
    
    # Process inputs
    results = []
    with open(args.input, 'r') as f:
        for line in f:
            item = json.loads(line.strip())
            
            if 'text' in item:
                embeddings = model.encode_text(
                    texts=[item['text']],
                    task=args.task,
                    truncate_dim=args.truncate_dim,
                    return_multivector=args.multivector
                )
            elif 'image' in item:
                embeddings = model.encode_image(
                    images=[item['image']],
                    task=args.task,
                    truncate_dim=args.truncate_dim,
                    return_multivector=args.multivector
                )
            else:
                continue
            
            # Store results
            result = {
                **item,
                'embedding': embeddings[0].cpu().numpy().tolist(),
                'dimension': args.truncate_dim
            }
            results.append(result)
    
    # Save embeddings
    with open(args.output, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\\n')

if __name__ == "__main__":
    main()
