from src.trainer.jina_trainer import PredictionOutputWithSamples

print('Testing PredictionOutputWithSamples...')
try:
    output = PredictionOutputWithSamples(predictions=None, label_ids=None, metrics={}, num_samples=10)
    print(f'✅ Success! num_samples = {output.num_samples}')
    print(f'Has predictions: {hasattr(output, "predictions")}')
    print(f'Has label_ids: {hasattr(output, "label_ids")}') 
    print(f'Has metrics: {hasattr(output, "metrics")}')
except Exception as e:
    print(f'❌ Failed: {e}')
