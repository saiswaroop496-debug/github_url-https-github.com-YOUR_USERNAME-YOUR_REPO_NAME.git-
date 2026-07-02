import json
from inference import run_inference

result = run_inference(
    home_team='Argentina',
    away_team='France',
    venue_factor=0.5,
    stage='final'
)

print('\n' + '='*50)
print('⚽ LOCAL PREDICTION TEST')
print('='*50)
print('Match: Argentina vs France')
print('Stage: Final (Neutral Venue)')
print('-'*50)

if result and 'error' not in result:
    print(f"🏠 Argentina Win: {result.get('home_win_prob', 0):.1%}")
    print(f"🤝 Draw:          {result.get('draw_prob', 0):.1%}")
    print(f"✈️ France Win:    {result.get('away_win_prob', 0):.1%}")
else:
    print(f"Error: {result.get('error', 'Unknown error')}")
print('='*50 + '\n')
