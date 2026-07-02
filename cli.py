import json
from inference import run_inference
import sys

def main():
    print("==================================================")
    print("⚽ V7 QUANTITATIVE ENGINE — LOCAL TERMINAL MODE")
    print("==================================================")
    print("Press Ctrl+C to exit at any time.\n")

    while True:
        try:
            home_team = input("Enter Home Team: ").strip()
            if not home_team: continue
            
            away_team = input("Enter Away Team: ").strip()
            if not away_team: continue

            venue_str = input("Venue Factor (0.0=Neutral, 1.0=Home Advantage) [default 0.3]: ").strip()
            venue_factor = float(venue_str) if venue_str else 0.3

            stage = input("Stage (group/round_of_16/quarter/semi/final) [default group]: ").strip()
            if not stage: stage = "group"

            print("\n[Running V7 Model Inference...]")
            
            result = run_inference(
                home_team=home_team,
                away_team=away_team,
                venue_factor=venue_factor,
                stage=stage
            )

            print("\n" + "-"*50)
            if result and 'error' not in result:
                print(f"Match: {home_team} vs {away_team}")
                print(f"Stage: {stage.capitalize()} (Venue Factor: {venue_factor})")
                print("-" * 50)
                print(f"🏠 {home_team} Win: {result.get('home_win_prob', 0):.1%}")
                print(f"🤝 Draw:          {result.get('draw_prob', 0):.1%}")
                print(f"✈️ {away_team} Win: {result.get('away_win_prob', 0):.1%}")
                
                # Check for Kelly/Value calculations if odds were provided
                # In this basic CLI, we aren't asking for odds, but they could be added.
            else:
                print(f"⚠️ Error: {result.get('error', 'Unknown error. Check if team names are spelled correctly.')}")
            
            print("-" * 50 + "\n")
            
        except KeyboardInterrupt:
            print("\n\nExiting engine. Goodbye!")
            sys.exit(0)
        except Exception as e:
            print(f"\n⚠️ Unexpected Error: {e}\n")

if __name__ == "__main__":
    main()
