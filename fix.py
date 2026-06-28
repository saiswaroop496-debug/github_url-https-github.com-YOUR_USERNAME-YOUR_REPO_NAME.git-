for f in ['train_test.py', 'telegram_bot.py', 'bet_tracker.py', r'features\rolling_features.py', 'api_server.py']:
    try:
        content = open(f, 'r', encoding='utf-8').read()
        content = content.replace('\"\\\"\"', '\"\"\"')
        open(f, 'w', encoding='utf-8').write(content)
    except:
        pass
