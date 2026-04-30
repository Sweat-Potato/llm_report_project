
import json
with open('data/chunks/chunking_03_hybrid.json', encoding='utf-8') as f:
    data = json.load(f)
chunks = data['chunks']
semi = [c for c in chunks if c.get('sector') == '반도체']
print(f'반도체 청크 수: {len(semi)}개')
if semi:
    sizes = [c['char_count'] for c in semi]
    print(f'평균: {sum(sizes)//len(sizes)}자')
    print(f'최소: {min(sizes)}자')
    print(f'최대: {max(sizes)}자')
