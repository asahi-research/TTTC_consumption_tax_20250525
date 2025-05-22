import pandas as pd
from tqdm import tqdm
from openai import OpenAI  # 新しいOpenAIクライアント

def embedding(config):
    dataset = config['output_dir']
    path = f"outputs/{dataset}/embeddings.pkl"
    arguments = pd.read_csv(f"outputs/{dataset}/args.csv")
    
    # OpenAIクライアントを初期化（APIキーは環境変数から自動取得）
    client = OpenAI()
    
    embeddings = []
    for i in tqdm(range(0, len(arguments), 1000)):
        args = arguments["argument"].tolist()[i: i + 1000]
        
       response = client.embeddings.create(
            model="text-embedding-ada-002",  # モデル名を明示
            input=args
        )
        

        batch_embeddings = [item.embedding for item in response.data]
        embeddings.extend(batch_embeddings)
    
    df = pd.DataFrame(
        [
            {"arg-id": arguments.iloc[i]["arg-id"], "embedding": e}
            for i, e in enumerate(embeddings)
        ]
    )
    df.to_pickle(path)