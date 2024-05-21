def get_dataset_pattern(pattern_name):
    if pattern_name == "pattern1":
        ratios = {
            "wiki_ja": 1,
            "dentaku": 1,
            "aozora": 1,
            "basic_math_dentaku": 0.01
        }
        target_list = {
                "wiki_ja": "/storage6/dataset/pretrain/gen_experet/WIKI/raw/wikipedia_ja/merged_wikipedia_ja_16.0.jsonl",
                "dentaku": "/storage6/dataset/pretrain/gen_experet/dentaku/train_data_dentaku_2keta_only_add_1_3keta.jsonl",
                "aozora": "/storage6/corpus/category/BOOK/raw/JA/aozora/ja_book.jsonl",
                "basic_math_dentaku": "/storage6/fujisawa/add_ja_3x3.jsonl",
                # "mc4": "/storage6/dataset/pretrain/router/1B/ja_mc4/merged_mc4_6.0.jsonl"
        }
        return ratios, target_list
    
    if pattern_name == "pattern2":
        ratios = {
           "wiki_ja": 1,
           "dentaku": 1,
           "aozora": 1,
           "basic_math_dentaku": 0.01,
           "wiki_en": 0.4,
        }
        target_list = {
            "wiki_ja": "/storage6/dataset/pretrain/gen_experet/WIKI/raw/wikipedia_ja/merged_wikipedia_ja_16.0.jsonl",
            "dentaku": "/storage6/dataset/pretrain/gen_experet/dentaku/train_data_dentaku_2keta_only_add_1_3keta.jsonl",
            "aozora": "/storage6/corpus/category/BOOK/raw/JA/aozora/ja_book.jsonl",
            "basic_math_dentaku": "/storage6/fujisawa/add_ja_3x3.jsonl",
            "wiki_en": "/storage6/dataset/pretrain/gen_experet/WIKI/raw/wikipedia_en/merged_expert_en_wikipedia_4.0.jsonl"
        }
        return ratios, target_list
    
    if pattern_name == "zoo":
        ratios = {
            "wiki_ja": 0.005,
            "aozora": 0.005,
            "wiki_en": 0.02,
        }
        target_list = {
            "wiki_ja": "/storage6/dataset/pretrain/gen_experet/WIKI/raw/wikipedia_ja/merged_wikipedia_ja_16.0.jsonl",
            "aozora": "/storage6/corpus/category/BOOK/raw/JA/aozora/ja_book.jsonl",
            "wiki_en": "/storage6/dataset/pretrain/gen_experet/WIKI/raw/wikipedia_en/merged_expert_en_wikipedia_4.0.jsonl"
        }
        return ratios, target_list