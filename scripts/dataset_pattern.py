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

    if pattern_name == "pattern3":
        ratios = {
           "wiki_ja": 0.13,
           "dentaku": 2.0,
           "basic_math_dentaku": 0.035,
        }
        target_list = {
            "wiki_ja": "/storage6/dataset/pretrain/gen_experet/WIKI/raw/wikipedia_ja/merged_wikipedia_ja_16.0.jsonl",
            "dentaku": "/storage6/dataset/pretrain/gen_experet/dentaku/train_data_dentaku_2keta_only_add_1_3keta.jsonl",
            "basic_math_dentaku": "/storage6/fujisawa/add_ja_3x3_delete_5_pattern.jsonl",
        }
        return ratios, target_list
    
    if pattern_name == "pattern4":
        ratios = {
           "wiki_ja": 0.15,
           "dentaku": 1.0,
           "basic_math_dentaku": 0.0175,
        }
        target_list = {
            "wiki_ja": "/storage6/dataset/pretrain/gen_experet/WIKI/raw/wikipedia_ja/merged_wikipedia_ja_16.0.jsonl",
            "dentaku": "/storage6/dataset/pretrain/gen_experet/dentaku/train_data_dentaku_2keta_only_add_1_3keta.jsonl",
            "basic_math_dentaku": "/storage6/fujisawa/add_ja_3x3_delete_5_pattern.jsonl",
        }
        return ratios, target_list
    
    if pattern_name == "pattern5":
        ratios = {
           "wiki_ja": 0.133,
           "dentaku": 4.80,
        }
        target_list = {
            "wiki_ja": "/storage6/dataset/pretrain/gen_experet/WIKI/raw/wikipedia_ja/merged_wikipedia_ja_16.0.jsonl",
            "dentaku": "/storage6/dataset/pretrain/gen_experet/dentaku/train_data_dentaku_2keta_only_add_1_3keta.jsonl",
        }
        return ratios, target_list

    if pattern_name == "pattern6":
        ratios = {
           "wiki_ja": 0.01,
           "basic_math_dentaku": 0.01,
        }
        target_list = {
            "wiki_ja": "/storage6/dataset/pretrain/gen_experet/WIKI/raw/wikipedia_ja/merged_wikipedia_ja_16.0.jsonl",
            "basic_math_dentaku": "/storage6/fujisawa/add_ja_3x3_delete_5_pattern.jsonl",
        }
        return ratios, target_list

    ## 5/22
    if pattern_name == "pattern7":
        ratios = {
           "wiki_ja": 0.1,
           "dentaku": 2.0,
           "basic_math_dentaku": 0.12,
        }
        target_list = {
            "wiki_ja": "/storage6/dataset/pretrain/gen_experet/WIKI/raw/wikipedia_ja/merged_wikipedia_ja_16.0.jsonl",
            "dentaku": "/storage6/dataset/pretrain/gen_experet/dentaku/train_data_dentaku_2keta_only_add_1_3keta.jsonl",
            "basic_math_dentaku": "/storage6/fujisawa/add_ja_3x3_delete_5_pattern.jsonl",
        }
        return ratios, target_list
    
    if pattern_name == "pattern8":
        ratios = {
           "wiki_ja": 0.067,
           "dentaku": 2.0,
           "basic_math_dentaku": 0.18,
        }
        target_list = {
            "wiki_ja": "/storage6/dataset/pretrain/gen_experet/WIKI/raw/wikipedia_ja/merged_wikipedia_ja_16.0.jsonl",
            "dentaku": "/storage6/dataset/pretrain/gen_experet/dentaku/train_data_dentaku_2keta_only_add_1_3keta.jsonl",
            "basic_math_dentaku": "/storage6/fujisawa/add_ja_3x3_delete_5_pattern.jsonl",
        }
        return ratios, target_list
    
    if pattern_name == "pattern9":
        ratios = {
           "wiki_ja": 0.033,
           "dentaku": 2.00,
           "basic_math_dentaku": 0.24,
        }
        target_list = {
            "wiki_ja": "/storage6/dataset/pretrain/gen_experet/WIKI/raw/wikipedia_ja/merged_wikipedia_ja_16.0.jsonl",
            "dentaku": "/storage6/dataset/pretrain/gen_experet/dentaku/train_data_dentaku_2keta_only_add_1_3keta.jsonl",
            "basic_math_dentaku": "/storage6/fujisawa/add_ja_3x3_delete_5_pattern.jsonl",
        }
        return ratios, target_list
    
    if pattern_name == "pattern10":
        ratios = {
           "wiki_ja": 0.1,
           "dentaku": 10.0,
           "basic_math_dentaku": 0.12,
        }
        target_list = {
            "wiki_ja": "/storage6/dataset/pretrain/gen_experet/WIKI/raw/wikipedia_ja/merged_wikipedia_ja_16.0.jsonl",
            "dentaku": "/storage6/dataset/pretrain/gen_experet/dentaku/train_data_dentaku_2keta_only_add_1_3keta.jsonl",
            "basic_math_dentaku": "/storage6/fujisawa/add_ja_3x3_delete_5_pattern.jsonl",
        }
        return ratios, target_list

    if pattern_name == "pattern11":
        ratios = {
           "wiki_ja": 0.066,
           "dentaku": 20.00,
           "basic_math_dentaku": 0.05,
        }
        target_list = {
            "wiki_ja": "/storage6/dataset/pretrain/gen_experet/WIKI/raw/wikipedia_ja/merged_wikipedia_ja_16.0.jsonl",
            "dentaku": "/storage6/dataset/pretrain/gen_experet/dentaku/train_data_dentaku_2keta_only_add_1_3keta.jsonl",
            "basic_math_dentaku": "/storage6/fujisawa/add_ja_3x3_delete_5_pattern.jsonl",
        }
        return ratios, target_list

    if pattern_name == "pattern12":
        ratios = {
           "wiki_ja": 0.033,
           "dentaku": 2.00,
           "basic_math_dentaku": 0.180,
            "aozora": 0.3,
        }
        target_list = {
            "wiki_ja": "/storage6/dataset/pretrain/gen_experet/WIKI/raw/wikipedia_ja/merged_wikipedia_ja_16.0.jsonl",
            "dentaku": "/storage6/dataset/pretrain/gen_experet/dentaku/train_data_dentaku_2keta_only_add_1_3keta.jsonl",
            "basic_math_dentaku": "/storage6/fujisawa/add_ja_3x3_delete_5_pattern.jsonl",
            "aozora": "/storage6/corpus/category/BOOK/raw/JA/aozora/ja_book.jsonl",
        }
        return ratios, target_list