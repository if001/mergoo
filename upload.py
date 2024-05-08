import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--upload_repo_id", type=str)
    parser.add_argument("--tokenizer", type=str, default="team-sanai/unigram_32000")

    args = parser.parse_args()
    print("args: ", args)
    return args

def main():
    args = parse_arguments()
    model.push_to_hub(args.upload_repo_id)

if __name__ == "__main__":
    main()